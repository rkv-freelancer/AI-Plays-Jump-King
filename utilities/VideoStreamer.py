import os

class VideoStreamer:
    """
    Streams rgb_array frames straight to an mp4 via imageio, so a multi-minute
    recording never holds thousands of full-res frames in RAM.

    A 1200 x 950 x 3 frame is ~3.4 MB; a 2-minute clip at 30 fps is 3,600 frames ≈
    12 GB if buffered in a list. Streaming to disk keeps memory flat regardless
    of clip length.

    Lifecycle:
        rec.start(path)   open the writer (begins a new clip)
        rec.add(frame)    append one frame  (no-op if not active)
        rec.is_full       True once target_frames have been written
        rec.finish(...)   close writer, optionally upload to W&B, optionally
                          delete the local file, reset to idle

    `is_active` lets the rollout loop skip the (slow) env.render() Selenium
    round-trip on steps where no recorder wants a frame.
    """

    def __init__(self, fps: int, target_frames: int, wandb_key: str, track: bool):
        self.fps           = fps
        self.target_frames = target_frames
        self.wandb_key     = wandb_key
        self.track         = track
        self._writer       = None
        self._path         = None
        self._count        = 0

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    @property
    def is_full(self) -> bool:
        return self._writer is not None and self._count >= self.target_frames

    def start(self, path: str) -> None:
        import imageio
        self._writer = imageio.get_writer(path, fps=self.fps, macro_block_size=1)
        self._path   = path
        self._count  = 0

    def add(self, frame) -> None:
        if self._writer is not None:
            self._writer.append_data(frame)
            self._count += 1

    def finish(self, log, global_step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as e:
            log.warning("[video] writer close failed for %s: %s", self._path, e)
        n, path      = self._count, self._path
        self._writer = None
        self._path   = None
        self._count  = 0
        if n == 0:
            return
        log.info("[video] wrote %d frames → %s (%.1f s @ %d fps)",
                 n, path, n / self.fps, self.fps)
        if self.track:
            try:
                import wandb
                wandb.log(
                    {self.wandb_key: wandb.Video(path, fps=self.fps, format="mp4")},
                    step=global_step,
                )
                log.info("[video] uploaded %s to W&B at step %d", self.wandb_key, global_step)
            except Exception as e:
                log.warning("[video] W&B upload failed for %s: %s", path, e)
            # W&B has copied it into its media dir — drop the staging file.
            try:
                # When not tracking, KEEP the local mp4 so it can be watched directly.
                os.remove(path)
            except OSError as e:
                
                log.warning("[video] could not delete %s: %s", path, e)