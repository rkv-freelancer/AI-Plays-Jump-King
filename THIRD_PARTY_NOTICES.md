# Third-party notices

This project's own code is MIT-licensed (see [`LICENSE`](LICENSE)). It also includes or depends on the third-party work below, which keeps its original terms.

## CleanRL

`models/PPO/ppo_rnd_envpool.py` and `models/PPO/ppo_rnd_envpool_SIL.py` are adapted from CleanRL's [`ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py).

```
MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Jump King game (`game/` submodule)

The `game/` folder is a git submodule pointing to [rkv-freelancer/Jump-King](https://github.com/rkv-freelancer/Jump-King), a fork of [Code Bullet's Jump-King](https://github.com/Code-Bullet/Jump-King). That code is not covered by this repository's license, and the upstream repository doesn't publish a license of its own.

*Jump King* and its artwork, level images, and sounds belong to Nexile. They are included in the upstream repository and are **not** covered by this repository's license.
