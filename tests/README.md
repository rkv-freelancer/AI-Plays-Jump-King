# Tests

Quick-iteration test suite for `JumpKingEnv_Multi` and the PBRS reward math.

## Running

From the project root:

```bash
# Fast: mocked tests only (no browser, ~1s)
python tests/main.py

# Full: include browser-backed smoke tests (~30s)
python tests/main.py --browser

# Equivalent direct pytest invocations:
pytest tests/                 # mocked only (default)
pytest tests/ -m browser      # browser only
pytest tests/ -m ""           # everything (-m "" overrides the default filter)
```

## Layout

```
tests/
├── conftest.py                  Pytest fixtures: driver (session-scoped, @browser),
│                                env_multi factory (function-scoped).
├── pytest.ini                   Test discovery + the @browser marker definition.
├── main.py                      One-command runner with --browser flag.
├── test_env_multi_contract.py   Shape / dtype / space contract for JumpKingEnv_Multi.
│                                @browser-marked.
├── test_env_multi_smoke.py      End-to-end: reset + 50 random steps, assert no crash
│                                and contract holds every step. @browser-marked.
└── test_pbrs_math.py            Pure-Python PBRS formula + telescoping check
                                 using a mock driver. No browser needed.
```

## Markers

| Marker | What it means |
|---|---|
| `@pytest.mark.browser` | Requires a running Chromium + the game server. Default `pytest` invocation skips these. |

## When to add a test here

- **Env contract change** (new obs dim, new termination reason) → contract test
- **Reward shaping change** (PBRS formula, terminal bonus) → math test (no browser needed)
- **JS↔Python bridge change** (new state field, action mapping) → smoke test with `@browser`

## Why mocked tests where possible

The Selenium round-trip dominates wall-clock per test (~30–50 ms each). A mocked-driver test runs the same Python code path in ~1 ms. For things like PBRS math, mocked tests catch regressions 30× faster — meaning you can run them on every save.

Browser tests stay opt-in (`-m browser`) so a regular `pytest` is fast enough to use as a development tight-loop.
