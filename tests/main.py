"""
One-command test runner.

Usage:
    python tests/main.py                  # mocked tests only (no browser, ~1s)
    python tests/main.py --browser        # include browser smoke tests (~30s)
    python tests/main.py --all            # everything regardless of markers
    python tests/main.py --file test_pbrs_math.py     # one test file
    python tests/main.py -k "telescoping"             # by name pattern

All unrecognised args are forwarded to pytest, so you can also pass --pdb,
--lf (last failed), -x (stop on first fail), etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent


def main() -> int:
    """Build the pytest argv and exec it. Returns pytest's exit code."""
    argv = list(sys.argv[1:])

    # Custom flags. Extract them, hand the rest to pytest.
    include_browser = False
    include_all     = False
    target_file: str | None = None

    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--browser":
            include_browser = True
        elif arg == "--all":
            include_all = True
        elif arg == "--file":
            if i + 1 >= len(argv):
                print("error: --file requires a filename")
                return 2
            target_file = argv[i + 1]
            i += 1
        else:
            cleaned.append(arg)
        i += 1

    # Resolve which tests to run.
    if target_file:
        target = TESTS_DIR / target_file
        if not target.exists():
            print(f"error: {target} not found")
            return 2
        test_target = str(target)
    else:
        test_target = str(TESTS_DIR)

    # Marker selection. pytest.ini default is -m "not browser"; we override per flag.
    marker_args: list[str] = []
    if include_all:
        marker_args = ["-m", ""]
    elif include_browser:
        # Run both mocked AND browser tests (i.e. everything tagged or untagged)
        marker_args = ["-m", ""]
    # Default: pytest.ini's "-m not browser" stays in effect, no override needed.

    pytest_args = [test_target, *marker_args, *cleaned]

    print(f"[tests/main.py] running: pytest {' '.join(pytest_args)}")
    print()
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
