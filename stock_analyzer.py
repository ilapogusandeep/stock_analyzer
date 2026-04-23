"""Backward-compat shim. Real code lives in stockiq.core.analyzer."""

from stockiq.core.analyzer import *  # noqa: F401,F403
from stockiq.core.analyzer import main  # explicit re-export for setup.py console_script

if __name__ == "__main__":
    main()
