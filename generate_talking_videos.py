"""
Thin wrapper to keep the entrypoint at the repo root.

The actual implementation lives in scripts/generate_talking_videos.py as required
by the project instructions.
"""

from scripts.generate_talking_videos import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
