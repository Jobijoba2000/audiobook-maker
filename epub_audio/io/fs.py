"""Project-relative assets and voice cache, independent of the launch directory."""
from pathlib import Path
import sys
import time

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / 'input'
OUTPUT_DIR = ROOT / 'output'


def retry_windows_io(operation, *args):
    """Allow brief Windows sharing/access locks to clear; preserve real failures."""
    for attempt in range(6):
        try:
            return operation(*args)
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)
