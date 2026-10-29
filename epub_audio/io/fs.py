"""Project-relative assets and voice cache, independent of the launch directory."""
from pathlib import Path
import sys

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / 'input'
OUTPUT_DIR = ROOT / 'output'
