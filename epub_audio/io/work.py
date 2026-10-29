"""Offline context shared with child processes through the environment."""
import os


def is_offline():
    return os.environ.get('EBOOK_TTS_OFFLINE') == '1'
