"""Compatibility hooks for the portable voice catalogue event protocol.

Readable download progress is delivered through the catalogue message callback.
"""


def emit(*args, **kwargs):
    # The catalogue also reports readable progress through its message callback.
    pass


def progress_stage(*args, **kwargs):
    pass
