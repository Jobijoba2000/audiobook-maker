"""Duration labels for chapter files and book folders."""


def format_duration_minutes_seconds(seconds):
    return f'{int(seconds // 60):02d}.{int(seconds % 60):02d}'


def format_duration_full(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f'{hours}h{minutes:02d}' if hours else f'{minutes}m'
