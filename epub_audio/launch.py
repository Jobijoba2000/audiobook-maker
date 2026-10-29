"""Shared error handling for the module entry point and legacy launcher."""
from multiprocessing import freeze_support
import sys
from epub_audio.cli.main import main


def run(argv=None, default_directory=None):
    freeze_support()
    try:
        return main(argv, default_directory=default_directory)
    except KeyboardInterrupt:
        print('\nAnnulé par l’utilisateur.')
        return 130
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        print(f'Erreur : {detail}', file=sys.stderr)
        return 1
