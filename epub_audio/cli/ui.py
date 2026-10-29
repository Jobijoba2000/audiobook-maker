"""Terminal presentation, independent of audio generation."""
import os
import shutil
import sys
import threading

from epub_audio.core.tts_progress import ChapterProgress, elapsed_label


def _terminal_controls(stream):
    """Enable cursor controls in Windows consoles; redirected logs stay plain."""
    if not stream.isatty():
        return False
    if os.name != 'nt':
        return True
    try:
        import ctypes
        from ctypes import wintypes
        import msvcrt
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        mode = wintypes.DWORD()
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, ValueError, AttributeError):
        pass
    return False


class ConsoleOutput:
    def __init__(self, stream=None, *, terminal_controls=None):
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = self.stream.isatty()
        self.controls = (self.interactive and _terminal_controls(self.stream)
                         if terminal_controls is None else self.interactive and terminal_controls)
        self.last_progress = None
        self.width = 0
        self._lock = threading.RLock()
        self._chapters = {}
        self._frozen = 0
        self._live_rows = 0
        self._elapsed = None
        self._logged = {}

    def _clear_live(self):
        if self._live_rows:
            count = self._live_rows
            self.stream.write(f'\x1b[{count}F' + '\x1b[2K\n' * count + f'\x1b[{count}F')
            self._live_rows = 0

    def _render_chapters(self):
        size = shutil.get_terminal_size(fallback=(80, 24))
        width = max(1, size.columns - 1)
        self._clear_live()
        permanent = []
        while (next_row := self._chapters.get(self._frozen + 1)) is not None and next_row.validated:
            permanent.append(next_row.row_text)
            self._frozen += 1
        live = [row.row_text for index, row in sorted(self._chapters.items()) if index > self._frozen]
        # Keep the editable area inside the viewport, including on small
        # terminals. Completed lines accumulate in scrollback normally.
        room = max(2, size.lines - 1)
        if len(live) + 1 > room:
            visible = room - 2
            hidden = len(live) - visible
            live = live[:visible] + [f'... {hidden} autres chapitres en attente ou en cours']
        if self._elapsed is not None:
            live.append(f'Durée TTS : {elapsed_label(self._elapsed)}')
        self.stream.write(''.join(line[:width] + '\n' for line in permanent + live))
        self._live_rows = len(live)
        self.stream.flush()

    def progress(self, text):
        with self._lock:
            if isinstance(text, ChapterProgress):
                if not self._chapters and self.last_progress is not None:
                    if self.interactive:
                        self.stream.write('\r' + ' ' * self.width + '\r')
                    self.last_progress = None
                    self.width = 0
                self._chapters[text.chapter] = text
                if text.elapsed is not None:
                    self._elapsed = text.elapsed
                if self.controls:
                    self._render_chapters()
                elif self.interactive:
                    print('\r' + text.ljust(self.width), end='', file=self.stream, flush=True)
                    self.width = len(text)
                    self.last_progress = text
                return
            if self.interactive:
                print('\r' + text.ljust(self.width), end='', file=self.stream, flush=True)
                self.width = len(text)
            self.last_progress = text

    def _finish_chapters(self):
        self._clear_live()
        for index, row in sorted(self._chapters.items()):
            if index > self._frozen:
                print(row.row_text, file=self.stream)
        self._chapters.clear()
        self._logged.clear()
        self._frozen = 0
        self._elapsed = None

    def message(self, text):
        with self._lock:
            if self._chapters:
                final = text.startswith('Durée génération TTS :')
                if self.controls:
                    self._clear_live()
                    if final:
                        self._finish_chapters()
                    print(text, file=self.stream, flush=True)
                    if not final:
                        self._render_chapters()
                    return
                if self.last_progress is not None and self.interactive:
                    print(file=self.stream)
                self.last_progress = None
                self.width = 0
                # No cursor escapes or timer spam in redirected logs.
                for index, row in sorted(self._chapters.items()):
                    state = (row.done, row.total, row.validated, row.retrying)
                    if self._logged.get(index) != state:
                        print(row.row_text, file=self.stream)
                        self._logged[index] = state
                if final:
                    self._chapters.clear()
                    self._logged.clear()
                    self._elapsed = None
                print(text, file=self.stream, flush=True)
                return
            if self.last_progress is not None:
                print('' if self.interactive else self.last_progress, file=self.stream, flush=True)
                self.last_progress = None
                self.width = 0
            print(text, file=self.stream, flush=True)
