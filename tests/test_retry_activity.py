"""A segment retry emits one concise line, even when it needs smaller pieces."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave

from epub_audio.core.tts_retry import retry_segment, synthesize_with_fallback


def write_pcm(text, destination):
    with wave.open(str(destination), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\xe8\x03' * 2400)
    return Path(destination)


class RetryActivityTests(unittest.TestCase):


    def test_split_retry_keeps_original_location_and_every_word_and_pcm_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            chunks = Path(folder) / '.segments-4'
            chunks.mkdir()
            output = chunks / '000006.wav'
            messages, calls, accepted = [], [], []
            text = 'Première phrase du passage, suffisamment longue pour être conservée. Deuxième phrase complète.'

            def synthesize(part, destination):
                calls.append(part)
                if len(calls) == 1:
                    raise RuntimeError('Limite globale atteinte')
                accepted.append(part)
                return write_pcm(part, destination)

            engine = SimpleNamespace(message=messages.append, synthesize=synthesize)
            synthesize_with_fallback(engine, text, output)
            self.assertEqual(' '.join(accepted), text)
            with wave.open(str(output), 'rb') as audio:
                self.assertEqual(audio.readframes(audio.getnframes()), b'\xe8\x03' * 4800)
            self.assertTrue(all(message.startswith('Chapitre 4, segment 7') for message in messages))
            self.assertEqual(len(messages), 1)
            self.assertIn('réessai en morceaux', messages[0])
            self.assertEqual(list(chunks.iterdir()), [output])

    def test_individual_retry_announces_one_line_with_location_and_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            chunks = Path(folder) / '.segments-12'
            chunks.mkdir()
            output = chunks / '000003.wav'
            messages = []
            engine = SimpleNamespace(message=messages.append, synthesize=write_pcm, settings={'max_new_tokens': 2048})
            text = 'Une phrase\nsur deux lignes.'
            retry_segment(engine, text, output, 'Limite estimée atteinte (64 étapes).')
            self.assertIn('Chapitre 12, segment 4', messages[0])
            self.assertIn('64 étapes', messages[0])
            self.assertEqual(len(messages), 1)
            self.assertTrue(output.is_file())

    def test_permanent_failure_is_announced_without_success_or_suppressed_error(self):
        messages = []
        original = RuntimeError('GPU unavailable')
        def fail(text, destination):
            raise original
        engine = SimpleNamespace(message=messages.append, synthesize=fail)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / '000000.wav'
            with self.assertRaises(RuntimeError) as error:
                retry_segment(engine, 'Oui.', output, 'Échec du groupe.')
            self.assertIs(error.exception, original)
            self.assertFalse(output.exists())
        self.assertEqual(len(messages), 1)
        self.assertIn('Réessai individuel', messages[0])
        self.assertFalse(any('audio validé' in message for message in messages))

    def test_fast_successful_normal_segments_remain_quiet(self):
        messages = []
        engine = SimpleNamespace(message=messages.append, synthesize=write_pcm)
        with tempfile.TemporaryDirectory() as folder:
            synthesize_with_fallback(engine, 'Un passage ordinaire.', Path(folder) / '000000.wav')
        self.assertEqual(messages, [])
