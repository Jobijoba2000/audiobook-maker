"""Navigation boundaries, metadata names, automatic fallback and intact narration."""
from pathlib import Path
from unittest.mock import Mock, patch
import io
import struct
import tempfile
import unittest
import wave
from zipfile import ZipFile

from ebooklib import epub
from epub_audio.io.epub import read_book
from epub_audio.io.naming import book_basename
from epub_audio.core.models import BookMetadata
from epub_audio.core.pipeline import process_book
from epub_audio.core.text import plan_segments
from epub_audio.core.tts_retry import synthesize_with_fallback
from epub_audio.io.epub_structure import read_document


class ShortVoice:
    def prepare(self):
        pass

    def synthesize(self, text, destination):
        with wave.open(str(destination), 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(struct.pack('<h', 1000) * 2400)
        return Path(destination)


class LimitedVoice(ShortVoice):
    def __init__(self):
        self.accepted = []
        self.requests = []

    def synthesize(self, text, destination):
        self.requests.append(text)
        if len(text) > 120:
            raise RuntimeError('Synthetic paragraph failure')
        self.accepted.append(text)
        return super().synthesize(text, destination)


def make_epub(root, documents, toc=(), *, ncx=False, metadata=True):
    book = epub.EpubBook()
    book.set_identifier('structure-test')
    book.set_title('Titre du livre')
    book.set_language('fr')
    if metadata:
        book.add_author('Joe Haldeman', file_as='Haldeman, Joe')
        book.add_metadata('DC', 'date', '1975-01-04')
        book.add_metadata('OPF', 'meta', '', {'name': 'calibre:series', 'content': 'Cycle'})
        book.add_metadata('OPF', 'meta', '', {'name': 'calibre:series_index', 'content': '2'})
    items = []
    for filename, html in documents:
        item = epub.EpubHtml(title=filename, file_name=filename, lang='fr', media_type='application/xhtml+xml')
        item.content = html
        book.add_item(item)
        items.append(item)
    book.spine = items
    book.toc = tuple(epub.Link(href, title, f'chapter-{index}') for index, (href, title) in enumerate(toc))
    if ncx:
        book.add_item(epub.EpubNcx())
    else:
        book.add_item(epub.EpubNav())
    path = root / 'original.epub'
    epub.write_epub(str(path), book)
    return path


class StructureTests(unittest.TestCase):
    def test_verne_xml_note_calls_and_backlinks_are_not_narrated(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [
                ('OPS/main0.xml', '<h1>Premier chapitre</h1>'
                 '<p>West-Point<a class="footnote" href="footnotes.xml#footnote_1" '
                 'id="footnote_1_call">[1]</a> ; il reste 42 jours.</p>'),
                ('OPS/footnotes.xml', '<div class="footnote" id="footnote_1">'
                 '<a class="marker" href="main0.xml#footnote_1_call">[1]</a> '
                 '<span class="isfootnote">École militaire des États-Unis.</span></div>')])
            original = source.read_bytes()
            book = read_book(source)
            paragraphs = [text for chapter in book.chapters for text in chapter.paragraphs]
            self.assertIn('West-Point ; il reste 42 jours.', paragraphs)
            self.assertIn('École militaire des États-Unis.', paragraphs)
            self.assertNotIn('[1]', ' '.join(paragraphs))

            voice = LimitedVoice()
            voice.serial_gpu = True
            result = process_book(source, voice, output_dir=Path(folder) / 'audio',
                                  audio_format='wav', message=lambda _: None)
            spoken = ' '.join(voice.accepted)
            self.assertNotIn('[1]', spoken)
            self.assertIn('quarante-deux jours', spoken)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(next(result.glob('*.epub')).read_bytes(), original)

    def test_epub_semantics_and_linked_superscript_note_markers_are_removed(self):
        markers = [
            '<a epub:type="noteref" href="notes.xhtml#n1">[1]</a>',
            '<a role="doc-noteref" href="#n1">voir la note</a>',
            '<a role="doc-backlink" href="#ref1">retour au texte</a>',
            '<a epub:type="backlink" href="#ref1">↩</a>',
            '<a class="footnote-ref" href="#n1">1</a>',
            '<a href="#fn1"><sup>1</sup></a>',
            '<a href="notes.xhtml#n1">[1]</a>',
            '<sup><a href="#n1">[1]</a></sup>',
            '<a href="#n1"><sup>[1]</sup></a>',
            '<a class="endnote-ref" href="#n1">†</a>',
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                item = epub.EpubHtml(file_name='chapter.xhtml')
                item.content = f'<h1>Titre{marker}</h1><p>L’{marker}univers reste vaste.</p>'
                epub.EpubBook().add_item(item)
                document = read_document(item, BookMetadata())
                self.assertEqual(document.title, 'Titre')
                self.assertEqual([block.text for block in document.blocks], ['Titre', 'L’univers reste vaste.'])

    def test_ordinary_brackets_numbers_links_and_exponents_are_preserved(self):
        item = epub.EpubHtml(file_name='chapter.xhtml')
        item.content = ('<h1>Chapitre 1</h1><p>En 1865, [1] homme mesure 2 m<sup>2</sup>. '
                        '<a href="#chapter2">[2]</a> '
                        '<sup><a href="#equation2">2</a></sup> '
                        '<a href="https://example.org/#note1">[3]</a> '
                        '<a href="http://[">[4]</a> '
                        '<a href="notes.xhtml#n1">une explication</a>.</p>')
        epub.EpubBook().add_item(item)
        document = read_document(item, BookMetadata())
        self.assertEqual(document.blocks[1].text,
                         'En 1865, [1] homme mesure 2 m2. [2] 2 [3] [4] une explication.')

    def test_two_chapters_in_one_html_are_split_at_navigation_anchors(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [('Text/all.xhtml',
                '<h1 id="a">Un</h1><p>Premier récit.</p><h1 id="b">Deux</h1><p>Second récit.</p>')],
                [('Text/all.xhtml#a', 'Un'), ('Text/all.xhtml#b', 'Deux')])
            book = read_book(source)
            self.assertEqual(book.mode, 'table des matières')
            self.assertEqual([chapter.paragraphs for chapter in book.chapters],
                             [['Un', 'Premier récit.'], ['Deux', 'Second récit.']])

    def test_ncx_parts_and_multiple_html_continuations_become_real_chapters(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [
                ('titlepage.xhtml', '<p>Titre du livre</p><p>Joe Haldeman</p>'),
                ('copyright.xhtml', '<p>Copyright 1975. Tous droits réservés.</p>'),
                ('part.xhtml', '<p>Première partie</p>'),
                ('one.xhtml', '<h2>1</h2><p>Premier récit.</p>'),
                ('continuation.xhtml', '<p>La suite du même chapitre.</p>'),
                ('two.xhtml', '<h2>2</h2><p>Second récit.</p>'),
                ('end.xhtml', '<h1>Épilogue</h1><p>Texte final.</p>')],
                [('part.xhtml', 'Première partie'), ('one.xhtml', '1'),
                 ('two.xhtml', '2'), ('end.xhtml', 'Épilogue')], ncx=True)
            book = read_book(source)
            self.assertFalse(book.fallback_reason)
            self.assertEqual(len(book.chapters), 3)
            self.assertEqual(book.chapters[0].paragraphs,
                             ['Première partie', '1', 'Premier récit.', 'La suite du même chapitre.'])
            self.assertEqual(book.chapters[-1].paragraphs, ['Épilogue', 'Texte final.'])
            self.assertEqual(book.excluded_pages, 2)

    def test_headings_work_when_navigation_is_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [('all.xhtml',
                '<h1>Premier</h1><p>Un récit.</p><h1>Second</h1><p>Une suite.</p>')])
            book = read_book(source)
            self.assertEqual((book.mode, len(book.chapters)), ('titres', 2))

    def test_damaged_ncx_offers_fallback_and_releases_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [('one.xhtml', '<h1>Un</h1><p>Récit intact.</p>')],
                               [('one.xhtml', 'Un')], ncx=True)
            with ZipFile(source) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            name = next(name for name in files if name.endswith('.ncx'))
            files[name] = b'<ncx>invalid'
            with ZipFile(source, 'w') as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
            book = read_book(source)
            self.assertEqual(book.mode, 'legacy')
            self.assertIn('NCX', book.fallback_reason)
            self.assertIn('Récit intact.', book.chapters[0].paragraphs)
            # Windows rejects this rename if the reader has left an open handle.
            source.rename(source.with_name('renamed.epub'))

    def test_unlinked_preface_triggers_fallback_instead_of_losing_text(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [
                ('preface.xhtml', '<p>Une préface à ne jamais supprimer.</p>'),
                ('chapter.xhtml', '<h1>Chapitre</h1><p>Un récit.</p>')], [('chapter.xhtml', 'Chapitre')])
            book = read_book(source)
            self.assertEqual(book.mode, 'legacy')
            self.assertIn('précède', book.fallback_reason)
            self.assertEqual(book.chapters[0].paragraphs, ['Une préface à ne jamais supprimer.'])

    def test_a_narrative_mention_of_copyright_is_not_a_technical_page(self):
        with tempfile.TemporaryDirectory() as folder:
            text = 'La question du copyright préoccupait le capitaine. Il reprit son voyage.'
            source = make_epub(Path(folder), [('one.xhtml', f'<p>{text}</p>'),
                ('two.xhtml', '<h1>Deux</h1><p>Suite du voyage.</p>')], [('two.xhtml', 'Deux')])
            book = read_book(source)
            self.assertTrue(book.fallback_reason)
            self.assertEqual(book.chapters[0].paragraphs, [text])

    def test_bad_and_reversed_navigation_require_explicit_fallback(self):
        for toc in ([('one.xhtml#missing', 'Un')], [('missing.xhtml', 'Un')],
                    [('two.xhtml', 'Deux'), ('one.xhtml', 'Un')]):
            with self.subTest(toc=toc), tempfile.TemporaryDirectory() as folder:
                source = make_epub(Path(folder), [('one.xhtml', '<h1>Un</h1><p>Début.</p>'),
                    ('two.xhtml', '<h1>Deux</h1><p>Fin.</p>')], toc)
                book = read_book(source)
                self.assertEqual(book.mode, 'legacy')
                self.assertTrue(book.fallback_reason)
                self.assertEqual(len(book.chapters), 2)

    def test_inline_ids_and_nested_technical_sections_do_not_hide_or_fragment_story(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_epub(Path(folder), [('one.xhtml',
                '<h1>Un</h1><p>L’<span id="word">univers</span> est vaste.</p>'
                '<section epub:type="colophon"><p>Note conservée.</p></section>')], [('one.xhtml', 'Un')])
            book = read_book(source)
            self.assertFalse(book.fallback_reason)
            self.assertEqual(book.chapters[0].paragraphs,
                             ['Un', 'L’univers est vaste.', 'Note conservée.'])

    def test_legacy_fallback_exports_all_documents_without_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make_epub(root, [('one.xhtml', '<p>Premier récit sans titres.</p>'),
                                      ('two.xhtml', '<p>Suite du récit sans titres.</p>')])
            original = source.read_bytes()
            messages = []
            with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
                result = process_book(source, ShortVoice(), output_dir=root / 'output',
                                      audio_format='wav', workers=1, message=messages.append)
            self.assertEqual(len(list(result.glob('*.wav'))), 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(next(result.glob('*.epub')).read_bytes(), original)
            self.assertTrue(any('secours automatique' in message for message in messages))

    def test_cli_fallback_converts_in_both_interactive_and_yes_modes(self):
        from epub_audio.cli.main import main
        for unattended in (False, True):
            with self.subTest(unattended=unattended), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = make_epub(root, [('one.xhtml', '<p>Texte sans titres.</p>')])
                output = root / 'output'
                flags = [str(source), '--output', str(output), '--no-llm-chapters', '--no-normalize',
                         '--no-batching', '--segmentation', 'paragraph', '--voice', 'expr_chap_s01_0004-7e5a77eb', '--rate', '1.0', '--workers', '1',
                         '--codec', 'mp3', '--bitrate', '128', '--frequency', '44.1']
                if unattended:
                    flags.append('--yes')
                with patch('epub_audio.cli.main.QwenEngine', return_value=ShortVoice()), \
                     patch('epub_audio.cli.main.load_preferences', return_value={}), \
                     patch('epub_audio.cli.main.save_preferences'), \
                     patch('builtins.input', side_effect=[] if unattended else ['', '']) as prompt, \
                     patch('sys.stdout', new=io.StringIO()) as messages:
                    self.assertEqual(main(flags), 0)
                self.assertEqual(prompt.call_count, 0 if unattended else 2)
                self.assertEqual(len(list(output.rglob('*.mp3'))), 1)
                self.assertIn('secours automatique', messages.getvalue())

    def test_cli_wav_master_runs_without_ffmpeg_and_preserves_pcm_format(self):
        from epub_audio.cli.main import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make_epub(root, [('one.xhtml', '<h1>Premier</h1><p>Un récit.</p>')],
                               [('one.xhtml', 'Premier')])
            output = root / 'output'
            original = source.read_bytes()
            with patch('epub_audio.cli.main.QwenEngine', return_value=ShortVoice()), \
                 patch('epub_audio.cli.main.load_preferences', return_value={}), \
                 patch('epub_audio.cli.main.save_preferences'), \
                 patch('shutil.which', side_effect=AssertionError('External tool lookup')), \
                 patch('epub_audio.adapters.ffmpeg.run_tool', side_effect=AssertionError('External command')), \
                 patch('builtins.input', side_effect=AssertionError('Unexpected question')), \
                 patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(main([str(source), '--codec', 'wav', '--yes', '--output', str(output),
                                       '--voice', 'expr_chap_s01_0004-7e5a77eb', '--rate', '1.0', '--workers', '1']), 0)
            tracks = list(output.rglob('*.wav'))
            self.assertEqual(len(tracks), 2)
            self.assertEqual(tracks[0].read_bytes(), tracks[1].read_bytes())
            self.assertEqual(len(list((output / 'master').rglob('*.wav'))), 1)
            with wave.open(str(tracks[0]), 'rb') as audio:
                self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getsampwidth()), (1, 24000, 2))
                self.assertGreater(audio.getnframes(), 0)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(next(output.rglob('*.epub')).read_bytes(), original)

    def test_mixed_batch_converts_every_book_and_keeps_subfolders(self):
        from epub_audio.cli.main import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_root, output = root / 'input', root / 'output'
            for name in ('A', 'B', 'C'):
                directory = input_root / name
                directory.mkdir(parents=True)
                toc = [('one.xhtml', 'Premier')] if name == 'B' else []
                html = '<h1>Premier</h1><p>Un récit.</p>' if toc else '<p>Récit sans titres.</p>'
                make_epub(directory, [('one.xhtml', html)], toc)
            with patch('epub_audio.cli.main.QwenEngine', return_value=ShortVoice()), \
                 patch('epub_audio.cli.main.load_preferences', return_value={}), \
                 patch('epub_audio.cli.main.save_preferences'), \
                 patch('builtins.input', side_effect=AssertionError('Unexpected question')), \
                 patch('sys.stdout', new=io.StringIO()) as messages:
                status = main([str(input_root), '--yes', '--output', str(output),
                               '--voice', 'expr_chap_s01_0004-7e5a77eb', '--rate', '1.0', '--workers', '1'])
            self.assertEqual(status, 0)
            self.assertEqual(messages.getvalue().count('secours automatique'), 2)
            for name in ('A', 'B', 'C'):
                self.assertEqual(len(list((output / name).rglob('*.wav'))), 1)
                self.assertEqual(len(list((output / name).rglob('*.epub'))), 1)

    def test_source_name_copy_and_first_chapter_introduction_preserve_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make_epub(root, [('one.xhtml', '<h1>Premier</h1><p>Premier récit.</p>'),
                ('two.xhtml', '<h1>Second</h1><p>Second récit.</p>')],
                [('one.xhtml', 'Premier'), ('two.xhtml', 'Second')])
            source = source.rename(root / "027 - Piège dans l'espace - Guieu, Jimmy.epub")
            original = source.read_bytes()
            messages = []
            result = process_book(source, ShortVoice(), audio_format='wav', workers=2,
                                  output_dir=root / 'output', message=messages.append)
            expected = "027 - Piège dans l'espace - Guieu, Jimmy"
            self.assertEqual(result.name, expected + ' - 0m')
            self.assertEqual((result / (expected + '.epub')).read_bytes(), original)
            self.assertEqual(source.read_bytes(), original)
            tracks = sorted(result.glob('*.wav'))
            self.assertEqual(len(tracks), 2)
            # The introduction stays separate, followed by a two-second pause.
            # Each fake spoken segment lasts 0.1 second.
            for track, expected_frames in zip(tracks, (52800, 2400)):
                with wave.open(str(track), 'rb') as audio:
                    self.assertEqual(audio.getnframes(), expected_frames)

    def test_source_name_keeps_original_name_and_unsafe_values_stay_in_output(self):
        self.assertEqual(book_basename(Path('Original.epub'), Path('output')), 'Original')
        name = book_basename(Path('Étoiles: demain?.epub'), Path('output'))
        self.assertNotRegex(name, r'[<>:"/\\|?*]')
        self.assertIn('Étoiles', name)

    def test_source_names_keep_existing_numbering_and_author_order(self):
        for name in ('Jules Verne - Vingt mille lieues sous les mers',
                     "027 - Piège dans l'espace - Guieu, Jimmy", '1a- La guerre éternelle (1975)'):
            with self.subTest(name=name):
                self.assertEqual(book_basename(Path(name + '.epub'), Path('output')), name)

    def test_spoken_introduction_puts_given_name_before_family_name(self):
        metadata = BookMetadata("L'Homme des jeux", ['Banks,Iain M.'])
        self.assertEqual(metadata.introduction(), "L'Homme des jeux. Un livre de Iain M. Banks.")

    def test_epub3_collection_metadata_is_used(self):
        book = epub.EpubBook()
        book.add_metadata('OPF', 'meta', 'Cycle', {'property': 'belongs-to-collection', 'id': 'series'})
        book.add_metadata('OPF', 'meta', 'series', {'refines': '#series', 'property': 'collection-type'})
        book.add_metadata('OPF', 'meta', '3', {'refines': '#series', 'property': 'group-position'})
        from epub_audio.io.epub_structure import read_metadata
        metadata = read_metadata(book)
        self.assertEqual((metadata.series, metadata.series_index), ('Cycle', '3'))


class ParagraphTests(unittest.TestCase):
    def test_long_paragraphs_remain_whole_with_original_paragraph_pauses(self):
        long = 'Une phrase complète et assez longue. ' * 100
        segments = list(plan_segments([long, 'Fin.']))
        self.assertEqual([part.text for part in segments], [long.strip(), 'Fin.'])
        self.assertEqual([part.pause_before_ms for part in segments], [0, 800])

    def test_failed_paragraph_is_split_without_omissions_or_added_silence(self):
        text = ' '.join(f'Voici la phrase numéro {number}.' for number in range(20))
        engine = LimitedVoice()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'chapter.wav'
            synthesize_with_fallback(engine, text, output)
            self.assertEqual(engine.requests[0], text)
            self.assertEqual(' '.join(engine.accepted), text)
            with wave.open(str(output), 'rb') as audio:
                self.assertEqual(audio.readframes(audio.getnframes()),
                                 struct.pack('<h', 1000) * (2400 * len(engine.accepted)))
            self.assertEqual(list(Path(folder).iterdir()), [output])

    def test_permanent_failure_is_bounded_and_never_publishes_partial_output(self):
        engine = Mock()
        engine.synthesize.side_effect = RuntimeError('Toujours en échec')
        text = 'Une phrase impossible à lire. ' * 100
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'chapter.wav'
            with self.assertRaisesRegex(RuntimeError, 'Toujours en échec'):
                synthesize_with_fallback(engine, text, output)
            self.assertLessEqual(engine.synthesize.call_count, 12)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
