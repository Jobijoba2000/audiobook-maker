"""Batching for Faster 0.6B and preserves text, PCM order and pauses."""
from pathlib import Path
from types import SimpleNamespace
import io
from collections import Counter
import json
import struct
import tempfile
import unittest
import wave
from unittest.mock import Mock, patch

from epub_audio.cli.args import arguments
from epub_audio.cli.main import main
from epub_audio.core.models import BookPlan, BookMetadata, Chapter
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book
from epub_audio.core.tts_generate import render_chapter
from epub_audio.core.tts_qwen import QwenEngine
from epub_audio.core.tts_batch_queue import BatchedSynthesisPool
from epub_audio.io.preferences import load_preferences, save_preferences

VOICE = 'expr_chap_s01_0004-7e5a77eb'


def pcm(n):
    return struct.pack('<h', n * 1000) * 2400


class BatchVoice:
    serial_gpu = True
    batch_size = 2
    max_segment_characters = None

    def __init__(self):
        self.groups = []
        self.calls = 0
        self.sample_ids = {}

    def synthesize_batch(self, texts, paths):
        self.groups.append(list(texts))
        for text, path in zip(texts, paths):
            self.calls += 1
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(pcm(self.sample_ids.get(text, self.calls)))
        return list(paths)


class BatchingTests(unittest.TestCase):
    def test_menu_selects_all_sizes_and_remembers_them(self):
        for choice, expected in (('1', False), ('2', True), ('3', True), ('4', True),
                                 ('5', True), ('6', True), ('7', True), ('8', True)):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'book.epub'
                source.write_bytes(b'fixture')
                flags = [str(source), '--voice', VOICE, '--codec', 'wav', '--segmentation', 'paragraph', '--no-llm-chapters', '--no-normalize']
                with patch('epub_audio.io.preferences.PREFERENCES_FILE', root/'prefs.json'), \
                     patch('epub_audio.cli.main.QwenEngine') as engine, \
                     patch('epub_audio.cli.main.process_book') as process, \
                     patch('builtins.input', side_effect=['', 'invalid', choice, '']), \
                     patch('sys.stdout', new=io.StringIO()) as output:
                    self.assertEqual(main(flags), 0)
                    self.assertIs(engine.call_args.kwargs['batching'], expected)
                    self.assertIs(load_preferences()['faster_small_batching'], expected)
                    self.assertEqual(engine.call_args.kwargs['batch_size'], {'1':6,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8}[choice])
                    self.assertEqual(load_preferences()['faster_small_batch_size'], {'1':6,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8}[choice])
                    self.assertEqual(process.call_args.kwargs['options'].workers, 20)
                    self.assertIn('1. Sans batching', output.getvalue())
                    self.assertIn('2. Batching par 2 passages', output.getvalue())
                    self.assertIn('3. Batching par 3 passages', output.getvalue())
                    self.assertIn('4. Batching par 4 passages', output.getvalue())
                    self.assertIn('5. Batching par 5 passages', output.getvalue())
                    self.assertIn('6. Batching par 6 passages', output.getvalue())
                    self.assertIn('7. Batching par 7 passages', output.getvalue())
                    self.assertIn('8. Batching par 8 passages', output.getvalue())
                    with patch('builtins.input', side_effect=['', '', '']) as prompts:
                        self.assertEqual(main(flags), 0)
                    self.assertIn('['+choice+']', prompts.call_args_list[1].args[0])
                    self.assertIs(engine.call_args.kwargs['batching'], expected)

    def test_removed_model_and_engine_flags_are_rejected_before_loading(self):
        for flags in (['--engine', 'qwen'], ['--engine', 'faster-qwen'], ['--model-size', '1.7B']):
            with self.subTest(flags=flags), patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                main([*flags, '--yes'])
            engine.assert_not_called()

    def test_noninteractive_overrides_preferences_without_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'book.epub'
            source.write_bytes(b'fixture')
            with patch('epub_audio.io.preferences.PREFERENCES_FILE',root/'prefs.json'), \
                 patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('epub_audio.cli.main.process_book'), \
                 patch('builtins.input',side_effect=AssertionError('Unexpected question')), \
                 patch('sys.stdout',new=io.StringIO()):
                save_preferences(VOICE,1,audio_format='wav',faster_small_batching=True)
                for extra,expected in (([],True),(['--no-batching'],False),(['--batching'],True)):
                    self.assertEqual(main([str(source),'--yes']+extra),0)
                    self.assertIs(engine.call_args.kwargs['batching'],expected)

    def test_invalid_preferences_and_unsupported_engine_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prefs=Path(folder)/'prefs.json'
            with patch('epub_audio.io.preferences.PREFERENCES_FILE',prefs):
                for bad in (None,0,1,2,'true',[],{}):
                    prefs.write_text(json.dumps(dict(faster_small_batching=bad)),encoding='utf-8')
                    self.assertNotIn('faster_small_batching',load_preferences())
                    with self.assertRaises(ValueError):
                        save_preferences(VOICE,1,faster_small_batching=bad)
        self.assertIsNone(arguments([]).batching)
        self.assertTrue(arguments(['--batching']).batching)
        self.assertFalse(arguments(['--no-batching']).batching)
        self.assertTrue(QwenEngine(batching=True).batching)
        self.assertEqual(QwenEngine().model_size, '0.6B')
        for bad in (1,0,None,'yes'):
            with self.assertRaises(ValueError):
                QwenEngine(batching=bad)

    def test_pairs_cross_chapters_without_changing_sentence_scene_or_intro_pauses(self):
        voice=BatchVoice()
        # Long initial segments keep this test focused on existing PCM pauses.
        title='Titre'+'x'*200
        scene='Scène nouvelle '+'x'*200+'.'
        voice.sample_ids={text:index for index,text in enumerate([title+'.','Première phrase.','Deuxième phrase.',scene,'Dernière phrase.'],1)}
        plan=BookPlan([Chapter('One',['Première phrase. Deuxième phrase.']),
                       Chapter('Two',[scene,'***','Dernière phrase.'])],
                      metadata=BookMetadata(title))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'book.epub'
            source.write_bytes(b'fixture')
            with patch('epub_audio.core.pipeline.read_book',return_value=plan):
                output=process_book(source,voice,output_dir=root/'out',message=lambda s:None,
                    options=AudioOptions(audio_format='wav',segmentation='sentence'))
            self.assertEqual(Counter(text for group in voice.groups for text in group),Counter(voice.sample_ids.keys()))
            tracks=sorted(output.glob('*.wav'))
            self.assertEqual(len(tracks),2)
            expected=[pcm(1)+b'\0\0'*48000+pcm(2)+pcm(3),pcm(4)+b'\0\0'*36000+pcm(5)]
            for track,frames in zip(tracks,expected):
                with wave.open(str(track),'rb') as audio:
                    self.assertEqual(audio.readframes(audio.getnframes()),frames)
            self.assertFalse(list(output.glob('.segments-*')))

    def test_single_chapter_keeps_every_paragraph_and_handles_odd_group(self):
        voice=BatchVoice()
        # Passages over 100 characters stay separate in paragraph mode.
        texts=[word+'x'*200 for word in ('One','Two','Three','Four','Five')]
        voice.sample_ids={text:index for index,text in enumerate(texts,1)}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            render_chapter(voice,texts,root/'segments',root/'chapter.wav',
                           workers=1,progress=lambda s:None)
            self.assertEqual(Counter(text for group in voice.groups for text in group),Counter(voice.sample_ids.keys()))
            self.assertEqual(sorted(map(len,voice.groups)),[1,2,2])
            with wave.open(str(root/'chapter.wav'),'rb') as audio:
                self.assertEqual(audio.readframes(audio.getnframes()),
                    (b'\0\0'*19200).join(pcm(n) for n in range(1,6)))

    def test_group_errors_and_initialization_errors_reach_every_future(self):
        for initialization_fails in (False,True):
            voice=BatchVoice()
            if initialization_fails:
                voice.initialize_worker=Mock(side_effect=RuntimeError('initialization failed'))
            else:
                voice.synthesize_batch=Mock(side_effect=OSError('disk failed'))
            pool=BatchedSynthesisPool(voice)
            with patch('concurrent.futures._base.LOGGER.critical'):
                try:
                    futures=pool.submit_batch([('A',Path('a.wav')),('B',Path('b.wav'))])
                    for future in futures:
                        with self.assertRaises(Exception):
                            future.result(timeout=5)
                finally:
                    pool.shutdown(wait=True,cancel_futures=True)

    def _fake_engine(self):
        engine=QwenEngine(batching=True,message=lambda s:None)
        engine.model=object()
        engine.fast_model=object()
        engine.prompt=[object()]
        engine.batch_backend=Mock()
        return engine

    def test_failed_batch_retries_both_passages_with_the_existing_single_path(self):
        import numpy as np
        engine=self._fake_engine()
        engine.batch_backend.generate.side_effect=RuntimeError('GPU out of memory')
        engine._generate=Mock(return_value=([np.ones(2400,dtype=np.float32)*0.1],24000,
                                            dict(ended_with_eos=True,steps=1)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'.cache').mkdir()
            paths=[root/'one.wav',root/'two.wav']
            with patch('epub_audio.core.tts_qwen.ROOT',root):
                self.assertEqual(engine.synthesize_batch(['One','Two'],paths),paths)
            self.assertEqual([c.args[0] for c in engine._generate.call_args_list],['One','Two'])
            self.assertTrue(all(p.is_file() for p in paths))
            entries=[json.loads(line) for line in (root/'.cache/qwen-segments.jsonl').read_text().splitlines()]
            self.assertEqual([r['batch_size'] for r in entries],[1,1])

    def test_incomplete_sample_alone_is_retried_and_good_audio_is_retained(self):
        import numpy as np
        engine=self._fake_engine()
        samples=np.ones(2400,dtype=np.float32)*0.1
        engine.batch_backend.generate.return_value=[(samples,24000,dict(ended_with_eos=True,steps=1)),
                                                    (None,24000,dict(ended_with_eos=False,steps=2048))]
        engine._generate=Mock(return_value=([samples],24000,dict(ended_with_eos=True,steps=1)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'.cache').mkdir()
            paths=[root/'one.wav',root/'two.wav']
            with patch('epub_audio.core.tts_qwen.ROOT',root):
                self.assertEqual(engine.synthesize_batch(['One','Two'],paths),paths)
            engine._generate.assert_called_once_with('Two')
            entries=[json.loads(line) for line in (root/'.cache/qwen-segments.jsonl').read_text().splitlines()]
            self.assertEqual([r['batch_size'] for r in entries],[2,1])
            self.assertIsNotNone(entries[0]['batch_id'])

    def test_individual_events_precede_files_but_failed_attempt_waits_for_retry(self):
        import numpy as np
        for invalid_wave in (False, True):
            with self.subTest(invalid_wave=invalid_wave), tempfile.TemporaryDirectory() as folder:
                engine = self._fake_engine()
                root = Path(folder)
                (root/'.cache').mkdir()
                paths = [root/'one.wav', root/'two.wav']
                samples = np.ones(2400, dtype=np.float32)*0.1
                def generate(texts, prompt, *, on_generated):
                    on_generated(0)
                    if invalid_wave:
                        on_generated(1)
                    return [(samples, 24000, dict(ended_with_eos=True, steps=1)),
                            (np.full(2400,np.nan) if invalid_wave else None,24000,
                             dict(ended_with_eos=invalid_wave, steps=64,
                                  estimated_limit_reached=not invalid_wave, generation_limit=64))]
                engine.batch_backend.generate.side_effect = generate
                engine._generate = Mock(return_value=([samples],24000,dict(ended_with_eos=True,steps=1)))
                events = []
                def notify(index, state):
                    events.append((index, state, paths[index].is_file()))
                with patch('epub_audio.core.tts_qwen.ROOT', root):
                    self.assertEqual(engine.synthesize_batch(['One','Two'],paths,on_progress=notify),paths)
                expected = [(0,'generated',False)]
                if invalid_wave:
                    expected.append((1,'generated',False))
                expected += [(0,'ready',True),(1,'retry',False),(1,'ready',True)]
                self.assertEqual(events, expected)
                engine._generate.assert_called_once_with('Two')

    def test_display_failure_is_not_mistaken_for_a_gpu_failure(self):
        engine = self._fake_engine()
        def generate(texts, prompt, *, on_generated):
            on_generated(0)
            raise AssertionError('The display failure should have propagated')
        engine.batch_backend.generate.side_effect = generate
        def broken_display(index, state):
            raise RuntimeError('display unavailable')
        with patch('epub_audio.core.tts_retry.synthesize_with_fallback') as retry, \
             patch('torch.cuda.empty_cache') as empty_cache:
            with self.assertRaisesRegex(RuntimeError, 'display unavailable'):
                engine.synthesize_batch(['One','Two'],[Path('one.wav'),Path('two.wav')],
                                        on_progress=broken_display)
            retry.assert_not_called()
            empty_cache.assert_not_called()

    def test_last_single_passage_uses_normal_generation_and_file_errors_propagate(self):
        engine=self._fake_engine()
        with patch('epub_audio.core.tts_retry.synthesize_with_fallback',return_value=Path('one.wav')) as retry:
            self.assertEqual(engine.synthesize_batch(['One'],[Path('one.wav')]),[Path('one.wav')])
            retry.assert_called_once_with(engine,'One',Path('one.wav'))
            engine.batch_backend.generate.assert_not_called()
        engine.batch_backend.generate.return_value=[([0.1],24000,dict(ended_with_eos=True))]*2
        with patch.object(engine,'_save_wave',side_effect=OSError('disk full')), \
             patch('epub_audio.core.tts_retry.synthesize_with_fallback') as retry:
            with self.assertRaises(OSError):
                engine.synthesize_batch(['One','Two'],[Path('one.wav'),Path('two.wav')])
            retry.assert_not_called()

    def test_backend_reuses_icl_prompt_and_decodes_with_trimmed_reference_context(self):
        import torch
        import numpy as np
        from epub_audio.core.tts_batch import FasterBatchBackend
        backend=FasterBatchBackend.__new__(FasterBatchBackend)
        backend.batch_size=2
        reference=torch.zeros(2,16,dtype=torch.long)
        codes=torch.ones(3,16,dtype=torch.long)
        tokenizer=SimpleNamespace(decode=Mock(return_value=([np.arange(500,dtype=np.float32)],24000)))
        model=SimpleNamespace(speech_tokenizer=tokenizer)
        backend.base=SimpleNamespace(sample_rate=24000,_prepare_generation=Mock(
            return_value=(model,object(),object(),object(),object(),object(),object(),reference)))
        backend.settings=dict(max_new_tokens=2048,temperature=0.9,sampling=True,repetition_penalty=1.05)
        backend.predictor=object()
        backend.talker=object()
        backend.codec_embeddings=object()
        prompt=[object()]
        with patch('epub_audio._vendor.faster_qwen3_tts_batch.batched_generate.batched_fast_generate',
                   return_value=[(codes,dict(ended_with_eos=True)),(None,dict(ended_with_eos=False))]) as generate:
            outputs=backend.generate(['One','Two'],prompt)
        from epub_audio.core.generation_budget import generation_step_budget
        self.assertIs(generate.call_args.kwargs['codec_embedding_graph'],backend.codec_embeddings)
        self.assertEqual(generate.call_args.kwargs['completion_check_interval'],4)
        self.assertEqual(generate.call_args.kwargs['max_steps_per_sample'],
                         [generation_step_budget('One'),generation_step_budget('Two')])
        self.assertEqual(len(outputs[0][0]),300)
        self.assertEqual(outputs[0][0][0],200)
        self.assertIsNone(outputs[1][0])
        tokenizer.decode.assert_called_once()
        self.assertEqual(tokenizer.decode.call_args.args[0]['audio_codes'].shape,(1,5,16))
        for call in backend.base._prepare_generation.call_args_list:
            self.assertIs(call.kwargs['voice_clone_prompt'],prompt)
            self.assertFalse(call.kwargs['xvec_only'])
            self.assertFalse(call.kwargs['append_silence'])
            self.assertTrue(call.kwargs['non_streaming_mode'])


    def test_cli_size_activates_batching_and_missing_size_defaults_to_six(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'book.epub'
            source.write_bytes(b'fixture')
            prefs=root/'prefs.json'
            with patch('epub_audio.io.preferences.PREFERENCES_FILE',prefs), \
                 patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('epub_audio.cli.main.process_book'), \
                 patch('builtins.input',side_effect=AssertionError('Unexpected question')), \
                 patch('sys.stdout',new=io.StringIO()):
                prefs.write_text(json.dumps(dict(voice=VOICE,audio_format='wav',faster_small_batching=True)),encoding='utf-8')
                self.assertEqual(main([str(source),'--yes']),0)
                self.assertEqual(engine.call_args.kwargs['batch_size'],6)
                for size in (2,3,4,5,6,7,8):
                    self.assertEqual(main([str(source),'--yes','--batch-size',str(size)]),0)
                    self.assertTrue(engine.call_args.kwargs['batching'])
                    self.assertEqual(engine.call_args.kwargs['batch_size'],size)
                    self.assertEqual(load_preferences()['faster_small_batch_size'],size)
                    self.assertEqual(main([str(source),'--yes']),0)
                    self.assertEqual(engine.call_args.kwargs['batch_size'],size)
                self.assertEqual(main([str(source),'--yes','--no-batching']),0)
                self.assertFalse(engine.call_args.kwargs['batching'])
                self.assertEqual(load_preferences()['faster_small_batch_size'],8)
                self.assertEqual(main([str(source),'--yes','--batching']),0)
                self.assertTrue(engine.call_args.kwargs['batching'])
                self.assertEqual(engine.call_args.kwargs['batch_size'],8)

    def test_invalid_sizes_and_conflicting_arguments_fail_before_gpu_loading(self):
        from epub_audio.core.tts_batch import FasterBatchBackend
        with tempfile.TemporaryDirectory() as folder:
            prefs=Path(folder)/'prefs.json'
            with patch('epub_audio.io.preferences.PREFERENCES_FILE',prefs):
                for value in (None,True,False,1,0,-1,9,'3',[],{}):
                    prefs.write_text(json.dumps(dict(faster_small_batch_size=value)),encoding='utf-8')
                    self.assertNotIn('faster_small_batch_size',load_preferences())
                    with self.assertRaises(ValueError):
                        save_preferences(VOICE,1,faster_small_batch_size=value)
                    with self.assertRaises(ValueError):
                        QwenEngine(batching=True,batch_size=value)
                    with self.assertRaises(ValueError):
                        FasterBatchBackend(None,{},batch_size=value)
        for flags in (['--batch-size','1'],['--batch-size','0'],['--batch-size','-1'],['--batch-size','9'],['--batch-size','3','--no-batching']):
            with patch('sys.stderr',new=io.StringIO()), self.assertRaises(SystemExit):
                arguments(flags)

    def test_larger_sizes_keep_order_pauses_and_partial_groups(self):
        texts=[word+'x'*200 for word in ('One','Two','Three','Four','Five','Six','Seven','Eight','Nine')]
        for size in (3,4,5,6,7,8):
            voice=BatchVoice()
            voice.batch_size=size
            voice.sample_ids={text:index for index,text in enumerate(texts,1)}
            with self.subTest(size=size), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                render_chapter(voice,texts,root/'segments',root/'chapter.wav',workers=1,progress=lambda s:None)
                self.assertEqual(Counter(text for group in voice.groups for text in group),Counter(texts))
                self.assertEqual(sorted(map(len,voice.groups)),sorted(len(texts[i:i+size]) for i in range(0,len(texts),size)))
                with wave.open(str(root/'chapter.wav'),'rb') as audio:
                    self.assertEqual(audio.readframes(audio.getnframes()),
                        (b'\0\0'*19200).join(pcm(n) for n in range(1,10)))

    def test_engine_accepts_larger_and_partial_groups_but_rejects_oversized_groups(self):
        import numpy as np
        for size in (3,4,5,6,7,8):
            engine=QwenEngine(batching=True,batch_size=size,
                              message=lambda s:None)
            engine.model=engine.fast_model=object()
            engine.prompt=[object()]
            engine.batch_backend=Mock()
            with self.subTest(size=size), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                (root/'.cache').mkdir()
                for count in (size,size-1):
                    paths=[root/f'{count}-{index}.wav' for index in range(count)]
                    engine.batch_backend.generate.return_value=[
                        (np.ones(2400,dtype=np.float32)*0.1,24000,dict(ended_with_eos=True,steps=1))]*count
                    with patch('epub_audio.core.tts_qwen.ROOT',root):
                        self.assertEqual(engine.synthesize_batch(['Passage']*count,paths),paths)
                    self.assertTrue(all(path.is_file() for path in paths))
                with self.assertRaises(ValueError):
                    engine.synthesize_batch(['Too many']*(size+1),[root/'unused.wav']*(size+1))



if __name__=='__main__':
    unittest.main()
