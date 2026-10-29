"""Estimated attempt limits keep completed audio and retry whole failed passages."""
from pathlib import Path
import hashlib,json,tempfile,unittest
from unittest.mock import Mock,patch
from epub_audio.core.generation_budget import generation_step_budget
from epub_audio.core.tts_qwen import QwenEngine

KNOWN_PASSAGES = [('À peine l’étroit panneau fut-il refermé sur moi, qu’une obscurité profonde m’enveloppa. Mes yeux, imprégnés de la lumière extérieure, ne purent rien percevoir. Je sentis mes pieds nus se cramponner aux échelons d’une échelle de fer. Ned Land et Conseil, vigoureusement saisis, me suivaient. Au bas de l’échelle, une porte s’ouvrit et se referma immédiatement sur nous avec un retentissement sonore.', 285), ('Quelle fut la durée de ce sommeil, je l’ignore ; mais il dut être long, car il nous reposa complètement de nos fatigues. Je me réveillai le premier. Mes compagnons n’avaient pas encore bougé, et demeuraient étendus dans leur coin comme des masses inertes.', 175)]

class GenerationBudgetTests(unittest.TestCase):
    def test_observed_verne_title_has_room_for_its_172_steps(self):
        text=('De la Terre à la Lune. Un livre de Jules Verne.\nDe la Terre à la Lune\nJules Verne\n'
              'Publication: mille huit cent soixante-cinq\n'
              'Catégorie(s): Fiction, Action & Aventure, Science Fiction')
        # The old 166-step cap interrupted this exact input; replay reached EOS at 172.
        self.assertEqual(generation_step_budget(text),300)
        self.assertGreater(generation_step_budget(text),172)
        self.assertEqual(generation_step_budget(text,maximum=180),180)

    def test_minimum_applies_after_fifty_percent_increase(self):
        self.assertEqual(generation_step_budget('Chapitre cinq Le Roman de la Lune'),300)
        self.assertEqual(generation_step_budget('Un exemple avec une virgule, et une fin.'),300)

    def test_known_passages_fit_a_tight_budget_far_below_global_limit(self):
        for text,steps in KNOWN_PASSAGES:
            with self.subTest(characters=len(text)):
                budget=generation_step_budget(text)
                self.assertGreaterEqual(budget,steps)
                self.assertLess(budget,steps*2)
                self.assertLess(budget,640)

    def test_short_prompts_numbers_and_global_limit(self):
        self.assertEqual(generation_step_budget('Oui.'),300)
        self.assertGreater(generation_step_budget('9'*50),generation_step_budget('a'*50))
        self.assertEqual(generation_step_budget('Très long paragraphe. '*1000),2048)
        self.assertEqual(generation_step_budget('Oui.',maximum=32),32)
        for maximum in (0,-1,True,None,1.5):
            with self.subTest(maximum=maximum),self.assertRaises(ValueError):
                generation_step_budget('Texte',maximum)

    def test_exhausted_attempt_retries_full_text_keeps_good_audio_and_logs_reason(self):
        import numpy as np
        messages=[]
        engine=QwenEngine(batching=True,message=messages.append)
        engine.model=engine.fast_model=object();engine.prompt=[object()]
        samples=np.ones(2400,dtype=np.float32)*0.1
        full_text=KNOWN_PASSAGES[1][0];limit=generation_step_budget(full_text)
        engine.batch_backend=Mock()
        engine.batch_backend.generate.return_value=[
            (samples,24000,dict(ended_with_eos=True,steps=1,generation_limit=64,estimated_limit_reached=False)),
            (None,24000,dict(ended_with_eos=False,steps=limit,generation_limit=limit,estimated_limit_reached=True,batch_size=2))]
        engine._generate=Mock(return_value=([samples],24000,dict(ended_with_eos=True,steps=1)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'.cache').mkdir();(root/'.segments-9').mkdir()
            paths=[root/'good.wav',root/'.segments-9/000041.wav']
            with patch('epub_audio.core.tts_qwen.ROOT',root):
                self.assertEqual(engine.synthesize_batch(['Oui.',full_text],paths),paths)
            engine._generate.assert_called_once_with(full_text)
            self.assertEqual(engine.settings['max_new_tokens'],2048)
            self.assertTrue(all(path.is_file() for path in paths))
            entries=[json.loads(line) for line in (root/'.cache/qwen-segments.jsonl').read_text().splitlines()]
            self.assertEqual([row['batch_size'] for row in entries],[2,1])
            failed=[json.loads(line) for line in (root/'.cache/qwen-batch-failures.jsonl').read_text().splitlines()]
            self.assertEqual(len(failed),1)
            self.assertFalse(failed[0]['finished'])
            self.assertTrue(failed[0]['estimated_limit_reached'])
            self.assertEqual(failed[0]['generation_limit'],limit)
            self.assertEqual(failed[0]['text_sha256'],hashlib.sha256(full_text.encode()).hexdigest())
            self.assertTrue(any('Limite estimée atteinte' in message and str(limit) in message for message in messages))
            retry_messages=[message for message in messages if 'Chapitre 9, segment 42' in message]
            self.assertIn('Réessai individuel',retry_messages[0])
            self.assertEqual(len(retry_messages),1)
            self.assertIn('Limite estimée atteinte',retry_messages[0])

    def test_failure_journal_problem_does_not_prevent_retry(self):
        messages=[]
        engine=QwenEngine(batching=True,message=messages.append)
        with tempfile.TemporaryDirectory() as directory,patch('epub_audio.core.tts_qwen.ROOT',Path(directory)),patch.object(Path,'open',side_effect=OSError('disk failed')):
            engine._record_batch_failure('Texte',dict(ended_with_eos=False),'batch','start','limit')
        self.assertTrue(any('Journal des réessais indisponible' in message for message in messages))

if __name__=='__main__':unittest.main()
