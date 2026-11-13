"""CPU-only checks of aggregation/protocol integrity; no model imports or GPU work."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import phase_timer as PT
import collect_timing as C
import run_suite as S


class TimingTests(unittest.TestCase):
    def test_invalid_samples_are_not_results(self):
        self.assertIsNone(PT.stats([]))
        for samples in [[float('nan')],[float('inf')],[-1]]:
            with self.assertRaises(ValueError):PT.stats(samples)

    def test_combination_uses_paired_samples(self):
        p=PT.Phases();p.t={'a':[0,0,100],'b':[100,0,0]}
        self.assertEqual(p.combined('a','b')['median_ms'],100)
        p.t['b'].append(1)
        with self.assertRaises(ValueError):p.combined('a','b')

    def test_serial_excludes_warmup_and_preserves_call_count(self):
        calls=[]
        with patch.object(PT,'sync',lambda:None):
            result=PT.measure_serial(lambda p,i:calls.append(i),2,4)
        self.assertEqual(calls,list(range(6)));self.assertEqual(result['n'],4)

    def test_missing_stage_stays_missing(self):
        r={'method':'RAGA','clips':{'neuman/bike':{'animation':{'total':PT.stats([10]),'segmentation':None}}}}
        rows,phases=C.tables([(Path('fixture.json'),r)])
        self.assertIsNone(rows[0]['serial_rate_fps'])
        self.assertEqual(rows[0]['instrumented_rate_fps'],100)
        self.assertIsNone(next(p for p in phases if p['phase']=='segmentation')['median_ms'])

    def test_collector_rejects_mixed_gpu_and_shortened_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for i,gpu in enumerate(['GPU A','GPU B']):
                (root/f'{i}.json').write_text(json.dumps({'method':'test','env':{'gpu':gpu},'clips':{},'protocol':{}}))
            with self.assertRaisesRegex(ValueError,'mixed GPU'):C.read_results([root])
            f=root/'short.json';f.write_text(json.dumps({'method':'test','clips':{},'protocol':{'shortened':True}}))
            with self.assertRaisesRegex(ValueError,'test-only'):C.read_results([f])

    def test_default_jobs_are_separate_rows_for_cooling(self):
        jobs=S.commands(Path('/inputs'),Path('/results'),['raga'],'neuman/bike','python')
        self.assertEqual(len(jobs),4)
        for name,cmd in jobs:
            self.assertEqual(cmd[cmd.index('--clips')+1],'neuman/bike')
            self.assertNotIn(',',cmd[cmd.index('--rows')+1])
        self.assertIn('fb_sprint_update',jobs[-1][1])

if __name__=='__main__':unittest.main()
