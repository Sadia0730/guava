import unittest

import torch

from tools.evaluate_pear_recovery import (
    angle_degrees, filter_stream, forward_joints, motion_metrics, recovery_time,
)
from main.live_pear_guava import OneEuroFilter


class RecoveryTests(unittest.TestCase):
    def test_matches_actual_live_filter(self):
        values=torch.randn(40,15)
        actual,_=filter_stream(values,30)
        filt=OneEuroFilter(2.,.3,1.)
        expected=torch.stack([filt(x.clone(),i/30).clone() for i,x in enumerate(values)])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_history_error_and_oracle_reset(self):
        clean=torch.zeros(60,3)
        corrupted=clean.clone()
        corrupted[10:30]=1
        a,_=filter_stream(clean,30)
        b,_=filter_stream(corrupted,30)
        c,resets=filter_stream(corrupted,30,reset_at=30)
        self.assertGreater(float((b[30]-a[30]).abs().max()),.1)
        torch.testing.assert_close(c[30:],a[30:],rtol=0,atol=0)
        self.assertEqual(resets,[30])

    def test_does_not_mutate_inputs(self):
        values=torch.randn(40,15)
        original=values.clone()
        filter_stream(values,30)
        torch.testing.assert_close(values,original,rtol=0,atol=0)

    def test_causal_prefix(self):
        values=torch.randn(40,15)
        changed=values.clone()
        changed[20:]+=100
        a,_=filter_stream(values,30)
        b,_=filter_stream(changed,30)
        torch.testing.assert_close(a[:20],b[:20],rtol=0,atol=0)

    def test_recovery_requires_consecutive_frames(self):
        self.assertIsNone(recovery_time(torch.tensor([0.,10.,0.,10.]),30))
        self.assertAlmostEqual(recovery_time(torch.tensor([10.,0.,0.,0.]),30),1/30)

    def test_static_motion_has_no_amplitude_ratio(self):
        result=motion_metrics(torch.zeros(10,2,3),torch.zeros(10,2,3),30)
        self.assertIsNone(result["amplitude_gain"])
        self.assertIsNone(result["direction_cosine"])

    def test_identity_fk_and_angles(self):
        rest=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        rotations=torch.eye(3).expand(2,3,3,3).clone()
        result=forward_joints(rotations,rest,torch.tensor([-1,0,1]))
        torch.testing.assert_close(result,rest.expand(2,3,3)*1000)
        self.assertEqual(float(angle_degrees(rotations,rotations).max()),0.)


if __name__=="__main__":
    unittest.main()
