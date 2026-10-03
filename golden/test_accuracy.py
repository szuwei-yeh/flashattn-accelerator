import unittest
import numpy as np
from report_accuracy import floating_attention, error_metrics


class AccuracyContract(unittest.TestCase):
    def test_uniform_scores_and_causal_prefix_mean(self):
        q=k=np.zeros((4,16),dtype=np.int8)
        v=np.broadcast_to(np.array([1,3,-2,6],dtype=np.int8)[:,None],(4,16))
        actual=floating_attention(q,k,v,(256,256,256))
        np.testing.assert_allclose(actual,np.full((4,16),2.))
        causal=floating_attention(q,k,v,(256,256,256),True)
        np.testing.assert_allclose(causal[:,0],[1,2,2/3,2])

    def test_signed_encoded_scale(self):
        q=k=np.zeros((4,16),dtype=np.int8)
        v=np.ones((4,16),dtype=np.int8)
        np.testing.assert_allclose(floating_attention(q,k,v,(256,256,0xff00)),-1)

    def test_documented_p_clipping_bias(self):
        reference=np.ones((64,16))
        actual=reference*(255/256)
        metrics=error_metrics(actual,reference)
        self.assertAlmostEqual(metrics['relative_l2'],1/256)
        self.assertAlmostEqual(metrics['max_abs'],1/256)

    def test_zero_reference_has_no_relative_error_metric(self):
        metrics=error_metrics(np.ones((2,2)),np.zeros((2,2)))
        self.assertIsNone(metrics['relative_l2'])
        self.assertEqual(metrics['rmse'],1)


if __name__ == '__main__':
    unittest.main()
