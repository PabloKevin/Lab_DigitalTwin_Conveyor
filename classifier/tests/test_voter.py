"""Decision rule of classifier/live.py (to be ported to the firmware).  python -m unittest classifier.tests.test_voter"""
import unittest

from classifier.live import Voter

C = ["solid", "stripes", "corners", "other"]
S = [0.9, 0.05, 0.03, 0.02]          # confident "solid"
T = [0.1, 0.8, 0.05, 0.05]           # confident "stripes"
W = [0.5, 0.3, 0.1, 0.1]             # weak "solid"
O = [0.05, 0.05, 0.05, 0.85]         # "other"


def run(v, seq, center=(100, 80)):
    out = None
    for pr in seq:
        out = v.update(True, pr, center)
    return out


class TestVoter(unittest.TestCase):
    def v(self):
        return Voter(C, n_frames=5, min_agree=4, tau=0.7)   # rule logic; the real config is in params.json

    def test_clear_majority(self):
        out = run(self.v(), [S, S, T, S, S])
        self.assertEqual(out["decision"], "solid")

    def test_no_decision_before_n_frames(self):
        self.assertIsNone(run(self.v(), [S, S, S, S])["decision"])

    def test_split_votes_is_dudoso(self):
        self.assertEqual(run(self.v(), [S, S, T, T, S])["decision"], "dudoso")      # 3/5 < 4

    def test_low_confidence_is_dudoso(self):
        self.assertEqual(run(self.v(), [W] * 5)["decision"], "dudoso")

    def test_other_is_dudoso(self):
        self.assertEqual(run(self.v(), [O] * 5)["decision"], "dudoso")

    def test_decision_held_then_reset_when_object_leaves(self):
        v = self.v()
        run(v, [S] * 5)
        self.assertEqual(run(v, [T] * 5)["decision"], "solid")                     # held while the object stays
        for _ in range(4):
            out = v.update(False)
        self.assertIsNone(out["decision"])                                          # gone -> reset
        self.assertEqual(run(v, [T] * 5)["decision"], "stripes")

    def test_new_object_far_away_resets(self):
        v = self.v()
        run(v, [S] * 5, center=(100, 80))
        self.assertIsNone(v.update(True, T, (400, 80))["decision"])                # jump > 60 px = new object


if __name__ == "__main__":
    unittest.main()
