import unittest

from motion import HOME, Motion, PHASES


class MotionTests(unittest.TestCase):
    def test_start_and_complete(self):
        motion = Motion()
        self.assertEqual(motion.progress, 0)
        motion.start()
        self.assertTrue(motion.running)
        holding_seen = False
        for _ in range(1300):
            motion.advance(0.01)
            holding_seen |= motion.holding
        self.assertTrue(holding_seen)
        self.assertTrue(motion.finished)
        self.assertEqual(motion.position, HOME)
        self.assertEqual(motion.progress, 100)
        self.assertFalse(motion.holding)

    def test_does_not_move_when_idle(self):
        motion = Motion()
        motion.advance(1.0)
        self.assertEqual(motion.position, HOME)

    def test_reset(self):
        motion = Motion()
        motion.start()
        motion.advance(sum(p.duration for p in PHASES) / 2)
        motion.reset()
        self.assertFalse(motion.running)
        self.assertFalse(motion.finished)
        self.assertEqual(motion.position, HOME)

    def test_negative_dt(self):
        with self.assertRaises(ValueError):
            Motion().advance(-0.1)


if __name__ == "__main__":
    unittest.main()
