"""The WebViewer's embedded controls use the same queue as the HTTP panel."""
import queue
import unittest

from viewer_controls import build_commands_gui


class FakeState:
    def __init__(self):
        self.commands = queue.Queue()
        self.snapshot = dict(
            phase="Prêt", progress=0, paused=False,
            vision_enabled=True, vision_detected=True,
        )

    def read(self):
        return self.snapshot.copy()


class FakeImGui:
    def __init__(self, click=None):
        self.click = click
        self.labels = []
        self.begin_called = False
        self.end_called = False

    def Begin(self, title):
        self.begin_called = title
        return True

    def TextUnformatted(self, text):
        self.labels.append(text)

    def Separator(self):
        pass

    def SameLine(self):
        pass

    def Button(self, label):
        self.labels.append(label)
        return label == self.click

    def End(self):
        self.end_called = True


class EmbeddedViewerControlsTests(unittest.TestCase):
    def test_run_restart_queues_start(self):
        state = FakeState()
        ui = FakeImGui(click="Run / Restart")
        build_commands_gui(state, ui)
        self.assertEqual(state.commands.get_nowait(), "start")
        self.assertTrue(ui.end_called)
        self.assertEqual(ui.begin_called, "EarlyBots | Robot commands")

    def test_reset_queues_reset(self):
        state = FakeState()
        build_commands_gui(state, FakeImGui(click="Reset"))
        self.assertEqual(state.commands.get_nowait(), "reset")

    def test_pause_queues_pause_and_resume_label_updates(self):
        state = FakeState()
        build_commands_gui(state, FakeImGui(click="Pause"))
        self.assertEqual(state.commands.get_nowait(), "pause")
        state.snapshot["paused"] = True
        ui = FakeImGui(click="Resume")
        build_commands_gui(state, ui)
        self.assertIn("Resume", ui.labels)
        self.assertEqual(state.commands.get_nowait(), "pause")

    def test_no_input_does_not_enqueue_commands(self):
        state = FakeState()
        ui = FakeImGui()
        build_commands_gui(state, ui)
        self.assertTrue(state.commands.empty())
        self.assertIn("RGB soap detection: yes", ui.labels)


if __name__ == "__main__":
    unittest.main()
