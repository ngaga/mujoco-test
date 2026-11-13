"""Custom EarlyBots commands rendered *inside* MuJoCo Studio WebViewer.

The standard Studio Run/Reset controls do not own our scripted pick-and-place
state machine. This viewer plugin routes user commands through the same
thread-safe queue as the optional HTTP control panel.
"""

from mujoco.studio import messages, plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui


def build_commands_gui(state, ui=imgui):
    """Build a lightweight dockable panel; never mutate MuJoCo data here."""
    snapshot = state.read()
    if ui.Begin("EarlyBots | Robot commands"):
        ui.TextUnformatted("Hotel bathroom - scripted soap restocking")
        ui.Separator()
        if ui.Button("Run / Restart"):
            state.commands.put("start")
        ui.SameLine()
        if ui.Button("Reset"):
            state.commands.put("reset")
        ui.SameLine()
        pause_label = "Resume" if snapshot["paused"] else "Pause"
        if ui.Button(pause_label):
            state.commands.put("pause")
        ui.Separator()
        ui.TextUnformatted(f'Phase: {snapshot["phase"]}')
        ui.TextUnformatted(f'Progress: {snapshot["progress"]}%')
        if snapshot["vision_enabled"]:
            detected = "yes" if snapshot["vision_detected"] else "no"
            ui.TextUnformatted(f'RGB soap detection: {detected}')
        ui.TextUnformatted("Use these buttons for the EarlyBots script.")
    ui.End()


class EarlyBotsViewerControls(plugin.ViewerPlugin):
    """Studio viewer-thread plugin: button callbacks only enqueue commands."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    @messages.handler
    def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        build_commands_gui(self.state)
