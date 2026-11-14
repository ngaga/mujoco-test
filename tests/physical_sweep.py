"""Diagnostic sweep for choosing a physically stable grip command."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import physical_grasp

model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "bathroom_physical.xml"))
for target in [0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.052, 0.056, 0.060, 0.065, 0.070, 0.075]:
    physical_grasp.JAW_CLOSED = target
    result = physical_grasp.run_contact_smoke(model, strict=False)
    print(f"target={target:.3f} peak={result['highest_z']:.3f} "
          f"transport={result['transport_peak_z']:.3f} "
          f"tray={result['in_tray']} final={result['soap_xyz']}", flush=True)
