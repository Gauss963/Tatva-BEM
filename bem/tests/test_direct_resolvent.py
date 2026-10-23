import runpy
from pathlib import Path

import numpy as np

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/build_resolvent_history.py"))


def test_work_coordinates_preserve_physical_response_and_energy_pairing():
    z = np.array([[4 + 2j, 0.4 - 0.1j, 0.8j], [0.4 - 0.1j, 7 + 3j, 0.6], [0.8j, 0.6, 6 + 4j]])
    c = np.diag([11.0, 13, 9])
    w = np.array([0.25, 7.0, 200.0])
    transformed = TOOL["compliance_in_work_coordinates"](z, c, w)
    force = np.array([3 + 1j, 2 - 2j, 17 + 1j])
    physical_q = np.linalg.solve(z + c, force)
    work_q = transformed @ (force / np.sqrt(w))
    np.testing.assert_allclose(work_q / np.sqrt(w), physical_q, rtol=1e-14)
    np.testing.assert_allclose(np.vdot(work_q, force / np.sqrt(w)), np.vdot(physical_q, force))
    weights = np.stack([transformed, 2 * transformed, -transformed])
    TOOL["physical_weights_inplace"](weights, w)
    for index, factor in enumerate([1, 2, -1]):
        np.testing.assert_allclose(weights[index] @ force, factor * physical_q, rtol=1e-14)
