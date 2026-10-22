import numpy as np
import pytest

from finiteblock_bem.comparison import panel_average


def test_nodal_to_panel_projection_preserves_linear_fields_and_resultant():
    y = np.array([0.0, 0.1, 0.7, 1.0])
    centers = np.array([0.125, 0.375, 0.625, 0.875])
    lengths = np.full(4, 0.25)
    projected = panel_average(y, 2 + 3 * y, centers, lengths)
    np.testing.assert_allclose(projected, 2 + 3 * centers)
    np.testing.assert_allclose(projected @ lengths, 3.5)
    with pytest.raises(ValueError, match="outside"):
        panel_average(y, y, [0.0], [0.5])
