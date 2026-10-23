"""Independently audit the complex-radius domain of the standard BDF2 symbol.

For z=r*exp(-i*theta), Re(delta(z))=0 is a quadratic in r. The branch
near r=1 has log(rcrit)=theta**4/4+O(theta**6), not a quadratic margin.
This calculation does not assert a complete interpolation complexity theorem.
"""

import json
from pathlib import Path

import numpy as np

from finiteblock_bem.cq import bdf2_symbol
from finiteblock_bem.io import write_json


def main():
    records = []
    for angle in [0.01, 0.03, 0.1, 0.2]:
        cosine = np.cos(angle)
        discriminant = 4 * cosine**2 - 3 * np.cos(2 * angle)
        # Rationalized small root of (cos(2theta)/2)*r²-2*cos(theta)*r+3/2.
        radius = 3 / (2 * cosine + np.sqrt(discriminant))
        real_symbol = float(bdf2_symbol(radius * np.exp(-1j * angle)).real)
        np.testing.assert_allclose(real_symbol, 0, atol=1e-14)
        example = float(bdf2_symbol(np.exp(0.3 * angle**2 - 1j * angle)).real)
        records.append(
            dict(
                angle=angle,
                critical_log_radius=float(np.log(radius)),
                critical_log_radius_over_angle_fourth=float(np.log(radius) / angle**4),
                real_symbol_at_exact_boundary=real_symbol,
                real_symbol_at_quadratic_radius_example=example,
            )
        )
    output = Path(__file__).resolve().parents[1] / "outputs/bdf2-domain"
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        records=records,
        symbol="1.5-2*z+0.5*z*z",
        near_origin_margin="log(rcrit)=theta^4/4+O(theta^6)",
        scope="Independent symbol/root calculation; no global interpolation theorem claim",
        literature_note="Compare arXiv:2410.15079v1 section 5.1; its stated quadratic root expansion is not used here",
    )
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
