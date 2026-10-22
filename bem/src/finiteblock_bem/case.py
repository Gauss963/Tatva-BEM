"""Finite-block model input; no Tatva execution or external-volume writes."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .friction import ContactState, SlipWeakeningContact
from .material import ElasticMaterial
from .mesh import BoundaryMesh, graded_fractions, polygon_boundary


@dataclass(frozen=True)
class TwoBlockCase:
    moving: BoundaryMesh
    stationary: BoundaryMesh
    material: ElasticMaterial
    contact: SlipWeakeningContact
    moving_contact: np.ndarray
    stationary_contact: np.ndarray
    spring_stiffness: float
    normal_pressure: float
    restart_actuator: float
    actuator_rate: float
    restart: ContactState
    payload: dict


def load_case(
    path,
    *,
    max_panel_length=None,
    contact_panel_length=None,
    contact_end_length=None,
    corner_panel_length=None,
):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        c = tomllib.load(stream)
    g = c["geometry"]
    h = c["diagnostic"]["max_panel_length"] if max_panel_length is None else max_panel_length
    moving_x, moving_y = g["moving_origin"]
    moving_width, moving_height = g["moving_size"]
    stationary_x, stationary_y = g["stationary_origin"]
    stationary_width, stationary_height = g["stationary_size"]
    cut = moving_y + moving_height - g["chamfer_length"]
    depth = g["chamfer_depth"]
    contact_h = h if contact_panel_length is None else contact_panel_length
    if not np.isfinite(contact_h) or contact_h <= 0:
        raise ValueError("Positive finite contact panel length required")
    edge_fractions = None
    length = cut - moving_y
    if contact_end_length is not None:
        edge_fractions = {"contact": graded_fractions(length, contact_h, contact_end_length)}
    elif contact_panel_length is not None:
        edge_fractions = {"contact": np.linspace(0, 1, int(np.ceil(length / contact_h)) + 1)}
    if (
        min(moving_width, moving_height, stationary_width, stationary_height) <= 0
        or abs(stationary_x - moving_x - moving_width) > 1e-10
        or stationary_y != moving_y
    ):
        raise ValueError("This case adapter expects adjacent positive rectangular blocks")
    if (
        not 0 < g["chamfer_length"] < moving_height
        or not 0 < depth < moving_width
        or stationary_y + stationary_height < moving_y + moving_height
    ):
        raise ValueError("Invalid chamfer or stationary-block extent")
    moving = polygon_boundary(
        [
            [moving_x, moving_y],
            [moving_x + moving_width, moving_y],
            [moving_x + moving_width, cut],
            [moving_x + moving_width - depth, moving_y + moving_height],
            [moving_x, moving_y + moving_height],
        ],
        ["platen", "contact", "chamfer", "free", "normal"],
        h,
        edge_fractions=edge_fractions,
        corner_length=corner_panel_length,
    )
    stationary = polygon_boundary(
        [
            [stationary_x, stationary_y],
            [stationary_x + stationary_width, stationary_y],
            [stationary_x + stationary_width, stationary_y + stationary_height],
            [stationary_x, stationary_y + stationary_height],
            [stationary_x, cut],
        ],
        ["free", "support_x", "support_y", "free", "contact"],
        h,
        edge_fractions=edge_fractions,
        corner_length=corner_panel_length,
    )
    im = moving.indices("contact")
    im = im[np.argsort(moving.centers[im, 1])]
    js = stationary.indices("contact")
    js = js[np.argsort(stationary.centers[js, 1])]
    if len(im) != len(js) or not np.allclose(
        moving.centers[im], stationary.centers[js], atol=1e-9, rtol=0
    ):
        raise ValueError("Contact panels must pair geometrically")
    if not np.allclose(moving.lengths[im], stationary.lengths[js], rtol=0, atol=1e-9):
        raise ValueError("Paired contact panels must have equal lengths")
    material = ElasticMaterial(**c["material"])
    contact = SlipWeakeningContact(**c["contact"])
    seed = Path(c["source"]["restart_interface"])
    if not seed.is_absolute():
        seed = path.parent / seed
    reference = np.genfromtxt(seed, delimiter=",", names=True)
    yy = moving.centers[im, 1]
    restart = ContactState(
        np.interp(yy, reference["y"], reference["plastic_slip"]),
        np.interp(yy, reference["y"], reference["cumulative_slip"]),
    )
    load = c["loading"]
    if load["spring_stiffness"] <= 0 or load["normal_pressure"] < 0:
        raise ValueError("Require positive spring stiffness and nonnegative pressure")
    return TwoBlockCase(
        moving,
        stationary,
        material,
        contact,
        im,
        js,
        load["spring_stiffness"],
        load["normal_pressure"],
        load["restart_actuator"],
        load["actuator_rate"],
        restart,
        c,
    )
