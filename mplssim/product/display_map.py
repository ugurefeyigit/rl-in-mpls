"""Display-only fixed engineering layout for the topology stage.

This registry is *presentation metadata*. `configs/topology.yaml` keeps the
scientific graph — router IDs, link IDs, endpoints, capacities, delays and
weights — and nothing here may change it. The x/y in the scientific config is a
schematic left-to-right diagram. The product reuses that proven placement,
scaled into SVG coordinates, so paths and node plates remain easy to separate.
City names provide the presentation vocabulary; their placement is explicitly
not geographic.

Positions are fixed. There is no force-directed layout and nothing moves during
a session: a node that drifts between two steps destroys the one visual anchor
the whole product depends on.

Coordinates preserve the original topology diagram with a display-only scale,
y increasing downward (SVG order). No coordinate is written back to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mplssim.display import CITY_NAMES

#: Operational role labels. Derived from the topology's own role field; the role
#: string in `configs/topology.yaml` remains the contract.
ROLE_LABELS: dict[str, str] = {
    "PE_IN": "LER · ingress",
    "PE_OUT": "LER · egress",
    "P": "LSR",
    "AGG": "LSR · aggregation",
}

#: Short role token used on the node plate edge. Non-colour role encoding.
ROLE_TOKENS: dict[str, str] = {
    "PE_IN": "LER", "PE_OUT": "LER", "P": "LSR", "AGG": "AGG",
}


@dataclass(frozen=True)
class NodeLayout:
    x: float
    y: float
    #: Where the plate sits relative to the node anchor, so labels do not collide.
    anchor: str


#: Label placement is presentation metadata; router coordinates come directly
#: from the existing pre-redesign schematic and remain read-only.
LABEL_ANCHORS: dict[str, str] = {
    router_id: "center" for router_id in (
        "PE1", "PE2", "PE3", "PE4", "P1", "P2", "P3", "P4", "P5",
        "P6", "P7", "P8", "A1", "A2", "PE5", "PE6", "PE7", "PE8"
    )
}

# The legacy Cytoscape view rendered this long P4-P7 edge as a Bezier curve.
# One fixed waypoint preserves that clearance in the SVG renderer so it does
# not pass through the P5 and P6 plates.
SCHEMATIC_BENDS: dict[str, tuple[tuple[float, float], ...]] = {
    "L18": ((80.0, 31.5),),
}

#: Links the governed study and the guided story point at by name. The layout
#: keeps these legible and separable at presentation distance.
SIGNATURE_LINKS: dict[str, str] = {
    "L11": "Ankara–Kayseri backbone (2 Gbps). The link_failure scenario cuts this one.",
    "L20": "Kayseri–Samsun trunk (1 Gbps). demo_evening fails this one at 20:15.",
}

#: Discrete capacity weight classes. Capacity is encoded by line weight, and the
#: exact value is always available in inspection — never weight alone.
CAPACITY_CLASSES: tuple[dict[str, Any], ...] = (
    {"id": "trunk", "min_mbps": 2000, "label": "2 Gbps trunk", "stroke": 4.2},
    {"id": "backbone", "min_mbps": 1000, "label": "1 Gbps backbone", "stroke": 3.0},
    {"id": "regional", "min_mbps": 500, "label": "500 Mbps regional", "stroke": 2.1},
    {"id": "spur", "min_mbps": 0, "label": "250 Mbps spur", "stroke": 1.4},
)

#: Discrete utilization bands. Stepped, never a continuous rainbow, and every
#: band carries a printed value and a non-colour marker.
UTILIZATION_BANDS: tuple[dict[str, Any], ...] = (
    {"id": "quiet", "max": 0.50, "label": "under 50%", "state": "normal", "ticks": 0},
    {"id": "working", "max": 0.75, "label": "50–75%", "state": "normal", "ticks": 0},
    {"id": "loaded", "max": 0.90, "label": "75–90%", "state": "pressure", "ticks": 1},
    {"id": "congested", "max": 1.00, "label": "90–100%", "state": "pressure", "ticks": 2},
    {"id": "overloaded", "max": None, "label": "over 100%", "state": "failure", "ticks": 3},
)

GEOGRAPHIC_PRECISION = "engineering_schematic"
LAYOUT_NOTE = "Fixed engineering schematic for readability · not geographic"


def capacity_class(capacity_mbps: float) -> dict[str, Any]:
    for cls in CAPACITY_CLASSES:
        if capacity_mbps >= cls["min_mbps"]:
            return cls
    return CAPACITY_CLASSES[-1]


def utilization_band(utilization: float) -> dict[str, Any]:
    for band in UTILIZATION_BANDS:
        if band["max"] is None or utilization < band["max"]:
            return band
    return UTILIZATION_BANDS[-1]


def display_map(topology: Any) -> dict[str, Any]:
    """The complete display-only layout served by `GET /api/product/display-map`.

    Reads the scientific topology; never writes to it.
    """
    nodes = []
    for router_id, router in topology.routers.items():
        layout = NodeLayout(router.x / 8.0 + 5.0, router.y / 10.0,
                            LABEL_ANCHORS[router_id])
        nodes.append({
            "id": router_id,
            "city": CITY_NAMES[router_id],
            "role": router.role,
            "role_label": ROLE_LABELS.get(router.role, router.role),
            "role_token": ROLE_TOKENS.get(router.role, router.role),
            "title": f"{CITY_NAMES[router_id].upper()} · {ROLE_TOKENS.get(router.role, router.role)}",
            "x": layout.x,
            "y": layout.y,
            "label_anchor": layout.anchor,
        })
    links = []
    for link_id, link in topology.link_defs.items():
        cls = capacity_class(link.capacity_mbps)
        links.append({
            "id": link_id,
            "a": link.a,
            "z": link.z,
            "a_city": CITY_NAMES[link.a],
            "z_city": CITY_NAMES[link.z],
            "label": f"{CITY_NAMES[link.a]}–{CITY_NAMES[link.z]}",
            "technical": f"{link.a}–{link.z}, {link_id}",
            "capacity_mbps": link.capacity_mbps,
            "capacity_class": cls["id"],
            "stroke": cls["stroke"],
            "bends": [list(point) for point in SCHEMATIC_BENDS.get(link_id, ())],
            "signature": SIGNATURE_LINKS.get(link_id),
        })
    return {
        "geographic_precision": GEOGRAPHIC_PRECISION,
        "layout_note": LAYOUT_NOTE,
        "viewbox": [0, 0, 135, 63],
        "nodes": sorted(nodes, key=lambda n: (n["y"], n["x"])),
        "links": links,
        "capacity_classes": list(CAPACITY_CLASSES),
        "utilization_bands": list(UTILIZATION_BANDS),
        "role_labels": dict(ROLE_LABELS),
    }
