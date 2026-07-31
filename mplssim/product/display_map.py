"""Display-only curated Turkey layout for the topology stage.

This registry is *presentation metadata*. `configs/topology.yaml` keeps the
scientific graph — router IDs, link IDs, endpoints, capacities, delays and
weights — and nothing here may change it. The x/y in the scientific config is a
schematic left-to-right diagram; these coordinates are a curated, geographically
recognizable arrangement of the same 18 routers so a room audience can read the
network as a national backbone.

The layout is **curated, not GIS**. West/east and regional relationships are
correct enough to be intuitive; absolute positions are not survey data, and the
product says so wherever network imagery appears.

Positions are fixed. There is no force-directed layout and nothing moves during
a session: a node that drifts between two steps destroys the one visual anchor
the whole product depends on.

Coordinates are normalized to a 0-100 box, y increasing downward (SVG order).
Bend points are display-only waypoints chosen so links clear node plates and so
the two links the study talks about most — `L11` Ankara-Kayseri and `L20`
Kayseri-Samsun — stay visually separable.
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


#: Curated positions, from the approved design specification (§9.1).
NODE_LAYOUT: dict[str, NodeLayout] = {
    "PE1": NodeLayout(8, 30, "left"),      # İstanbul
    "PE2": NodeLayout(7, 64, "left"),      # İzmir
    "PE3": NodeLayout(16, 45, "left"),     # Bursa
    "PE4": NodeLayout(28, 87, "below"),    # Antalya
    "P1": NodeLayout(28, 48, "below"),     # Eskişehir
    "P2": NodeLayout(41, 40, "above"),     # Ankara
    "P3": NodeLayout(41, 70, "below"),     # Konya
    "P4": NodeLayout(31, 31, "above"),     # Bolu
    "P5": NodeLayout(54, 53, "right"),     # Kayseri
    "P6": NodeLayout(55, 82, "below"),     # Adana
    "P7": NodeLayout(67, 79, "below"),     # Gaziantep
    "P8": NodeLayout(57, 21, "above"),     # Samsun
    "A1": NodeLayout(65, 48, "above"),     # Sivas
    "A2": NodeLayout(71, 62, "right"),     # Malatya
    "PE5": NodeLayout(71, 17, "above"),    # Trabzon
    "PE6": NodeLayout(83, 35, "right"),    # Erzurum
    "PE7": NodeLayout(80, 70, "below"),    # Diyarbakır
    "PE8": NodeLayout(94, 53, "right"),    # Van
}

#: Display-only waypoints. A link with no entry is drawn straight.
#: Each bend exists to clear a node plate or to make an unavoidable crossing
#: read as deliberate rather than accidental.
LINK_BENDS: dict[str, tuple[tuple[float, float], ...]] = {
    "L3": ((26, 58),),                       # İzmir-Ankara, south of Eskişehir
    "L4": ((28, 41),),                       # Bursa-Ankara, north of Eskişehir
    "L7": ((24, 37),),                       # İstanbul-Ankara, north of Bolu's approach
    "L12": ((46, 60),),                      # Ankara-Adana, west of Kayseri
    "L14": ((54, 73),),                      # Konya-Gaziantep, north of Adana
    "L15": ((45, 35),),                      # Bolu-Kayseri, north of Ankara
    "L18": ((36, 42), (52, 62), (60, 74)),   # Bolu-Gaziantep long detour arc
    "L30": ((74, 54),),                      # Sivas-Diyarbakır, east of Malatya
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

GEOGRAPHIC_PRECISION = "curated_not_gis"
LAYOUT_NOTE = "Curated geographic layout · not exact GIS"


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
        layout = NODE_LAYOUT[router_id]
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
            "bends": [list(p) for p in LINK_BENDS.get(link_id, ())],
            "signature": SIGNATURE_LINKS.get(link_id),
        })
    return {
        "geographic_precision": GEOGRAPHIC_PRECISION,
        "layout_note": LAYOUT_NOTE,
        "viewbox": [0, 0, 100, 100],
        "nodes": sorted(nodes, key=lambda n: (n["y"], n["x"])),
        "links": links,
        "capacity_classes": list(CAPACITY_CLASSES),
        "utilization_bands": list(UTILIZATION_BANDS),
        "role_labels": dict(ROLE_LABELS),
    }
