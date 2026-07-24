"""Central presentation display-name registry.

Internal IDs (PE1, P5, L11, D2, scenario keys, …) are the contract used by
the pretrained model, configs, tests and result files — they are NEVER
renamed. This module maps them to presentation labels: Turkish city names on
a FICTIONAL scaled national backbone (not a real operator topology — the UI
carries that disclaimer).

Everything user-facing derives its labels from this one registry, served to
both frontends via GET /api/display.
"""

from __future__ import annotations

from typing import Any

CITY_NAMES: dict[str, str] = {
    "PE1": "İstanbul", "PE2": "İzmir", "PE3": "Bursa", "PE4": "Antalya",
    "P1": "Eskişehir", "P2": "Ankara", "P3": "Konya", "P4": "Bolu",
    "P5": "Kayseri", "P6": "Adana", "P7": "Gaziantep", "P8": "Samsun",
    "A1": "Sivas", "A2": "Malatya",
    "PE5": "Trabzon", "PE6": "Erzurum", "PE7": "Diyarbakır", "PE8": "Van",
}

SCENARIO_NAMES: dict[str, str] = {
    "full_day": "Normal National Traffic Day",
    "morning_ramp": "Morning Business Ramp-Up",
    "evening_peak": "Evening Streaming Peak",
    "night_consolidation": "Quiet Night Operations",
    "flash_crowd": "Major Live Event Traffic Surge",
    "link_failure": "Ankara–Kayseri Backbone Failure",
    "demand_forecast_error": "Unforecast Demand Surge",
    "deceptive_local_optimum": "Hidden Shared Bottleneck",
    "demo_evening": "Guided Operator Demonstration",
    "random_day": "Randomized Training Day",
    "ood_double_failure": "Two-Link Regional Outage",
    "overload_stress": "Nationwide Capacity Overload",
}

CLASS_NAMES: dict[str, str] = {
    "voice": "voice", "video": "video", "vpn": "enterprise VPN",
    "besteffort": "consumer internet", "bulk": "bulk data", "critical": "critical services",
}

DISCLAIMER = ("Fictional scaled national backbone for demonstration — "
              "not a real operator topology.")

GLOSSARY: dict[str, str] = {
    "Traffic demand": "A group of data travelling between two cities.",
    "Route": "The sequence of cities carrying that traffic.",
    "Link utilization": "How much of a connection's capacity is being used.",
    "SLA problem": "A service is experiencing too much delay or packet loss.",
    "AI Advisor": "A trained policy that recommends which traffic route to change.",
    "Total reward": "The experiment's combined score for delivery, congestion, "
                    "service quality, and route stability. It is a simulation "
                    "score, not money or a direct industry KPI.",
    "Fast reroute": "Immediate use of a backup path after a link fails.",
    "Traditional controller": "A rule-based method that reacts to current congestion.",
}


def scale_mbps(value_mbps: float, factor: float) -> float:
    """Display-only scaling used by Presentation Mode's 'scaled national
    backbone' view. Applied identically to loads AND capacities so that
    utilization is invariant; never applied to delay, loss, SLA counts,
    reroutes or reward. The frontend mirrors this exact rule."""
    return value_mbps * factor


def city(router_id: str) -> str:
    return CITY_NAMES.get(router_id, router_id)


def path_label(routers: list[str] | tuple[str, ...]) -> str:
    """'İstanbul → Eskişehir → Kayseri → Samsun → Erzurum'"""
    return " → ".join(city(r) for r in routers)


def link_label(a: str, z: str) -> str:
    """'Ankara–Kayseri link' (technical id shown separately in tooltips)."""
    return f"{city(a)}–{city(z)} link"


def demand_label(src: str, dst: str, cls: str) -> str:
    """'İstanbul → Erzurum video traffic'"""
    return f"{city(src)} → {city(dst)} {CLASS_NAMES.get(cls, cls)} traffic"


def scenario_label(key: str) -> str:
    return SCENARIO_NAMES.get(key, key)


def display_bundle(topology: Any = None) -> dict[str, Any]:
    """Everything the frontends need, in one payload (GET /api/display)."""
    links = {}
    if topology is not None:
        for ld in topology.link_defs.values():
            links[ld.id] = {
                "label": link_label(ld.a, ld.z),
                "technical": f"{ld.a}–{ld.z}, {ld.id}",
            }
    return {
        "cities": CITY_NAMES,
        "scenarios": SCENARIO_NAMES,
        "classes": CLASS_NAMES,
        "links": links,
        "disclaimer": DISCLAIMER,
        "glossary": GLOSSARY,
    }
