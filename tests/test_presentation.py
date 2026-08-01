"""Presentation-hardening tests: page smoke, websocket reconnect behaviour,
display-scale agreement between Python and JS, and the guarantee that starting
a presentation session never launches training.

These cover the frontend contract from the Python side: the pages must serve,
they must contain the element IDs the scripts bind to, and the one constant
that is duplicated in JavaScript must match its Python source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.display import display_bundle, scale_mbps
from server.main import STATE, WS_HUB, app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------- page smoke
# IDs the module scripts bind to at boot. If one is renamed in the HTML without
# updating the JS, the page throws on load and the presentation dies on stage —
# so the contract is pinned here.
PRODUCT_IDS = [
    "mode-nav", "mode-presentation", "mode-network", "mode-rl",
    "provenance-stamp", "context-ledger", "stage", "atlas-svg",
    "topology-list", "moment-rail", "timeband", "mode-surface",
    "panel-presentation", "panel-network", "panel-rl", "control-panel",
    "btn-audience", "btn-audience-exit", "btn-fullscreen",
    "recommendation", "drawer-explain", "drawer-help", "error-banner",
]


@pytest.mark.parametrize("path,ids,title", [
    ("/", PRODUCT_IDS, "National Backbone Dispatch Atlas"),
    ("/advanced", PRODUCT_IDS, "National Backbone Dispatch Atlas"),
    ("/present", PRODUCT_IDS, "National Backbone Dispatch Atlas"),
])
def test_pages_serve_with_expected_elements(client: TestClient, path, ids, title):
    r = client.get(path)
    assert r.status_code == 200
    html = r.text
    assert title in html
    missing = [i for i in ids if f'id="{i}"' not in html]
    assert not missing, f"{path} is missing element ids: {missing}"


def test_pages_reference_only_vendored_scripts(client: TestClient):
    """No CDN dependencies: this machine has no Node and the demo must run
    fully offline."""
    for path in ("/", "/present"):
        html = client.get(path).text
        for src in re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html):
            assert src.startswith("/static/") or src.startswith("/api/"), \
                f"{path} pulls an external asset: {src}"


def test_static_module_graph_resolves(client: TestClient):
    """Every relative import in the frontend modules must resolve to a file
    that the static mount actually serves."""
    js_dir = FRONTEND / "js"
    for mod in js_dir.rglob("*.js"):
        for imp in re.findall(r'from\s+"((?:\./|\.\./)[^"]+)"',
                              mod.read_text(encoding="utf-8")):
            target = (mod.parent / imp).resolve()
            assert target.exists(), f"{mod.relative_to(js_dir)} imports missing module {imp}"
            url = "/static/js/" + target.relative_to(js_dir).as_posix()
            assert client.get(url).status_code == 200


# ------------------------------------------------------------ display scale
def test_js_display_scale_matches_python():
    """Presentation Mode's 10x 'scaled national backbone' factor is duplicated
    in JavaScript (frontend/js/fmt.js). Keep the two in lockstep, and keep the
    pointer back to the Python source that defines the rule."""
    fmt = (FRONTEND / "js" / "fmt.js").read_text(encoding="utf-8")
    m = re.search(r"export const DISPLAY_SCALE\s*=\s*(\d+)", fmt)
    assert m, "fmt.js must export DISPLAY_SCALE"
    js_factor = int(m.group(1))
    assert js_factor == 10
    # the JS constant must point at the Python rule it mirrors
    assert "scale_mbps" in fmt
    # and the Python rule must behave the way the JS assumes: loads and
    # capacities scale together, so utilization is unchanged.
    load, cap = 137.5, 500.0
    assert scale_mbps(load, js_factor) / scale_mbps(cap, js_factor) == \
        pytest.approx(load / cap)


def test_display_bundle_has_every_label_the_frontends_use():
    from mplssim.factory import get_topology
    bundle = display_bundle(get_topology())
    for key in ("cities", "scenarios", "classes", "links", "disclaimer", "glossary"):
        assert key in bundle and bundle[key]
    assert bundle["cities"]["PE1"] == "İstanbul"
    assert bundle["scenarios"]["demo_evening"] == "Guided Operator Demonstration"
    assert bundle["links"]["L11"]["label"] == "Ankara–Kayseri link"
    assert bundle["links"]["L11"]["technical"] == "P2–P5, L11"


# --------------------------------------------------------- websocket reconnect
def test_ws_reconnect_does_not_duplicate_handling(client: TestClient):
    """A reconnecting client must receive exactly one tick per interval.

    Two sequential connections are opened. If the hub kept the dead socket, or
    if the session accumulated a second subscriber, the second client would see
    the same interval twice — so each received tick's step must advance by
    exactly one.
    """
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["static"], "seed": 5,
        "autostart": False, "environment": "v1", "model_tag": None})

    with client.websocket_connect("/ws/telemetry") as ws:
        assert ws.receive_json()["type"] == "tick"      # snapshot on connect
        client.post("/api/simulation/step")
        assert ws.receive_json()["runs"][0]["snapshot"]["step"] == 1

    # first socket is gone from the hub before the second one arrives
    assert WS_HUB.sockets == []

    with client.websocket_connect("/ws/telemetry") as ws2:
        first = ws2.receive_json()
        assert first["type"] == "tick"
        assert first["runs"][0]["snapshot"]["step"] == 1   # replayed state only
        assert len(WS_HUB.sockets) == 1
        for expected in (2, 3, 4):
            client.post("/api/simulation/step")
            msg = ws2.receive_json()
            # exactly one message per step: a duplicate would repeat `expected-1`
            assert msg["runs"][0]["snapshot"]["step"] == expected

    assert WS_HUB.sockets == []


def test_ws_receives_intervention_out_of_band(client: TestClient):
    """Interventions broadcast immediately without advancing the clock."""
    client.post("/api/simulation/start", json={
        "scenario": "full_day", "algorithms": ["static"], "seed": 5,
        "autostart": False, "environment": "v1", "model_tag": None})
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.receive_json()
        client.post("/api/simulation/step")
        tick = ws.receive_json()
        step_before = tick["runs"][0]["snapshot"]["step"]
        client.post("/api/failure/inject", json={"link": "L20"})
        msg = ws.receive_json()
        assert msg["type"] == "intervention"
        assert msg["runs"][0]["snapshot"]["step"] == step_before  # no time advance
        assert "L20" in msg["runs"][0]["snapshot"]["failed_links"]


# ------------------------------------------------------ no training on launch
def test_presentation_launch_never_touches_training(client: TestClient):
    """Starting a presentation session must not create a training job."""
    STATE["training"] = None
    r = client.post("/api/simulation/start", json={
        "scenario": "demo_evening", "algorithms": ["static"], "seed": 42,
        "autostart": False, "environment": "v1", "model_tag": None, "interface_mode": "present",
        "advisor": True})
    assert r.status_code == 200
    assert r.json()["interface_mode"] == "present"
    assert STATE["training"] is None

    # stepping, fast-forwarding and resetting must not either
    client.post("/api/simulation/step")
    client.post("/api/simulation/run-until",
                json={"condition": "congestion", "max_steps": 3, "util_threshold": 0.9})
    client.post("/api/simulation/reset")
    assert STATE["training"] is None

    progress = client.get("/api/training/progress").json()
    assert progress["active"] is False
    assert "allowed" in progress


def test_training_requires_confirmation(client: TestClient, monkeypatch):
    """The endpoint refuses without confirm=true, and refuses outright when the
    demo launcher has set ALLOW_TRAINING=false."""
    STATE["training"] = None
    monkeypatch.setenv("ALLOW_TRAINING", "true")
    r = client.post("/api/agent/train", json={"timesteps": 10, "tag": "unit_test"})
    assert r.status_code == 400 and "confirm" in r.json()["detail"]
    assert STATE["training"] is None

    monkeypatch.setenv("ALLOW_TRAINING", "false")
    r = client.post("/api/agent/train",
                    json={"timesteps": 10, "tag": "unit_test", "confirm": True})
    assert r.status_code == 403
    assert STATE["training"] is None
    assert client.get("/api/training/progress").json()["allowed"] is False


# ------------------------------------------------------------------ benchmark
def test_benchmark_reports_honest_winners(client: TestClient):
    """The benchmark endpoint is the source for both UIs' 'published results'
    panels. RL must NOT be reported as the winner everywhere — the panels lean
    on this to stay honest."""
    bench = client.get("/api/benchmark").json()
    winners = {k: v["winner"] for k, v in bench["scenarios"].items()}
    assert winners["full_day"] == "rl"
    assert winners["deceptive_local_optimum"] == "rl"
    assert winners["flash_crowd"] == "greedy"
    assert winners["link_failure"] == "greedy"
    assert set(winners.values()) != {"rl"}, "RL cannot win every scenario"
    for scen in bench["scenarios"].values():
        assert scen["display_name"] and scen["display_name"] != ""
        for stats in scen["algorithms"].values():
            assert stats["n_seeds"] == 5
