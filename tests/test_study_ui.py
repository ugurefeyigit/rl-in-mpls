"""The `/study` surface contract, checked from Python.

Same shape as `test_presentation.py`: the page must serve, it must contain the ids the
module script binds to, it must stay offline-capable, and — because this page renders a
closed scientific record — it must not carry a single scientific number of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

# Ids `study.js` binds to at boot. Renaming one in the markup without updating the
# script throws on load, so the contract is pinned here.
STUDY_IDS = [
    "mode-nav", "mode-presentation", "mode-network", "mode-rl",
    "provenance-stamp", "context-ledger", "source-switch", "stage",
    "atlas-svg", "topology-list", "mode-surface", "panel-rl",
    "drawer-conclusion", "error-banner", "live-region",
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_study_page_has_every_bound_element(client):
    html = client.get("/study").text
    missing = [i for i in STUDY_IDS if f'id="{i}"' not in html]
    assert not missing, f"/study is missing element ids: {missing}"


def test_study_page_serves_with_its_title(client):
    r = client.get("/study")
    assert r.status_code == 200
    assert "National Backbone Dispatch Atlas" in r.text


def test_study_page_uses_only_vendored_assets(client):
    """No CDN: the demo machine may be offline and a missing asset kills the page."""
    html = client.get("/study").text
    for src in re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html):
        assert src.startswith("/static/") or src.startswith("/api/"), \
            f"/study pulls an external asset: {src}"


def test_study_page_hardcodes_no_scientific_number():
    """Every figure must arrive from /api/v2. A literal in the markup would drift
    from the frozen evidence silently, which is the exact failure this page exists
    to prevent."""
    html = (FRONTEND / "study.html").read_text(encoding="utf-8")
    body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    for forbidden in ("18.221", "9.036", "9.185", "-2.327", "1.107", "20.183",
                      "2.148", "152.093", "87.09", "82.10"):
        assert forbidden not in body, f"study.html hardcodes {forbidden}"


def test_study_script_never_calls_a_mutating_endpoint():
    js = (FRONTEND / "js" / "study.js").read_text(encoding="utf-8")
    assert "POST" not in js.upper().replace("POSTURE", "")
    for banned in ("/api/agent/train", "/api/simulation/", "/api/export/save-run",
                   "/api/failure/", "/api/traffic/", "/api/advisor/"):
        assert banned not in js, f"study.js reaches a mutating endpoint: {banned}"


def test_study_script_reads_only_the_v2_evidence_api():
    js = (FRONTEND / "js" / "study.js").read_text(encoding="utf-8")
    endpoints = set(re.findall(r'"(/api/[^"]*)"', js))
    assert endpoints, "study.js must call the evidence API"
    for e in endpoints:
        assert e.startswith("/api/v2/"), f"study.js calls a non-evidence endpoint: {e}"


def test_replay_is_marked_recorded_and_never_live():
    html = (FRONTEND / "study.html").read_text(encoding="utf-8")
    js = (FRONTEND / "js" / "study.js").read_text(encoding="utf-8")
    assert "Recorded" in html or "RECORDED" in html
    assert "recorded_replay" in js
    # a live indicator on a replay would be a lie
    assert "live-dot" not in html and "is-live" not in html


def test_development_region_is_labelled_as_not_holdout():
    html = (FRONTEND / "study.html").read_text(encoding="utf-8").lower()
    assert "development" in html and "continuity" in html
    assert "not holdout" in html or "not final-holdout" in html


def test_page_declares_its_states_for_loading_empty_and_failure():
    html = (FRONTEND / "study.html").read_text(encoding="utf-8")
    for required in ('id="error-banner"', 'id="empty-state"', 'data-state="loading"'):
        assert required in html, f"study.html is missing {required}"


def test_stylesheet_honours_reduced_motion_and_visible_focus():
    css = (FRONTEND / "css" / "study.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css


def test_page_is_keyboard_reachable_and_landmarked():
    html = (FRONTEND / "study.html").read_text(encoding="utf-8")
    assert "skip-link" in html
    assert "<nav" in html and "<main" in html
    assert 'lang="en"' in html


def test_static_module_graph_still_resolves(client):
    js_dir = FRONTEND / "js"
    for mod in js_dir.glob("*.js"):
        for imp in re.findall(r'from\s+"(\./[^"]+)"', mod.read_text(encoding="utf-8")):
            target = (js_dir / imp[2:]).resolve()
            assert target.exists(), f"{mod.name} imports missing module {imp}"
            assert client.get(f"/static/js/{target.name}").status_code == 200


def test_legacy_frontend_routes_share_the_unified_shell(client):
    """The cutover preserves URLs while replacing the disconnected surfaces."""
    for path, marker in (("/", "National Backbone Dispatch Atlas"),
                         ("/present", "National Backbone Dispatch Atlas")):
        r = client.get(path)
        assert r.status_code == 200 and marker in r.text
