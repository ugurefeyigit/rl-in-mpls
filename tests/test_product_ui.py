"""Unified four-mode product-shell contracts.

These tests exercise the served routes and the complete static ES-module graph.
They intentionally avoid copying scientific values into frontend fixtures.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mplssim.product.contracts import FORBIDDEN_PRODUCT_PHRASES
from server.main import app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
APP = FRONTEND / "app.html"

SHELL_IDS = {
    "mode-nav", "mode-presentation", "mode-network", "mode-rl", "mode-compare",
    "provenance-stamp", "provenance-word", "context-ledger",
    "source-switch", "stage", "atlas-svg", "atlas-links", "atlas-nodes",
    "topology-list", "moment-rail", "timeband", "mode-surface",
    "panel-presentation", "panel-network", "panel-rl", "panel-compare", "control-panel",
    "drawer-explain", "drawer-help", "drawer-conclusion", "live-region",
    "btn-session-primary", "btn-audience-exit",
}

#: Controls the control panel renders once a state exists. They are asserted
#: against the module that builds them rather than against static HTML, because
#: the panel is rendered, not hand-written.
CONTROL_PANEL_IDS = [
    "cp-environment", "cp-scenario", "cp-seed", "cp-execution", "cp-policy-a",
    "cp-compare", "cp-speed", "cp-start",
    "btn-playpause", "btn-step", "btn-next-event",
    "btn-stop", "btn-reset-run", "btn-full-reset",
    "btn-approve", "btn-reject",
    "btn-story-toggle", "btn-story-auto", "btn-story-next", "btn-story-prev",
    "btn-conclusion", "btn-questions",
]


def test_every_run_control_lives_in_the_one_control_panel():
    """No core control is split across the header, a bottom bar or a drawer."""
    panel = (FRONTEND / "js" / "product" / "control-panel.js").read_text(
        encoding="utf-8")
    missing = [i for i in CONTROL_PANEL_IDS if f'"{i}"' not in panel]
    assert not missing, f"control panel is missing: {missing}"
    html = APP.read_text(encoding="utf-8")
    assert 'class="cockpit"' not in html
    assert 'id="cockpit"' not in html


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/", "/advanced", "/present", "/study", "/compare"])
def test_legacy_routes_serve_the_unified_dispatch_atlas(client: TestClient, path: str):
    response = client.get(path)
    assert response.status_code == 200
    assert "National Backbone Dispatch Atlas" in response.text
    assert '/static/js/product/main.js' in response.text
    missing = sorted(item for item in SHELL_IDS if f'id="{item}"' not in response.text)
    assert not missing, f"{path} is missing unified-shell IDs: {missing}"


def test_exactly_four_primary_modes_and_guided_story_is_nested():
    html = APP.read_text(encoding="utf-8")
    modes = re.findall(r'class="mode"[^>]+data-mode="([^"]+)"', html)
    assert modes == ["presentation", "network", "rl", "compare"]
    assert 'data-mode="guided-story"' not in html
    panel = (FRONTEND / "js" / "product" / "control-panel.js").read_text(
        encoding="utf-8")
    assert "Start Guided Story" in panel
    assert "Guided Story" not in re.sub(r"<nav class=\"modes\".*?</nav>", "",
                                        html, flags=re.S).split("<main", 1)[0]


def test_presentation_only_moment_rail_starts_hidden_without_a_flash():
    html = APP.read_text(encoding="utf-8")
    assert re.search(r'<section class="moment-rail" id="moment-rail"[^>]*\shidden>', html)


def test_topology_has_a_readable_non_color_legend():
    html = APP.read_text(encoding="utf-8")
    legend = re.search(r'<ul class="legend" id="atlas-legend".*?</ul>', html, re.S)
    assert legend
    for label in ("Normal", "Pressure", "Failed", "Current path", "Alternate path"):
        assert label in legend.group(0)


def test_backward_compatible_route_contract_is_explicit_in_client_router():
    contracts = (FRONTEND / "js" / "product" / "contracts.js").read_text(
        encoding="utf-8"
    )
    for literal in (
        '"/": { mode: "network", source: "live_session" }',
        '"/advanced": { mode: "network", source: "live_session" }',
        '"/present": { mode: "presentation", source: "live_session" }',
        '"/study": { mode: "rl", source: "final_holdout_evidence", rlView: "study" }',
    ):
        assert literal in contracts


def test_recursive_product_module_graph_resolves_through_static_mount(client: TestClient):
    product_root = FRONTEND / "js" / "product"
    imports = re.compile(r'from\s+["\']((?:\./|\.\./)[^"\']+)["\']')
    for module in product_root.rglob("*.js"):
        for relative in imports.findall(module.read_text(encoding="utf-8")):
            target = (module.parent / relative).resolve()
            assert target.exists(), f"{module.relative_to(FRONTEND)} imports {relative}"
            static_path = "/static/" + target.relative_to(FRONTEND).as_posix()
            assert client.get(static_path).status_code == 200


def test_product_copy_keeps_provenance_and_policy_semantics_distinct():
    product_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [APP, *(FRONTEND / "js" / "product").rglob("*.js")]
    )
    for required in (
        "LIVE", "RECORDED", "DEVELOPMENT", "FINAL EVIDENCE",
        "action probability", "action score", "Changed features",
        "descriptive change, not causal importance",
        "REFERENCE TOPOLOGY · NO RECORDED LINK TELEMETRY",
    ):
        assert required in product_text
    # Comments document the prohibited vocabulary so future maintainers know
    # why it must not ship. Check only executable/visible text.
    visible_text = re.sub(r"/\*.*?\*/|//[^\n]*", "", product_text, flags=re.S)
    lowered = visible_text.casefold()
    for phrase in FORBIDDEN_PRODUCT_PHRASES:
        if phrase.casefold() == "causal importance":
            assert not re.search(r"(?<!not )causal importance", lowered)
        else:
            assert phrase.casefold() not in lowered


def test_all_required_mode_stylesheets_are_local_and_present(client: TestClient):
    html = client.get("/").text
    styles = re.findall(r'<link[^>]+href="([^"]+)"', html)
    assert styles
    assert all(path.startswith("/static/") for path in styles)
    for path in styles:
        assert client.get(path).status_code == 200, path


def test_live_adapter_recognizes_the_real_session_status_contract():
    module = (FRONTEND / "js" / "product" / "adapters" / "live-v1.js").as_uri()
    script = f"""
      import {{ hasActiveSession }} from {module!r};
      if (!hasActiveSession({{session_id:'abc', state:'paused'}})) process.exit(2);
      if (!hasActiveSession({{session_id:'abc', state:'idle'}})) process.exit(4);
      if (hasActiveSession({{session:null, state:'idle'}})) process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_evidence_adapter_requires_an_explicit_matching_stage():
    module = (FRONTEND / "js" / "product" / "adapters" / "evidence-v2.js").as_uri()
    script = f"""
      import {{ assertStage, FINAL }} from {module!r};
      let missing = false, wrong = false;
      try {{ assertStage({{}}, FINAL); }} catch {{ missing = true; }}
      try {{ assertStage({{stage:'development'}}, FINAL); }} catch {{ wrong = true; }}
      if (!missing || !wrong) process.exit(2);
      if (assertStage({{stage:'final_holdout'}}, FINAL).stage !== 'final_holdout') process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_paused_step_zero_session_is_not_labelled_as_no_session():
    module = (FRONTEND / "js" / "product" / "provenance.js").as_uri()
    script = f"""
      import {{ detailFor }} from {module!r};
      const base = {{source:{{kind:'live_session'}}, context:{{step:0,hour:17}},
        data:{{snapshot:{{session:{{state:'idle',session_id:'abc'}}}}}}}};
      if (detailFor(base) !== 'idle · step 0 · 17:00') process.exit(2);
      base.data.snapshot = null;
      if (detailFor(base) !== 'No session running') process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_atlas_uses_the_registry_frame_without_reprojecting_nodes():
    module = (FRONTEND / "js" / "product" / "topology-atlas.js").as_uri()
    script = f"""
      import {{ atlasFrame }} from {module!r};
      const source = {{viewbox:[0,0,135,63], nodes:[{{id:'r1',x:20,y:30}}]}};
      const frame = atlasFrame(source);
      if (frame.join(' ') !== '0 0 135 63') process.exit(2);
      if (source.nodes[0].x !== 20 || source.nodes[0].y !== 30) process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_question_jumps_follow_the_declared_destination_contract():
    module = (FRONTEND / "js" / "product" / "help.js").as_uri()
    script = f"""
      import {{ QUESTIONS, questionDestination }} from {module!r};
      const byId = Object.fromEntries(QUESTIONS.map((q) => [q.id, questionDestination(q)]));
      if (byId['what-is-mpls'].mode !== 'network') process.exit(2);
      if (byId['why-this-action'].rlView !== 'decision') process.exit(3);
      if (byId['how-validated'].source !== 'final_holdout_evidence') process.exit(4);
      if (!byId['did-planning-help'].conclusion) process.exit(5);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_source_change_clears_live_only_state_and_story_bookmarks():
    module = (FRONTEND / "js" / "product" / "store.js").as_uri()
    script = f"""
      import {{ createStore, captureSource, isCurrentSource }} from {module!r};
      const store = createStore();
      const liveRequest = captureSource(store.state);
      store.patch({{
        context: {{sessionId:'live-1', step:9}},
        playback: {{state:'paused', running:false}},
        story: {{active:true, bookmarks:[{{id:'event-1'}}]}},
        data: {{timeline:{{events:[{{id:'event-1'}}]}}, comparison:{{paired:true}},
                recommendation:{{pending:true}}, decision:{{pipeline:[]}}}},
      }});
      store.setSource('recorded_replay');
      const state = store.state;
      if (isCurrentSource(state, liveRequest)) process.exit(7);
      if (state.data.timeline !== null || state.data.comparison !== null) process.exit(2);
      if (state.data.decision !== null || state.data.recommendation !== null) process.exit(3);
      if (state.story.active || state.story.bookmarks.length) process.exit(4);
      if (state.context.sessionId !== null || state.context.step !== null) process.exit(5);
      if (state.playback.state !== 'idle') process.exit(6);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_live_store_refuses_non_live_or_incomplete_provenance():
    module = (FRONTEND / "js" / "product" / "store.js").as_uri()
    script = f"""
      import {{ createStore }} from {module!r};
      const store = createStore();
      const base = {{session_id:'s', generation:0, sequence:0, step:0,
        environment_version:'v1', scenario:'demo_evening', policy_id:'greedy'}};
      if (store.acceptSnapshot({{provenance:{{...base,source_kind:'recorded_replay',live:false}}}})) process.exit(2);
      if (store.acceptSnapshot({{provenance:{{...base,source_kind:'live_session'}}}})) process.exit(3);
      if (!store.acceptSnapshot({{provenance:{{...base,source_kind:'live_session',live:true}},time:{{hour:17}}}})) process.exit(4);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_governed_study_never_mixes_development_and_final_regions():
    module = (FRONTEND / "js" / "product" / "governed-study.js").as_uri()
    script = f"""
      import {{ studyRegionsForSource }} from {module!r};
      const development = studyRegionsForSource('development_evidence');
      const final = studyRegionsForSource('final_holdout_evidence');
      const live = studyRegionsForSource('live_session');
      if (development.includes('final') || !development.includes('development')) process.exit(2);
      if (final.includes('development') || !final.includes('final')) process.exit(3);
      if (live.length !== 0) process.exit(4);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_governed_study_helpers_match_real_evidence_schema_and_fail_closed():
    module = (FRONTEND / "js" / "product" / "governed-study.js").as_uri()
    script = f"""
      import {{ holdoutRows, scenarioRows, noopBlocks, conclusionFindings }} from {module!r};
      const holdout = {{aggregate:[{{algorithm:'masked_bandit',operational_return_mean:18.2}}],
        conclusions:['frozen conclusion']}};
      if (holdoutRows(holdout)[0].algorithm !== 'masked_bandit') process.exit(2);
      const scenarios = scenarioRows({{scenarios:[{{scenario:'full_day',bandit:1,ppo:0,
        baselines:{{greedy:-2}}}}]}});
      if (scenarios[0].scenario !== 'full_day' || scenarios[0].greedy !== -2) process.exit(3);
      const blocks = noopBlocks({{noop:{{pooled_step_share:{{greedy:.6}},
        episode_mean_share:{{greedy:.5}}, pooled_grain:'x', steps_per_policy:10}}}}, {{
          step_pooled_noop_share:{{label:'pooled'}},
          episode_mean_noop_frequency:{{label:'episode'}}
        }});
      if (blocks.length !== 2 || blocks.some((b)=>!b.meta)) process.exit(4);
      if (conclusionFindings({{error:'outage',finalHoldout:holdout}}) !== null) process.exit(5);
      if (conclusionFindings({{finalHoldout:null}}) !== null) process.exit(6);
      if (conclusionFindings({{finalHoldout:holdout}})[0] !== 'frozen conclusion') process.exit(7);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_governed_study_renders_reward_components_and_seed42_payloads():
    module = (FRONTEND / "js" / "product" / "governed-study.js").read_text(
        encoding="utf-8"
    )
    assert "rewardRegion(rewardComponents)" in module
    assert "seed42?.methods" in module
    assert "payload.component_names" in module


def test_live_rl_route_round_trips_without_becoming_final_evidence():
    module = (FRONTEND / "js" / "product" / "router.js").as_uri()
    script = f"""
      import {{ locationForState, readLocation }} from {module!r};
      const state = {{mode:'rl', rlView:'decision', workflow:null,
        source:{{kind:'live_session'}}, selection:{{objectType:null,objectId:null,eventId:null}}}};
      const url = locationForState(state);
      const parsed = readLocation(new URL('http://local' + url));
      if (parsed.mode !== 'rl' || parsed.source !== 'live_session' || parsed.rlView !== 'decision') process.exit(2);
      state.source.kind = 'final_holdout_evidence'; state.rlView = 'study';
      if (locationForState(state) !== '/study') process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_latest_refresh_coordinator_replays_one_request_arriving_in_flight():
    module = (FRONTEND / "js" / "product" / "latest-refresh.js").as_uri()
    script = f"""
      import {{ coalesceLatest }} from {module!r};
      let calls = 0, release;
      const gate = new Promise((resolve)=>{{release=resolve;}});
      const refresh = coalesceLatest(async()=>{{ calls += 1; if (calls === 1) await gate; }}, ()=>true);
      const first = refresh(); const second = refresh(); release();
      await Promise.all([first,second]);
      if (calls !== 2) process.exit(2);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_guided_story_executes_the_declared_single_step_beat():
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    assert 'beat.advance?.kind === "step"' in main
    assert "await step()" in main


def test_guided_story_owns_a_real_demo_evening_session_contract():
    module = (FRONTEND / "js" / "product" / "guided-story.js").as_uri()
    script = f"""
      import {{ matchesStorySession, storySessionConfig }} from {module!r};
      const config = storySessionConfig();
      if (config.scenario !== 'demo_evening' || config.seed !== 42) process.exit(2);
      if (config.algorithms.join(',') !== 'masked_bandit,greedy') process.exit(3);
      if (config.environment !== 'v2') process.exit(7);
      if (!config.advisor || config.autostart) process.exit(4);
      const exact = {{session_id:'s', scenario:'demo_evening', seed:42,
        environment:'v2', advisor:true, algorithms:['masked_bandit','greedy']}};
      if (!matchesStorySession(exact)) process.exit(5);
      if (matchesStorySession({{...exact,seed:7}})) process.exit(6);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_guided_story_deep_link_forces_live_and_establishes_owned_session():
    router = (FRONTEND / "js" / "product" / "router.js").as_uri()
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    script = f"""
      import {{ readLocation }} from {router!r};
      const route = readLocation(new URL('http://local/present?workflow=guided-story&source=final_holdout_evidence'));
      if (route.workflow !== 'guided-story' || route.source !== 'live_session') process.exit(2);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "if (store.state.story.active) await ensureStorySession()" in main
    assert "if (!matchesStorySession(status))" in main


def test_live_refresh_consumes_one_atomic_moment_endpoint():
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    assert "const moment = await liveApi.moment()" in main
    assert "Promise.all([\n      liveApi.snapshot()" not in main


def test_guided_story_has_optional_automatic_playback_with_manual_equivalent():
    panel = (FRONTEND / "js" / "product" / "control-panel.js").read_text(
        encoding="utf-8")
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    assert '"btn-story-auto"' in panel
    assert '"btn-story-next"' in panel and '"btn-story-prev"' in panel
    assert '"btn-story-restart"' in panel
    assert "scheduleStoryAuto" in main
    assert "storyNext" in main


def test_automatic_story_playback_holds_at_a_pending_recommendation():
    """Automatic pacing must stop for approve or reject, never answer for you."""
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    schedule = main.split("function scheduleStoryAuto()", 1)[1].split("\n}", 1)[0]
    assert "recommendation?.pending" in schedule
    advance = main.split("async function storyNext()", 1)[1].split("\n}", 1)[0]
    assert "recommendation?.pending" in advance
    # Approve and reject both resume the schedule they interrupted.
    for name in ("async function approve()", "async function reject()"):
        body = main.split(name, 1)[1].split("\n}", 1)[0]
        assert "scheduleStoryAuto()" in body


def test_guided_story_failure_and_repair_beats_wait_for_the_real_states():
    story = (FRONTEND / "js" / "product" / "guided-story.js").read_text(
        encoding="utf-8"
    )
    assert 'condition: "failure"' in story
    assert 'condition: "recovery"' in story


def test_progressed_session_without_a_browser_prior_does_not_claim_first_interval():
    module = (FRONTEND / "js" / "product" / "modes" / "presentation.js").as_uri()
    script = f"""
      import {{ changeSentence }} from {module!r};
      const state = {{data:{{previousSnapshot:null, snapshot:{{
        time:{{step:49}}, metrics:{{available:true}},
      }}}}}};
      const text = changeSentence(state);
      if (!text.includes('browser snapshot') || text.includes('first interval')) process.exit(2);
      state.data.snapshot.time.step = 1;
      if (!changeSentence(state).includes('first completed interval')) process.exit(3);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_presentation_first_viewport_keeps_full_incident_and_no_duplicate_change_copy():
    css = (FRONTEND / "css" / "presentation-mode.css").read_text(encoding="utf-8")
    mode = (FRONTEND / "js" / "product" / "modes" / "presentation.js").read_text(
        encoding="utf-8"
    )
    assert "minmax(210px" in css
    assert ".moment-cell__value { display: block; overflow: visible;" in css
    assert 'text: "What changed"' not in mode


def test_conclusion_renderer_inserts_each_returned_node():
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    assert "replaceChildren(...renderConclusion(store.state))" in main


def test_history_navigation_revalidates_the_guided_story_session():
    main = (FRONTEND / "js" / "product" / "main.js").read_text(encoding="utf-8")
    assert "async function loadAppliedRoute()" in main
    assert "onNavigate(async (route) => { applyRoute(route); await loadAppliedRoute(); });" in main
    assert 'active: route.workflow === "guided-story"' in main
