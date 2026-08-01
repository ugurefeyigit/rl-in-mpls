"""Presentation Mode: one control panel, a real exit, and honest wording.

Part 1's usability contract, pinned so it cannot quietly regress:

- every control that configures or drives a run is in the persistent left
  panel, in the documented order, and nothing that starts a run is hidden in
  the header, a bottom bar or a drawer;
- audience view always has a visible exit and Escape always leaves it;
- automatic execution never offers an approval affordance, and advisor
  execution never describes a held action as already applied;
- study evidence sits in its own named region with plain-language labels.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
PRODUCT = FRONTEND / "js" / "product"
APP = FRONTEND / "app.html"
PANEL = PRODUCT / "control-panel.js"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================== one panel
def test_the_control_panel_is_a_persistent_landmark_in_the_page():
    html = read(APP)
    assert '<aside class="control-panel" id="control-panel"' in html
    assert 'aria-label="Run setup and controls"' in html
    # It precedes the working column, so tab order reaches it first.
    assert html.index('id="control-panel"') < html.index('id="work"')


def test_the_panel_orders_the_controls_the_way_a_newcomer_needs_them():
    """Environment, scenario, seed, execution, controllers, root, speed, start."""
    panel = read(PANEL)
    order = ["Environment", "Scenario", "Seed", "Execution", "Controller A",
             "Compare two controllers", "Checkpoint root", "Speed", "Start run"]
    positions = [panel.index(f'"{label}"') if f'"{label}"' in panel
                 else panel.index(label) for label in order]
    assert positions == sorted(positions), dict(zip(order, positions))


def test_the_panel_sections_are_numbered_so_a_newcomer_can_be_told_where_to_start():
    panel = read(PANEL)
    for heading in ("1 · Set up the run", "2 · Run it", "4 · Guided Story",
                    "Study evidence and results"):
        assert heading in panel


def test_no_run_control_survives_outside_the_panel():
    html = read(APP)
    for legacy in ("cockpit", "btn-propose", "sel-speed", "btn-playpause"):
        assert f'id="{legacy}"' not in html
        assert f'class="{legacy}"' not in html


def test_the_panel_carries_every_reset_and_transport_control():
    panel = read(PANEL)
    for control in ("btn-playpause", "btn-step", "btn-next-event", "btn-stop",
                    "btn-reset-run", "btn-full-reset"):
        assert f'"{control}"' in panel


def test_the_panel_is_hidden_in_audience_view_and_outside_presentation():
    panel = read(PANEL)
    assert 'state.mode === "presentation" && !state.ui.audienceView' in panel


# =========================================================== reset semantics
def test_reset_run_and_full_reset_are_described_as_distinct_and_non_destructive():
    panel = read(PANEL)
    note = " ".join(panel.split("Reset run puts", 1)[1].split()[:70])
    assert "same scenario, seed and controllers" in note
    assert "keeps the run it replaces" in note
    assert "Full reset stops everything" in note
    assert "Neither changes a model or any" in note


def test_full_reset_clears_transient_state_and_closes_open_workflows():
    main = read(PRODUCT / "main.js")
    body = main.split("async function fullReset()", 1)[1].split("\n}", 1)[0]
    assert "liveApi.stop()" in body
    assert "closeDrawer()" in body
    assert "audienceView: false" in body
    assert "story:" in body and "active: false" in body
    assert "storyTimer" in body


def test_reset_run_keeps_the_configuration_and_the_previous_run():
    main = read(PRODUCT / "main.js")
    body = main.split("async function resetRun()", 1)[1].split("\n}", 1)[0]
    assert "liveApi.reset()" in body
    # Reset run does not restart from the setup form; it reuses the session.
    assert "liveApi.start" not in body


# ============================================================= audience view
def test_the_audience_exit_is_outside_the_chrome_that_audience_view_hides():
    html = read(APP)
    css = read(FRONTEND / "css" / "presentation-mode.css")
    assert 'id="btn-audience-exit"' in html
    head_tools = html.split('class="head-tools"', 1)[1].split("</div>", 1)[0]
    assert "btn-audience-exit" not in head_tools
    hide_rule = css.split('body[data-audience="on"] .head-tools', 1)[1].split(
        "}", 1)[0]
    assert ".audience-exit" not in hide_rule
    assert ".control-panel { display: none !important; }" in css.replace("\n", " ")


def test_the_audience_exit_is_pinned_visible_and_keyboard_reachable():
    css = read(FRONTEND / "css" / "presentation-mode.css")
    rule = css.split(".audience-exit {", 1)[1].split("}", 1)[0]
    assert "position: fixed" in rule
    assert "z-index" in rule
    assert "min-height: 44px" in rule
    assert ".audience-exit:focus-visible" in css


def test_escape_leaves_audience_view_before_fullscreen_and_restores_focus():
    shell = read(PRODUCT / "shell.js")
    block = shell.split('if (event.key === "Escape") {', 1)[1].split(
        "\n    }", 1)[0]
    assert block.index("audienceView") < block.index("fullscreenElement")
    assert "exitAudience()" in block
    assert 'btn-audience").focus()' in block
    # Leaving audience view is a state change, never a reload.
    assert "location.reload" not in shell


def test_the_keyboard_reference_documents_the_audience_escape():
    keys = read(PRODUCT / "contracts.js")
    row = keys.split('["Esc",', 1)[1].split("]", 1)[0]
    assert "audience view" in row


# ================================================ automatic vs advisor truth
def test_no_preview_recommendation_affordance_survives_anywhere():
    for path in (APP, PANEL, PRODUCT / "shell.js", PRODUCT / "main.js"):
        text = read(path)
        assert "Preview recommendation" not in text
        assert "View policy recommendation" not in text


def test_the_approval_controls_appear_only_in_advisor_execution():
    panel = read(PANEL)
    section = panel.split("function decisionSection", 1)[1].split(
        "\n/* ", 1)[0]
    assert 'session.execution !== "advisor"' in section
    automatic = section.split('session.execution !== "advisor"', 1)[1].split(
        "return el", 2)[1]
    assert "btn-approve" not in automatic
    assert "nothing to approve" in automatic


def test_a_completed_automatic_decision_is_worded_as_an_explanation():
    card = read(PRODUCT / "recommendation-card.js")
    assert 'proposal.kind === "proposal"' in card
    body = card.split("export function explanationFromDecision", 1)[1].split(
        "\n}", 1)[0]
    assert 'kind: "explanation"' in body
    assert "pending: false" in body
    # No fabricated preview for something that already ran.
    assert "expected: null" in body
    assert "Outcome estimate unavailable" in body


def test_a_held_proposal_says_nothing_has_been_applied():
    card = read(PRODUCT / "recommendation-card.js")
    assert "Nothing has been applied" in card
    assert "Awaiting your approval" in card


def test_the_bandit_output_is_labelled_from_declared_semantics():
    card = read(PRODUCT / "recommendation-card.js")
    body = card.split("function outputSemantics", 1)[1].split("\n}", 1)[0]
    assert "proposal?.outputSemantics" in body


# ================================================================ evidence
def test_study_evidence_has_its_own_named_region_in_the_panel():
    panel = read(PANEL)
    section = " ".join(panel.split("function evidenceSection", 1)[1].split())
    assert "Study evidence and results" in section
    assert 'source.group === "study_evidence"' in section
    assert "not simulation settings" in section
    assert "chosen as a model" in section


def test_the_panel_never_offers_evidence_as_a_controller_or_environment():
    panel = read(PANEL)
    setup = panel.split("function setupSection", 1)[1].split(
        "function policyOption", 1)[0]
    for word in ("evidence", "holdout", "Development", "Final Evidence"):
        assert word not in setup


def test_the_record_switch_uses_plain_language_not_a_bare_stamp():
    provenance = read(PRODUCT / "provenance.js")
    body = provenance.split("export function renderSourceSwitch", 1)[1]
    assert "plain_label" in body
    assert "Study result ·" in body


# ============================================== validation and unavailability
def test_an_invalid_seed_blocks_start_with_a_readable_reason():
    panel = read(PANEL)
    body = panel.split("function seedError", 1)[1].split("\n}", 1)[0]
    assert "whole number" in body
    assert "holdout_seeds_blocked_for_live" in body
    assert 'aria-invalid' in panel


def test_an_unavailable_controller_is_disabled_with_its_verification_reason():
    panel = read(PANEL)
    option = panel.split("function policyOption", 1)[1].split("\n}", 1)[0]
    assert "disabled: !policy.available" in option
    assert "unavailable" in option
    note = panel.split("function policyNote", 1)[1].split("\n}", 1)[0]
    assert "policy.unavailable_reason" in note


def test_the_start_button_is_blocked_while_anything_is_unrunnable():
    panel = read(PANEL)
    assert "const blocked = seedProblem" in panel
    assert "disabled: Boolean(blocked)" in panel


# ============================================================== responsive
def test_the_control_column_never_forces_a_horizontal_page_scroll():
    css = read(FRONTEND / "css" / "responsive.css")
    shell = read(FRONTEND / "css" / "shell.css")
    assert "html, body { overflow-x: hidden; }" in css
    assert "--control-width" in shell
    for width in ("1440px", "1280px"):
        assert width in css
    tablet = css.split("@media (max-width: 768px)", 1)[1]
    assert ".control-panel {" in tablet


def test_presentation_gives_the_topology_the_whole_work_column():
    shell = read(FRONTEND / "css" / "shell.css")
    rule = shell.split('body[data-mode="presentation"] .stage-wrap {', 1)[1].split(
        "}", 1)[0]
    assert "grid-template-columns: minmax(0, 1fr)" in rule
    assert "min-height: clamp(480px" in rule


@pytest.mark.parametrize("path", sorted(
    p for p in PRODUCT.rglob("*.js")))
def test_every_product_module_parses(path: Path):
    result = subprocess.run(["node", "--check", str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_no_module_reaches_the_network_outside_its_adapter():
    for module in PRODUCT.rglob("*.js"):
        if module.parent.name == "adapters":
            continue
        text = read(module)
        assert not re.search(r"\bfetch\(", text), module.name
        assert "XMLHttpRequest" not in text
