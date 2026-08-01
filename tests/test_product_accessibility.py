"""Static accessibility and responsive contracts for the product shell."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _html() -> str:
    return (FRONTEND / "app.html").read_text(encoding="utf-8")


def _css() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in (FRONTEND / "css").glob("*.css")
    )


def test_shell_has_landmarks_skip_links_dialog_names_and_live_region():
    html = _html()
    assert html.count('class="skip-link"') >= 2
    for landmark in ("<header", "<nav", "<main", "<aside", "<footer"):
        assert landmark in html
    assert 'role="dialog"' in html and 'aria-modal="true"' in html
    assert 'id="live-region"' in html and 'aria-live="polite"' in html


def test_topology_has_keyboard_surface_and_synchronized_list_alternative():
    html = _html()
    assert 'id="atlas" role="application"' in html
    assert 'id="topology-list"' in html
    assert 'id="btn-topology-list" aria-expanded="false"' in html
    assert "Arrow keys move between cities" in html


def test_styles_define_visible_focus_reduced_motion_and_touch_targets():
    css = _css()
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "min-height: 44px" in css or "min-block-size: 44px" in css
    assert "min-width: 44px" in css or "min-inline-size: 44px" in css


def test_responsive_contract_covers_required_widths_without_page_overflow():
    css = _css()
    # 1920 and 1440 use the desktop base rules; narrower layouts have explicit
    # adaptations at the three specified collapse points.
    for width in (1280, 768, 390):
        assert str(width) in css
    assert "overflow-x: hidden" in css
    assert "overflow-x: auto" in css
    phone = css.split("@media (max-width: 390px)", 1)[1].split("@media print", 1)[0]
    assert ".atlas {" in phone and "overflow-x: auto" in phone
    assert "#atlas-svg" in phone and "min-width:" in phone


def test_mobile_shell_uses_page_flow_instead_of_squeezing_the_control_panel():
    css = (FRONTEND / "css" / "responsive.css").read_text(encoding="utf-8")
    tablet = css.split("@media (max-width: 768px)", 1)[1].split(
        "@media (max-width: 390px)", 1
    )[0]
    assert "height: auto" in tablet
    assert ".shell-main { overflow-y: visible; }" in tablet
    # Below 768 the control column stacks above the map rather than squeezing it,
    # and its controls keep a 44px touch target.
    assert "grid-template-columns: minmax(0, 1fr);" in tablet
    assert ".control-panel {" in tablet
    assert "min-height: 44px" in tablet


def test_phone_ledgers_wrap_without_native_scrollbar_strips():
    css = (FRONTEND / "css" / "responsive.css").read_text(encoding="utf-8")
    phone = css.split("@media (max-width: 390px)", 1)[1].split(
        "@media print", 1
    )[0]
    for selector in (".modes", ".source-switch__options", ".context", ".moment-rail__row"):
        assert selector in phone
    assert ".source-switch__options { display: grid; grid-template-columns: repeat(4" in phone
    assert ".context { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(6" in phone
    assert ".moment-rail__row { grid-template-columns: repeat(4" in phone
    assert ".moment-cell__value { white-space: normal;" in phone


def test_shell_grid_children_cannot_expand_the_page_from_intrinsic_content():
    css = (FRONTEND / "css" / "shell.css").read_text(encoding="utf-8")
    body_rule = css.split("\nbody {", 1)[1].split("}", 1)[0]
    assert "grid-template-columns: minmax(0, 1fr)" in body_rule
    assert ".shell-head, .ledger, .shell-main" in css
    shell_rule = css.split("\n.shell-main {", 1)[1].split("}", 1)[0]
    assert "grid-template-columns: minmax(0, 1fr)" in shell_rule
    work_rule = css.split("\n.work {", 1)[1].split("}", 1)[0]
    assert "grid-template-rows: auto auto auto auto auto" in work_rule
    assert ".shell-main > *" in css
    rule = css.split(".shell-main > *", 1)[1].split("}", 1)[0]
    assert "min-width: 0" in rule
    assert "width: 100%" in rule

    stage_rule = css.split(".stage-wrap", 1)[1].split("}", 1)[0]
    assert "min-height: clamp(" in stage_rule


def test_source_states_have_non_colour_words_and_patterns():
    html = _html()
    css = _css()
    for word in ("LIVE", "RECORDED", "DEVELOPMENT", "FINAL EVIDENCE"):
        assert word in html or word in (FRONTEND / "js" / "product" / "contracts.js").read_text(
            encoding="utf-8"
        )
    assert "repeating-linear-gradient" in css or "pattern" in css
