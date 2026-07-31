# Accessibility and keyboard reference

The unified UI targets WCAG 2.1 AA. Accessibility behavior is shared by all
three modes and all four record types.

## Keyboard

- Skip links move directly to the topology or current mode surface.
- `Alt+1`, `Alt+2`, and `Alt+3` select the three primary modes.
- Arrow keys move through routers in stable schematic reading order; Enter or
  Space selects the focused router.
- List view is a complete non-graphical topology alternative and shares the
  same selection.
- `/` focuses the active Network or observation search.
- `Space`, `→`, `←`, `G`, `E`, `[`, `]`, `?`, and `Esc` have the Presentation
  behaviors listed in [PRESENTATION_MODE.md](PRESENTATION_MODE.md).
- Shortcuts do not fire while a text field, select, or other typing control has
  focus. Every shortcut has a visible control.

Drawers trap focus while open, close with Escape, and restore focus to their
invoker. Mode changes focus the new mode surface. The interface uses visible
`:focus-visible` outlines and a logical DOM order matching visual order.

## Semantics

The header, primary navigation, main region, topology region, context rail,
tables, time band, presenter cockpit, drawers, and live status use named
landmarks or equivalent native elements. Tables include captions, column
headers, and row headers. Source changes and errors retain visible text; the
polite live region announces status without replacing on-screen information.

Color is never the only state signal. Provenance uses word, icon, border, and
pattern. Link pressure adds ticks, failure uses a dashed line and break marker,
selection uses structure and `aria-selected`, and legends print every state.

## Motion

Control transitions are 120–180 ms, panels 200–320 ms, and one-shot topology
events 400–800 ms. There is no ambient, bouncing, parallax, typewriter, animated
background, or unstable-node motion. Under `prefers-reduced-motion: reduce`,
durations collapse and the proposed-route drawing animation is removed. No
information depends on animation.

## Responsive containment

The application has deliberate breakpoints at 1280, 768, and 390 CSS pixels.
At 768 and below the context rail follows the topology, touch targets reach
44×44 px, and drawers become bottom sheets. Wide timelines, tables, observation
pipelines, and story beats own their horizontal scrolling; the page itself does
not scroll horizontally. At 390 px, controls and facts stack while all primary
modes remain reachable.
