# City display mapping

## The rule

**Internal IDs are never renamed.** `PE1`, `P5`, `L11`, `D2` and the scenario
keys are the contract shared by the pretrained model (`models/ppo_te`, obs dim
586, action dim 69), the YAML configs, the test suite and every committed
result file in `results/`. Renaming one would invalidate all of them.

City names are a **display layer only**. They exist in exactly one place —
`mplssim/display.py` — and are served to both frontends through
`GET /api/display`. No frontend hardcodes a city name.

```
mplssim/display.py          ← the only definition
        │
        └── GET /api/display
                    │
                    ├── frontend/js/display.js   (advanced console)
                    └── frontend/js/present.js   (presentation mode)
```

Where an internal ID still needs to be visible for an engineer — hover cards,
the decision tape's technical column, the recommendation card's footer, the
link and demand dropdowns — it appears in a dimmed monospace detail line
*alongside* the city label, never instead of it.

## Disclaimer

Shown in the footer of both UIs and in the printed summary:

> Fictional scaled national backbone for demonstration — not a real operator
> topology.

The topology is invented for this project. The Turkish city names were chosen
to make an 18-node graph readable on a projector; the geography carries no
engineering meaning, and link capacities and demand volumes are laboratory
values, not anyone's real network.

## Routers

| ID | Role | City |
|---|---|---|
| `PE1` | ingress PE | İstanbul |
| `PE2` | ingress PE | İzmir |
| `PE3` | ingress PE | Bursa |
| `PE4` | ingress PE | Antalya |
| `P1` | core | Eskişehir |
| `P2` | core | Ankara |
| `P3` | core | Konya |
| `P4` | core | Bolu |
| `P5` | core | Kayseri |
| `P6` | core | Adana |
| `P7` | core | Gaziantep |
| `P8` | core | Samsun |
| `A1` | aggregation | Sivas |
| `A2` | aggregation | Malatya |
| `PE5` | egress PE | Trabzon |
| `PE6` | egress PE | Erzurum |
| `PE7` | egress PE | Diyarbakır |
| `PE8` | egress PE | Van |

## Links

Link labels are derived, not enumerated: `link_label(a, z)` produces
`"{city(a)}–{city(z)} link"`, and the technical string is `"{a}–{z}, {id}"`.
Two that come up in the demo:

| ID | Endpoints | Display label | Technical |
|---|---|---|---|
| `L11` | P2–P5 | Ankara–Kayseri link | `P2–P5, L11` |
| `L20` | P5–P8 | Kayseri–Samsun link | `P5–P8, L20` |

`L20` is the link the guided demonstration fails at t=195 and repairs at
t=240. `L11` is the 2 Gbps backbone link behind the `link_failure` scenario.

Inside a sentence the trailing word "link" is dropped
(`"Ankara–Kayseri is running at 95 % of capacity"`); the full label is used as
a standalone noun.

## Traffic demands

Demands are named from their endpoints and class rather than listed:
`demand_label(src, dst, cls)` → `"İstanbul → Erzurum video traffic"`. The class
names are also display strings:

| Internal | Display |
|---|---|
| `voice` | voice |
| `video` | video |
| `vpn` | enterprise VPN |
| `besteffort` | consumer internet |
| `bulk` | bulk data |
| `critical` | critical services |

Routes render as city chains — `İstanbul → Eskişehir → Kayseri → Sivas →
Diyarbakır`. In tight single-line contexts (the decision tape) only the
**transit** cities are shown, because two candidate routes for the same demand
share their endpoints and the transit chain is what distinguishes them.

## Scenarios

| Key | Display name |
|---|---|
| `full_day` | Normal National Traffic Day |
| `morning_ramp` | Morning Business Ramp-Up |
| `evening_peak` | Evening Streaming Peak |
| `night_consolidation` | Quiet Night Operations |
| `flash_crowd` | Major Live Event Traffic Surge |
| `link_failure` | Ankara–Kayseri Backbone Failure |
| `demand_forecast_error` | Unforecast Demand Surge |
| `deceptive_local_optimum` | Hidden Shared Bottleneck |
| `demo_evening` | Guided Operator Demonstration |
| `random_day` | Randomized Training Day |
| `ood_double_failure` | Two-Link Regional Outage |
| `overload_stress` | Nationwide Capacity Overload |

## Controllers

Audience-facing names, defined in `frontend/js/display.js`:

| Internal | Display | Technical (engineering console) |
|---|---|---|
| `rl` | AI Advisor | RL (MaskablePPO) |
| `greedy` | Traditional controller | greedy (util-aware) |
| `static` | Fixed routing | static shortest path |
| `cspf` | CSPF re-optimizer | CSPF periodic reopt |
| `random` | Random baseline | random floor |

Unlike the city registry, these live on the JS side: they name *controllers*,
not simulation objects, so no server payload depends on them.

## Adding or changing a name

1. Edit `mplssim/display.py` only.
2. Run `python -m pytest tests/test_presentation.py -q` — it pins the bundle
   shape and a sample of the mapping.
3. Change nothing in `configs/`, `models/`, `results/` or any test that
   references an internal ID.
