"""Product-layer contracts, catalogues and serializers.

Everything in this package is *presentation* infrastructure: it reads the
scientific modules (`mplssim.sim`, `mplssim.rl`, `mplssim.evidence`) and the
live session engines, and turns them into typed payloads the unified product
shell can render without inventing a value.

Nothing here defines science. No module in this package may change an
observation, action, mask, reward, scenario, seed, topology, checkpoint or
frozen artifact, and none of them writes under `results/` or `runs/`.
"""
