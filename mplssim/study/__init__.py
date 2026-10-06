"""Post-V2 research study: sequentiality diagnostics, extended learners and
controlled-coupling experiments on the frozen MPLS-TE V2 environment.

Nothing in this package modifies a frozen V2 definition
(:data:`mplssim.experiments.v2_factory.FROZEN_DEFINITION_PATHS`). Environment
variants that change the decision problem (for example delayed TE
activation) live in :mod:`mplssim.study.variants` and carry their own version
identity; with their coupling parameter at zero they are tested to reproduce
the frozen environment exactly.
"""
