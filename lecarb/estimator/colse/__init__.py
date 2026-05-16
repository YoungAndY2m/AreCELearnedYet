"""lecarb estimator: CoLSE (Copula based Learned Selectivity Estimator).

Vendored from upstream https://github.com/<original-author>/CoLSE
(local mirror: Desktop/AllModels/CoLSE/, paper: Rathuwadu et al., 2024).

Layout:
  _vendor/colse/          full upstream `colse` package, untouched. Reached via sys.path
                          injection below so internal `from colse.X import Y` keeps working.
  _drivers_ref/           upstream driver scripts (dvine_copula_recursive_dynamic_v2.py +
                          residual_model_train.py + error_comp_network.py + default_args.py)
                          kept for reference; lecarb integration goes through wrapper.py.
  colse.py                lecarb-side adapter: train_colse / test_colse / load_colse /
                          class CoLSE(Estimator).
"""
import os
import sys

# Make the vendored `colse` package importable by L0's absolute imports
# (e.g. `from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive`).
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_ROOT = os.path.join(_HERE, "_vendor")
if _VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_ROOT)

# Re-export the lecarb-facing API from the wrapper.
from .colse import train_colse, test_colse, load_colse, CoLSE  # noqa: E402, F401
