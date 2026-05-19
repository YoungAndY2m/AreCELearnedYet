# ============================================================================
# __init__.py (L2) — lecarb 的 colse package 入口 + sys.path 注入魔法
# ============================================================================
# (教学注释 by Claude, 不动原代码)
#
# 这文件做 2 件事:
#   1. sys.path 注入: 把 _vendor 目录加进模块搜索路径, 让 L0 vendored 代码
#      用 `from colse.X import Y` 这种 *绝对 import* 也能跑 (而不必改成
#      `from .X import Y` 相对 import — 因为 vendored 代码不能改).
#   2. Re-export: 把 wrapper.py 的 4 个 API 提到 package 级, 让 lecarb caller
#      可以写 `from lecarb.estimator.colse import train_colse`.
#
# `_vendor/colse/` 的内部 import 长这样:
#   from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive
# Python 默认从 sys.path 上的目录找 `colse` package. _vendor 加进 sys.path 后,
# `_vendor/colse/` 这个目录就变成了顶层 `colse` package, import 成功.
# ============================================================================
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
# __file__ = 本 __init__.py 路径; dirname = 当前 package 目录; abspath 转 abs
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_ROOT = os.path.join(_HERE, "_vendor")
# sys.path.insert(0, ...) 插到最前: 保证 _vendor/colse 优先于其它同名 colse package
if _VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_ROOT)

# Re-export the lecarb-facing API from the wrapper.
# 加 `noqa: E402, F401`: 禁用 linter 抱怨 "import 不在文件顶部" / "import 但未直接用"
# 因为 sys.path 注入必须先发生; 而且 re-export 的目的就是让外部能 `from . import X`
from .colse import train_colse, test_colse, load_colse, CoLSE  # noqa: E402, F401
