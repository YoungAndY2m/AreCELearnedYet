# ============================================================================
# colse.py (L2 wrapper) — lecarb 集成层: 把 L0 CoLSE 接进 ARELY benchmark
# ============================================================================
# (教学注释 by Claude, 不动原代码)
#
# 这是 ARELY/lecarb 这个 benchmark framework 集成 CoLSE 的 "adapter / wrapper".
# 角色对比:
#   - L0 ([AllModels/CoLSE/](../../../../AllModels/CoLSE/)): paper-faithful 完整研究系统, ~9583 行 76 文件
#   - L1 (ARELY standalone): **∅ 不存在** (LOG_STRUCTURE.md §10.1 详)
#   - L2 (本文件 + __init__.py + _vendor/): lecarb plug-in 适配层
#
# L2 设计取舍 (LOG_STRUCTURE.md §10.2):
#   - 绕开 L0 的 DatasetNames hardcoded enum → 用 lecarb 的 Table abstraction
#   - 自写一个 ColumnCDF (代替 SplineDequantizer 的 per-column CDF 部分)
#   - Theta storage 进 .pt 而非 pickle cache 文件
#   - Stage-2 ECN 暂关 (v1 TODO; v2 移植)
#   - 强制 ThetaStorage parellel=False, 跟 lecarb 的 NUM_THREADS=4 兼容
#
# lecarb 调用接口 (跟 mscn / naru / deepdb 等 estimator 完全一致):
#   - train_colse(seed, dataset, version, workload, params, sizelimit)
#   - test_colse(dataset, version, workload, params, overwrite)
#   - load_colse(dataset, model_name) -> (estimator, state)
#   - class CoLSE(Estimator) with .query(q) -> (est_card, dur_ms)
# 入口由 lecarb/__main__.py 通过 --estimator colse CLI 分发.
# ============================================================================
"""lecarb adapter for CoLSE.

Stage-1 only (D-vine Copula + per-column PCHIP CDF). The optional Stage-2
Error Compensation Network (ECN, see L0's residual_model_train.py) is wired
as a TODO hook — the v1 estimator works copula-only.

Lecarb interface contract:
  - train_colse(seed, dataset, version, workload, params, sizelimit)
  - test_colse(dataset, version, workload, params, overwrite)
  - load_colse(dataset, model_name) -> (estimator, state)
  - class CoLSE(Estimator) with .query(q) -> (est_card, dur_ms)

Why a thin per-column CDF helper instead of L0's SplineDequantizer:
  L0's SplineDequantizer is coupled to `dataset_names.DatasetNames` (which
  declares per-dataset categorical-column lists) and to L0's cache layout
  (`DataPathDir.CDF_CACHE`). For lecarb we derive the same info from the
  Table object directly and stash CDF parameters in the .pt checkpoint, so
  CoLSE generalizes to any lecarb-registered dataset without registering a
  new DatasetNames enum value.
"""
import time
import logging
import pickle
# typing.Union 不直接 import — 这文件用 typing.Dict/Any/Tuple/List 这种 3.8-compatible 形式
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
# PchipInterpolator: scipy 单调 cubic Hermite 样条; 同 L0 SplineDequantizer 用的 CDF 拟合器
from scipy.interpolate import PchipInterpolator

import torch

from ..estimator import Estimator, OPS
from ..utils import qerror, evaluate, run_test
from ...dataset.dataset import load_table, Table
from ...workload.workload import load_queryset, load_labels, query_2_triple, Query
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS
from ...dtypes import is_categorical

# Imports from the vendored `colse` package (resolves via sys.path injection
# in __init__.py).
# 这 4 行从 vendored L0 复用算法核心: D-vine 主算法 + 类型 enum + θ 算法 + ECN MLP
from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive
from colse.copula_types import CopulaTypes
from colse.theta_storage import ThetaStorage
from colse.error_comp_model import ErrorCompModel  # for v2 ECN extension

L = logging.getLogger(__name__)

DEFAULT_COPULA_TYPE = CopulaTypes.GUMBEL


# ---------------------------------------------------------------------------
# Per-column CDF (lecarb-native, mimics L0 SplineDequantizer per-column logic
# but driven by lecarb's Column abstraction)
# ---------------------------------------------------------------------------

class ColumnCDF:
    """Per-column marginal CDF estimator.

    For numeric columns: histogram (5000 bins) → cumulative → PCHIP interpolator.
    For categorical columns: discretize → cumulative frequency table.

    Mirrors L0's SplineDequantizer._fit_continuous_column / _fit_single_column
    but is callable per-column instead of per-dataset.

    设计动机 (LOG_STRUCTURE.md §10.2.2):
      L0 `SplineDequantizer` 跟 `DatasetNames` 强耦合 (要知道每个 dataset 哪些列
      是 categorical). lecarb 用通用 `Table` 抽象 + `is_categorical(dtype)` 自动
      判断, 不需要每个 dataset 注册 enum. 所以 L2 这里 *只复用 L0 算法逻辑*
      (5000-bin histogram + PCHIP), 但接口完全重写.
    """

    def __init__(self, col_name: str, series: pd.Series, is_cat: bool, hist_bins: int = 5000):
        # __init__ 中根据 is_cat dispatch 到不同 fit 路径
        self.col_name = col_name
        self.is_cat = is_cat
        if is_cat:
            self._fit_categorical(series)
        else:
            self._fit_continuous(series, hist_bins)

    def _fit_continuous(self, x: pd.Series, B: int) -> None:
        """连续列拟合: histogram → 累积概率 → PCHIP. 同 L0 _fit_continuous_column."""
        # .dropna() 去 NaN; .astype(np.float64) 统一 dtype; .values → numpy ndarray
        clean = x.dropna().astype(np.float64).values
        # np.histogram(x, bins=B): 返回 (counts shape (B,), edges shape (B+1,))
        counts, edges = np.histogram(clean, bins=B, density=False)
        N = len(clean)
        p = counts / float(N)
        cdf_bin = np.concatenate(([0.0], np.cumsum(p)))
        # PchipInterpolator: monotone cubic; extrapolate=False → 范围外返回 NaN (调用方 clamp)
        self._spline_cdf = PchipInterpolator(edges.astype(np.float64), cdf_bin, extrapolate=False)
        self._min = float(edges[0])
        self._max = float(edges[-1])

    def _fit_categorical(self, x: pd.Series) -> None:
        """Categorical 列拟合: factorize → cumulative frequency, 不走 spline."""
        # pd.factorize(x, sort=True): (codes ∈ [0, K-1], uniques) — codes -1 表示 NaN
        codes, uniques = pd.factorize(x, sort=True)
        if (codes < 0).any():
            # NaN bucket — re-factorize after dropping NaN, but remember NaN gets cdf=0
            codes = codes[codes >= 0]
        K = len(uniques)
        N = len(codes)
        # np.bincount(codes, minlength=K): 每个 code 出现次数, shape (K,)
        counts = np.bincount(codes, minlength=K)
        p = counts.astype(np.float64) / float(N)
        cdf_vals = np.cumsum(p)
        self._cat_mapping = {val: idx for idx, val in enumerate(uniques)}
        self._cat_cdf_vals = cdf_vals  # cdf_vals[code] = P(X <= category[code])

    def cdf(self, value, lower_bound: bool) -> float:
        """Return F(value). For lower_bound=True (i.e., evaluating for a lower predicate boundary),
        categorical returns the strict-less cdf to mimic L0's `ub=False` behavior."""
        if value is None:
            return 0.0 if lower_bound else 1.0
        try:
            v_float = float(value)
            if v_float == float("-inf"):
                return 0.0
            if v_float == float("inf"):
                return 1.0
        except (TypeError, ValueError):
            # string-typed categorical 转 float 失败, 记为 None
            v_float = None

        if self.is_cat:
            if value in self._cat_mapping:
                code = self._cat_mapping[value]
            elif v_float is not None and v_float in self._cat_mapping:
                code = self._cat_mapping[v_float]
            else:
                # unseen → conservative
                return 0.0 if lower_bound else 1.0
            if lower_bound:
                return 0.0 if code == 0 else float(self._cat_cdf_vals[code - 1])
            return float(self._cat_cdf_vals[code])

        # continuous
        if v_float is None:
            return 0.0 if lower_bound else 1.0
        if v_float <= self._min:
            return 0.0
        if v_float >= self._max:
            return 1.0
        out = float(self._spline_cdf(v_float))
        return float(np.clip(out, 0.0, 1.0))


def _build_column_cdfs(table: Table) -> Dict[str, ColumnCDF]:
    """对 lecarb Table 的每列建一个 ColumnCDF, 集中存进 dict[col_name → cdf].

    is_categorical(dtype) 是 lecarb 内置 helper, 替代 L0 的 DatasetNames.get_non_continuous_columns()
    (后者需要 hardcoded per-dataset categorical 列表).
    """
    cdfs: Dict[str, ColumnCDF] = {}
    for col_name in table.data.columns:
        is_cat = is_categorical(table.data[col_name].dtype)
        cdfs[col_name] = ColumnCDF(col_name, table.data[col_name], is_cat)
    return cdfs


# ---------------------------------------------------------------------------
# Predicate → (lb, ub) CDF pairs
# ---------------------------------------------------------------------------

def _query_to_cdf_pairs(query: Query, table: Table, cdfs: Dict[str, ColumnCDF]
                       ) -> Tuple[List[float], List[int]]:
    """Convert a lecarb Query into the (cdf_lb, cdf_ub, ...) flat list expected
    by DivineCopulaDynamicRecursive.predict, plus the 1-indexed column id list."""
    col_names = list(table.data.columns)
    col_to_idx = {c: i + 1 for i, c in enumerate(col_names)}  # L0 expects 1-indexed columns

    cdf_pairs: List[float] = []
    col_ids: List[int] = []
    for c, pred in query.predicates.items():
        if pred is None:
            # None = 该列无 predicate, 跳过 (D-vine 也不需要该列)
            continue
        op, val = pred
        if op == "[]":
            # Range query: (lb, ub) tuple
            lb_val, ub_val = val[0], val[1]
        elif op in (">", ">="):
            # Open interval to +inf
            lb_val, ub_val = val, float("inf")
        elif op in ("<", "<="):
            # Open interval from -inf
            lb_val, ub_val = float("-inf"), val
        elif op == "=":
            # Point query: lb=ub
            lb_val, ub_val = val, val
        else:
            raise ValueError(f"Unsupported operator {op}")
        # cdf(lb, lower_bound=True): strict less (=  P(X < lb))
        # cdf(ub, lower_bound=False): less or equal (= P(X <= ub))
        cdf_lb = cdfs[c].cdf(lb_val, lower_bound=True)
        cdf_ub = cdfs[c].cdf(ub_val, lower_bound=False)
        cdf_pairs.append(cdf_lb)
        cdf_pairs.append(cdf_ub)
        col_ids.append(col_to_idx[c])

    return cdf_pairs, col_ids


# ---------------------------------------------------------------------------
# Train / Test / Load
# ---------------------------------------------------------------------------

class Args:
    """Hyperparameter container (lecarb pattern, see mscn.py Args)."""

    def __init__(self, **kwargs):
        self.copula = "gumbel"
        self.theta_cache = True  # cache theta_dict to avoid Kendall-tau recomputation
        self.ecn_epochs = 0      # 0 = skip Stage-2 ECN training (copula-only inference)
        self.ecn_hid = "256_256_128_64"
        self.ecn_bs = 32
        self.ecn_lr = 0.001
        # 让 user 通过 --params '{"copula": "frank"}' 覆盖任何默认值
        self.__dict__.update(kwargs)


def _model_filename(table: Table, workload: str, args: Args, seed: int) -> str:
    """组装 .pt 文件名约定: <version>_<workload>-colse_<copula>_ecn<N>-<seed>.pt."""
    return (f"{table.version}_{workload}-colse_{args.copula}"
            f"_ecn{args.ecn_epochs}-{seed}.pt")


def train_colse(seed, dataset, version, workload, params, sizelimit):
    """Stage-1: fit per-column CDFs + theta_dict for all column pairs.
    Stage-2 (ECN) is skipped when args.ecn_epochs == 0 (default)."""
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    args = Args(**(params or {}))
    L.info(f"params: {params}")

    # 加载 lecarb Table (.data = DataFrame, .row_num = N, .data_size_mb = 大小)
    table = load_table(dataset, version)
    L.info(f"Loaded table {dataset}/{version}: {table.row_num} rows × {len(table.data.columns)} cols")

    # Stage-1a: per-column marginal CDFs
    # 等价 L0 SplineDequantizer.fit, 但用本文件自写的 ColumnCDF
    start = time.time()
    column_cdfs = _build_column_cdfs(table)
    L.info(f"Fitted {len(column_cdfs)} per-column CDFs in {time.time()-start:.1f}s")

    # Stage-1b: theta_dict via Kendall-tau over column pairs (uses L0 ThetaStorage)
    # 直接调用 L0 ThetaStorage (vendored), 不重新实现
    n_cols = len(table.data.columns)
    start = time.time()
    # parellel=False (typo 是 L0 固化的): 强制单线程避免跟 NUM_THREADS=4 冲突
    theta_storage = ThetaStorage(DEFAULT_COPULA_TYPE, n_cols, parellel=False)
    # ThetaStorage.get_theta caches to a pickle file if cache_name is given;
    # we pass None to compute fresh (theta_dict is stashed inside the .pt anyway).
    theta_dict = theta_storage.get_theta(table.data, cache_name=None)
    L.info(f"Fitted theta_dict ({len(theta_dict)} pairs) in {time.time()-start:.1f}s")

    # === Pack 所有 fit 结果进 state dict, 准备 torch.save 进 .pt ===
    # 注意 column_cdfs 单独 pickle 是因为 torch.save 不能直接序列化任意 Python 对象
    # (scipy PchipInterpolator), 必须先 pickle.dumps → bytes 再让 torch save bytes
    state: Dict[str, Any] = {
        "seed": seed,
        "args": args.__dict__,
        "dataset": dataset,
        "version": version,
        "workload": workload,
        "row_num": table.row_num,
        "col_names": list(table.data.columns),
        "theta_dict": theta_dict,
        "copula_type": DEFAULT_COPULA_TYPE,
        "column_cdfs_pickle": pickle.dumps(column_cdfs),
        "ecn_state": None,  # filled by Stage-2 if enabled (TODO)
    }

    # Stage-2: optional ECN residual training
    # v1 实现: 仅 warn, 不真训练. v2 把 _drivers_ref/residual_model_train.py 的
    # train_lw_nn 函数移植成 in-memory training (无需写 xlsx 中转), 详 LOG_STRUCTURE.md §10.2.4.
    if args.ecn_epochs > 0:
        L.warning(
            "ECN training (args.ecn_epochs>0) is wired as a TODO in this v1 adapter. "
            "See _drivers_ref/residual_model_train.py for the upstream loop. "
            "Falling back to copula-only inference."
        )
        # TODO(v2): port residual_model_train.train_lw_nn() to lecarb here.
        #   1. For each query in queryset['train'], compute (cdf_list, y_bar, y_actual)
        #   2. Build feature: norm_query (paired min-max) + log y_bar + log AVI estimate
        #   3. Train ErrorCompModel(fea_num, "256_256_128_64", output_len=3) with
        #      sign(BCEWithLogitsLoss) + abs(MSELoss) custom loss
        #   4. Save model_state_dict + fea_num + max/min_values into state["ecn_state"]

    # 落盘: MODEL_ROOT/{dataset}/{model_filename}.pt
    model_path = MODEL_ROOT / dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / _model_filename(table, workload, args, seed)

    # estimate footprint
    # column_cdfs pickle bytes + 8 bytes / θ (float64)
    size_bytes = len(state["column_cdfs_pickle"]) + 8 * len(theta_dict)
    size_mb = size_bytes / 1024 / 1024
    if sizelimit > 0 and size_mb > (sizelimit * table.data_size_mb):
        # lecarb 约定: 模型超过 dataset 大小的 sizelimit 倍数, 视为"太大不公平", 不保存
        L.info(f"Exceeds size limit {size_mb:.2f}MB > {sizelimit} x {table.data_size_mb}, abort save")
        return
    L.info(f"CoLSE model size = {size_mb:.2f}MB (cdfs + theta), saving to {model_file}")
    torch.save(state, model_file)


def load_colse(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    """Load .pt → 还原 (estimator, state). 标准 lecarb load 入口.

    流程: 反 pickle column_cdfs → 重建 CoLSE estimator → return.
    """
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load CoLSE model from {model_file} ...")
    # weights_only=False: 允许 pickle 反序列化任意对象 (PyTorch 2.x 安全 flag)
    state = torch.load(model_file, map_location=DEVICE, weights_only=False)

    table = load_table(dataset, state["version"])
    # pickle.loads(bytes) → ColumnCDF dict (反向 column_cdfs_pickle)
    column_cdfs: Dict[str, ColumnCDF] = pickle.loads(state["column_cdfs_pickle"])
    theta_dict = state["theta_dict"]
    copula_type = state["copula_type"]

    estimator = CoLSE(
        table=table,
        model_name=model_name,
        column_cdfs=column_cdfs,
        theta_dict=theta_dict,
        copula_type=copula_type,
        row_num=state["row_num"],
    )
    return estimator, state


def test_colse(dataset: str, version: str, workload: str,
               params: Dict[str, Any], overwrite: bool) -> None:
    """Build CoLSE estimator from saved model and run lecarb's standard test harness."""
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    estimator, _state = load_colse(dataset, params["model"])
    L.info(f"built CoLSE estimator: {estimator}")
    run_test(dataset, version, workload, estimator, overwrite)


# ---------------------------------------------------------------------------
# Estimator (per-query inference)
# ---------------------------------------------------------------------------

class CoLSE(Estimator):
    """Single-table CE via D-vine Copula factorization over per-column marginal CDFs."""

    def __init__(self,
                 table: Table,
                 model_name: str,
                 column_cdfs: Dict[str, ColumnCDF],
                 theta_dict: Dict[Tuple[int, int], float],
                 copula_type: CopulaTypes,
                 row_num: int):
        super().__init__(table=table, model=model_name)
        self.column_cdfs = column_cdfs
        self.theta_dict = theta_dict
        self.copula_type = copula_type
        self.row_num = row_num
        # 直接 reuse L0 D-vine inference 类 (vendored, 0 changes)
        self.dvine = DivineCopulaDynamicRecursive(
            theta_dict=theta_dict, copula_type=copula_type, verbose=False
        )

    def query(self, query: Query) -> Tuple[float, float]:
        """单 query inference 入口 (lecarb 标准签名).

        Returns:
            est_card: 估计的 cardinality (≥ 1, 即使 sel=0 也输出 1)
            dur_ms: 推理耗时 (毫秒)
        """
        cdf_pairs, col_ids = _query_to_cdf_pairs(query, self.table, self.column_cdfs)
        if len(cdf_pairs) == 0:
            # no predicates → full table
            # 没有谓词 (e.g. SELECT *) → 全表 N 行, dur 0
            return float(self.row_num), 0.0

        start_stmp = time.time()
        # numpy array, dtype float64 (D-vine 内部用 numpy 算)
        cdf_arr = np.asarray(cdf_pairs, dtype=np.float64)
        # 调 L0 D-vine recursive 主算法 → selectivity ∈ [EPSILON, 1]
        y_bar = self.dvine.predict(cdf_arr, column_list=col_ids)
        dur_ms = (time.time() - start_stmp) * 1e3

        if y_bar is None or np.isnan(y_bar):
            # numerical edge case (very rare); 防 lecarb 报告 NaN
            est_card = 0.0
        else:
            # max(., 1.0): 即使 sel=0 也至少输出 1 行 (CE 论文标准做法)
            est_card = max(float(y_bar) * self.row_num, 1.0)
        return est_card, dur_ms
