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
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
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
    """

    def __init__(self, col_name: str, series: pd.Series, is_cat: bool, hist_bins: int = 5000):
        self.col_name = col_name
        self.is_cat = is_cat
        if is_cat:
            self._fit_categorical(series)
        else:
            self._fit_continuous(series, hist_bins)

    def _fit_continuous(self, x: pd.Series, B: int) -> None:
        clean = x.dropna().astype(np.float64).values
        counts, edges = np.histogram(clean, bins=B, density=False)
        N = len(clean)
        p = counts / float(N)
        cdf_bin = np.concatenate(([0.0], np.cumsum(p)))
        self._spline_cdf = PchipInterpolator(edges.astype(np.float64), cdf_bin, extrapolate=False)
        self._min = float(edges[0])
        self._max = float(edges[-1])

    def _fit_categorical(self, x: pd.Series) -> None:
        codes, uniques = pd.factorize(x, sort=True)
        if (codes < 0).any():
            # NaN bucket — re-factorize after dropping NaN, but remember NaN gets cdf=0
            codes = codes[codes >= 0]
        K = len(uniques)
        N = len(codes)
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
            continue
        op, val = pred
        if op == "[]":
            lb_val, ub_val = val[0], val[1]
        elif op in (">", ">="):
            lb_val, ub_val = val, float("inf")
        elif op in ("<", "<="):
            lb_val, ub_val = float("-inf"), val
        elif op == "=":
            lb_val, ub_val = val, val
        else:
            raise ValueError(f"Unsupported operator {op}")
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
        self.__dict__.update(kwargs)


def _model_filename(table: Table, workload: str, args: Args, seed: int) -> str:
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

    table = load_table(dataset, version)
    L.info(f"Loaded table {dataset}/{version}: {table.row_num} rows × {len(table.data.columns)} cols")

    # Stage-1a: per-column marginal CDFs
    start = time.time()
    column_cdfs = _build_column_cdfs(table)
    L.info(f"Fitted {len(column_cdfs)} per-column CDFs in {time.time()-start:.1f}s")

    # Stage-1b: theta_dict via Kendall-tau over column pairs (uses L0 ThetaStorage)
    n_cols = len(table.data.columns)
    start = time.time()
    theta_storage = ThetaStorage(DEFAULT_COPULA_TYPE, n_cols, parellel=False)
    # ThetaStorage.get_theta caches to a pickle file if cache_name is given;
    # we pass None to compute fresh (theta_dict is stashed inside the .pt anyway).
    theta_dict = theta_storage.get_theta(table.data, cache_name=None)
    L.info(f"Fitted theta_dict ({len(theta_dict)} pairs) in {time.time()-start:.1f}s")

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

    model_path = MODEL_ROOT / dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / _model_filename(table, workload, args, seed)

    # estimate footprint
    size_bytes = len(state["column_cdfs_pickle"]) + 8 * len(theta_dict)
    size_mb = size_bytes / 1024 / 1024
    if sizelimit > 0 and size_mb > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {size_mb:.2f}MB > {sizelimit} x {table.data_size_mb}, abort save")
        return
    L.info(f"CoLSE model size = {size_mb:.2f}MB (cdfs + theta), saving to {model_file}")
    torch.save(state, model_file)


def load_colse(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load CoLSE model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE, weights_only=False)

    table = load_table(dataset, state["version"])
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
        self.dvine = DivineCopulaDynamicRecursive(
            theta_dict=theta_dict, copula_type=copula_type, verbose=False
        )

    def query(self, query: Query) -> Tuple[float, float]:
        cdf_pairs, col_ids = _query_to_cdf_pairs(query, self.table, self.column_cdfs)
        if len(cdf_pairs) == 0:
            # no predicates → full table
            return float(self.row_num), 0.0

        start_stmp = time.time()
        cdf_arr = np.asarray(cdf_pairs, dtype=np.float64)
        y_bar = self.dvine.predict(cdf_arr, column_list=col_ids)
        dur_ms = (time.time() - start_stmp) * 1e3

        if y_bar is None or np.isnan(y_bar):
            est_card = 0.0
        else:
            est_card = max(float(y_bar) * self.row_num, 1.0)
        return est_card, dur_ms
