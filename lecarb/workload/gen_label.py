# ================================================================
# 教学注释 (annotation pass) — gen_label.py 总览
# ================================================================
# 给已有 workload 生成 ground truth labels (cardinality + selectivity)。
# 两种生成方式:
#   - generate_labels: 用 Oracle (= 扫全表算精确真值), 准但慢
#   - update_labels:   用 Sampling 估计 (= 在 sample 上算, 快得多但有误差)
#                      给 data shift 实验快速更新 label 用 (新 version 数据真扫一遍太贵)
# 由 lecarb CLI 触发 (`lecarb label ...` / `lecarb update-label ...`)。
# ================================================================
import logging
from typing import List, Dict

from .workload import Label, Query, load_queryset, dump_labels
from ..estimator.estimator import Oracle
from ..estimator.sample import Sampling
from ..dataset.dataset import Table, load_table

L = logging.getLogger(__name__)

# ================================================================
# generate_labels_for_queries: 调 Oracle 对每条 query 扫全表算真值
# ================================================================
# Oracle.query(q) → (card, dur_ms), 这里只取 card, dur 不存。
# group ∈ {'train', 'valid', 'test'}, 每个 group 单独遍历。
def generate_labels_for_queries(table: Table, queryset: Dict[str, List[Query]]) -> Dict[str, List[Label]]:
    oracle = Oracle(table)
    labels = {}
    for group, queries in queryset.items():
        l = []
        for i, q in enumerate(queries):
            card, _ = oracle.query(q)
            l.append(Label(cardinality=card, selectivity=card/table.row_num))
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} labels generated for {group}")
        labels[group] = l

    return labels

# ================================================================
# generate_labels: CLI entry, 精确版 (用 Oracle)
# ================================================================
# 一次性扫全表给整个 workload 算真值, 大表上比较慢 (DMV 11M × 2000 query 可达几小时)。
def generate_labels(dataset: str, version: str, workload: str) -> None:

    L.info("Load table...")
    table = load_table(dataset, version)

    L.info("Load queryset from disk...")
    queryset = load_queryset(dataset, workload)

    L.info("Start generate ground truth labels for the workload...")
    labels = generate_labels_for_queries(table, queryset)

    L.info("Dump labels to disk...")
    dump_labels(dataset, version, workload, labels)

# ================================================================
# update_labels_for_queries: 用 Sampling 估计快速生成近似 labels
# ================================================================
# 数据更新后 (e.g. data shift 实验追加了一批数据) 想快速看 label 怎么变,
# 不想真扫全表 → 在 sampling_ratio 比例的 sample 上算 cardinality 当近似真值。
# **注意**: 这是近似! 用作"快速 sanity check", paper 报数字时还得用 Oracle 精确版。
def update_labels_for_queries(table: Table, queryset: Dict[str, List[Query]], seed: int, sampling_ratio: float=0.05) -> Dict[str, List[Label]]:
    sample_ester = Sampling(table, sampling_ratio, seed)
    labels = {}
    for group, queries in queryset.items():
        l = []
        for i, q in enumerate(queries):
            card, _ = sample_ester.query(q)
            l.append(Label(cardinality=card, selectivity=card/table.row_num))
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} labels generated for {group}")
        labels[group] = l
    return labels

def update_labels(seed: int, dataset: str, version: str, workload: str, sampling_ratio: float=0.05) -> None:

    L.info("Load table...")
    table = load_table(dataset, version)

    L.info("Load queryset from disk...")
    queryset = load_queryset(dataset, workload)

    L.info("Updating ground truth labels for the workload, with sample size {}...".format(sampling_ratio))
    labels = update_labels_for_queries(table, queryset, seed, sampling_ratio)

    L.info("Dump labels to disk...")
    dump_labels(dataset, version, workload, labels)

