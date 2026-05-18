# ================================================================
# 教学注释 (annotation pass) — gen_workload.py 总览
# ================================================================
# CLI entry, 生成新 workload (= 一组 Query) + 可选 labels。
# 由 `lecarb workload --dataset DS --version V --name W --params P` 触发。
#
# 生成流程
# ----------------------------------------------------------------
# 1. 解析 params 拿到 3 类启发式函数:
#    - attr (attribute selection): 怎么选哪些列做 predicate
#    - center (center selection):  predicate value 中心点怎么选
#    - width (width selection):    range query 区间宽度怎么定
#    具体函数在 generator.py 里 (asf_xxx / csf_xxx / wsf_xxx 命名)
# 2. 构造 QueryGenerator (持有 table + 3 类函数)
# 3. 对每个 group (train/valid/test) 生成指定数量的 query
# 4. dump queryset + 可选 dump labels
#
# Focused workload (data shift 用)
# ----------------------------------------------------------------
# 如果传了 old_version + win_ratio, 只在 "新数据的最后 win_ratio 部分"
# 上生成 query —— 模拟 "用户只关心最近数据" 的场景。
# ================================================================
import random
import logging
import numpy as np
from typing import Dict, Any
import copy

from . import generator
from .generator import QueryGenerator
from .gen_label import generate_labels_for_queries
from .workload import dump_queryset, dump_labels
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

# ================================================================
# get_focused_table: 截取 table 的尾部 win_ratio 比例
# ================================================================
# 给"只在新数据上生成 query"用 (= focused workload)。
# 例 win_ratio=0.2 + ref_table 10000 行 → focused_table 取 table 最后 2000 行。
# 之后 parse_columns() 在子集上重算 vocab (= 新数据可能有原 table 没出现的值)。
def get_focused_table(table, ref_table, win_ratio):
    focused_table = copy.deepcopy(table)
    win_size = int(win_ratio * len(ref_table.data))
    focused_table.data = focused_table.data.tail(win_size).reset_index(drop=True)
    focused_table.parse_columns()
    return focused_table

# ================================================================
# generate_workload: 主入口, 由 CLI 调
# ================================================================
# params 结构 (json):
#   {'number': {'train': 10000, 'valid': 100, 'test': 2000},
#    'attr':   {'naru': 0.5, 'mscn': 0.5},    # asf_xxx 函数名 → 概率权重
#    'center': {'sample_data': 0.5, 'random': 0.5},
#    'width':  {'sample_data': 1.0}}
def generate_workload(
    seed: int, dataset: str, version: str,
    name: str, no_label: bool, old_version: str, win_ratio: str,
    params: Dict[str, Dict[str, Any]]
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    # 把 params['attr'] / ['center'] / ['width'] 里的字符串 key 映射成 generator.py
    # 里的实际函数对象 (getattr 用 'asf_' / 'csf_' / 'wsf_' 前缀, 见 generator.py)。
    # value 是该函数的采样权重 (多个函数加权混合用)。
    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    L.info("Load table...")
    table = load_table(dataset, version)
    if old_version and win_ratio:
        L.info(f"According to {old_version}, generate queries for updated data in {version}...")
        win_ratio = float(win_ratio)
        assert 0<win_ratio<=1
        old_table = load_table(dataset, old_version)
        query_table = get_focused_table(table, old_table, win_ratio)
        qgen = QueryGenerator(
                table=query_table,
                attr=attr_funcs,
                center=center_funcs,
                width=width_funcs,
                attr_params=params.get('attr_params') or {},
                center_params=params.get('center_params') or {},
                width_params=params.get('width_params') or {})
    else:
        qgen = QueryGenerator(
            table=table,
            attr=attr_funcs,
            center=center_funcs,
            width=width_funcs,
            attr_params=params.get('attr_params') or {},
            center_params=params.get('center_params') or {},
            width_params=params.get('width_params') or {})

    queryset = {}
    for group, num in params['number'].items():
        L.info(f"Start generate workload with {num} queries for {group}...")
        queries = []
        for i in range(num):
            queries.append(qgen.generate())
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} queries generated")
        queryset[group] = queries

    L.info("Dump queryset to disk...")
    dump_queryset(dataset, name, queryset)

    if no_label:
        L.info("Finish without generating corresponding ground truth labels")
        return

    L.info("Start generate ground truth labels for the workload...")
    labels = generate_labels_for_queries(table, queryset)

    L.info("Dump labels to disk...")
    dump_labels(dataset, version, name, labels)
