# ================================================================
# 教学注释 (annotation pass) — generator.py 总览
# ================================================================
# Query 生成的具体启发式函数库。每条 query 由三步构造:
#   1. attr selection (asf_*)  : 选哪些列做 predicate
#   2. center selection (csf_*): predicate value 的中心点 (= "锚点")
#   3. width selection (wsf_*) : range 区间的宽度 + 决定 op (=/[]/>=/<=)
# QueryGenerator 把三步组合, 每步可以混合多个函数 (按权重采样)。
#
# 命名约定 (gen_workload.py 用 getattr 查):
#   asf_xxx → attribute selection function
#   csf_xxx → center selection function
#   wsf_xxx → width selection function
#
# 关键启发式 (paper 提到的几种)
# ----------------------------------------------------------------
#   asf_naru     : Naru paper 的方式 (5-12 列随机选)
#   asf_pred_number: 指定列数范围, 支持 whitelist/blacklist
#   csf_distribution: 按数据分布采 center (= 保证 non-empty query, 跟 L0 SampleTupleThenRandom 同)
#   csf_domain  : 从 (sorted attrs) 的真实组合里采 (= 保证 carda > 0)
#   csf_*_ood   : "out of distribution" 版本, 各列独立采 (= 可能 carda = 0)
#   wsf_uniform : 区间宽度 ~ uniform(0, max-min)
#   wsf_exponential: 区间宽度 ~ exponential (偏向小宽度)
#   wsf_naru   : 用 = / <= / >= 三种 op 随机, 小 domain 强制 =  (跟 Naru L0 同)
#   wsf_equal  : 全部 = (point query)
# ================================================================
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from typing_extensions import Protocol

import numpy as np
import pandas as pd

from ..dtypes import is_categorical
from ..dataset.dataset import Table, Column
from .workload import Query, new_query

L = logging.getLogger(__name__)

"""====== Attribute Selection Functions ======"""

# Protocol: structural typing (= duck typing 的类型化版本)。
# 表明任何参数签名匹配的函数都能当 AttributeSelFunc 用。
class AttributeSelFunc(Protocol):
    def __call__(self, table: Table, params: Dict[str, Any]) -> List[str]: ...

# ============================================================
# asf_pred_number: 指定 predicate 列数, 支持 white/blacklist
# ============================================================
# params: {'whitelist': [...], 'blacklist': [...], 'nums': [1, 2, 3]}
# 例: nums=[3,5] → 50% 概率出 3 列 / 50% 5 列 (np.random.choice 等概率)。
def asf_pred_number(table: Table, params: Dict[str, Any]) -> List[str]:
    if 'whitelist' in params:
        attr_domain = params['whitelist']
    else:
        blacklist = params.get('blacklist') or []
        attr_domain = [c for c in list(table.data.columns) if c not in blacklist]
    nums = params.get('nums')
    nums = nums or range(1, len(attr_domain)+1)
    num_pred = np.random.choice(nums)
    assert num_pred <= len(attr_domain)
    return np.random.choice(attr_domain, size=num_pred, replace=False)

def asf_comb(table: Table, params: Dict[str, Any]) -> List[str]:
    assert 'comb' in params and type(params['comb']) == list, params
    for c in params['comb']:
        assert c in table.columns, c
    return params['comb']

# ============================================================
# asf_naru: Naru paper Section 6.2 的方式 — 随机 5-12 列
# ============================================================
# 跟 [L0 eval_model.py:GenerateQuery](../../../AllModels/Naru/eval_model.py) 一致。
def asf_naru(table: Table, params: Dict[str, Any]) -> List[str]:
    num_filters = np.random.randint(5, 12)
    return np.random.choice(table.data.columns, size=num_filters, replace=False)

"""====== Center Selection Functions ======"""

class CenterSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]: ...

# ============================================================
# csf_domain: 从 (attrs) 列的真实 distinct 组合里采 center → 保证 carda > 0
# ============================================================
# 算法: 对选中的 attrs, 取 table.data 的 distinct 行组合, 随机抽一行的值当 center。
# DOMAIN_CACHE 缓存每组 attrs 的 distinct row indices, 避免重复 drop_duplicates。
# 适用场景: 想生成"non-empty query" 工作负载, label 全 > 0 方便算 q-error。
DOMAIN_CACHE = {}
# This domain version makes sure that query's cardinality > 0
def csf_domain(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    global DOMAIN_CACHE
    key = tuple(sorted(attrs))
    if key not in DOMAIN_CACHE:
        data_from = params.get('data_from') or 0
        DOMAIN_CACHE[key] = table.data[data_from:][attrs].drop_duplicates().index
        assert len(DOMAIN_CACHE[key]) > 0, key
    #  L.debug(f'Cache size: {len(DOMAIN_CACHE)}')
    row_id = np.random.choice(DOMAIN_CACHE[key])
    return [table.data.at[row_id, a] for a in attrs]

# ============================================================
# csf_distribution: 按真实数据分布采 (从 table.data 随机抽一行的值)
# ============================================================
# 跟 csf_domain 不同: 不去重, 频次高的值被采到的概率高 (= 跟数据分布对齐)。
# Batch 缓存机制: 每 1000 次 query 一次性抽 1000 个 row_ids 缓存, 减少 np.random.choice 调用开销
# (对每个 query 调用一次太慢, 实测 batch 化能加速 5-10x)。
ROW_CACHE = None
GLOBAL_COUNTER = 1000
def csf_distribution(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    global GLOBAL_COUNTER
    global ROW_CACHE
    if GLOBAL_COUNTER >= 1000:
        data_from = params.get('data_from') or 0
        ROW_CACHE = np.random.choice(range(data_from, len(table.data)), size=1000)
        GLOBAL_COUNTER = 0
    row_id = ROW_CACHE[GLOBAL_COUNTER]
    GLOBAL_COUNTER += 1
    #  data_from = params.get('data_from') or 0
    #  row_id = np.random.choice(range(data_from, len(table.data)))
    return [table.data.at[row_id, a] for a in attrs]

# ============================================================
# csf_ood / csf_vocab_ood / csf_domain_ood: "out of distribution" 变种
# ============================================================
# 各列 center 独立采 (= 列间不 jointly 真实), 可能造出真实数据里不存在的组合。
# 用作 OOD 测试集 — 检查 estimator 对训练时没见过的 query 怎么样。
# - csf_ood       : 每列从一个独立 row 抽 (= 还是 in-data 值, 但跨行混搭)
# - csf_vocab_ood : 每列从 vocab 均匀采 (= 完全脱离 row 结构)
# - csf_domain_ood: numerical 列从 [min, max] uniform 采 (= 可能不是 vocab 中的值)
def csf_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

def csf_vocab_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        centers.append(np.random.choice(col.vocab))
    return centers

def csf_domain_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        if is_categorical(col.dtype): # randomly pick one point from domain for categorical
            centers.append(np.random.choice(col.vocab))
        else: # uniformly pick one point from domain for numerical
            centers.append(random.uniform(col.minval, col.maxval))
    return centers

# ============================================================
# csf_naru: Naru paper 的方式 — 一行所有列的真实值
# ============================================================
# 跟 csf_distribution 同思路, 但不 batch cache (直接 randint, 慢 5x 但代码简单)。
# 跟 [L0 SampleTupleThenRandom](../../../AllModels/Naru/eval_model.py) 一致。
def csf_naru(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_id = np.random.randint(0, len(table.data))
    return [table.data.at[row_id, a] for a in attrs]

def csf_naru_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

"""====== Width Selection Functions ======"""

class WidthSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query: ...

# ============================================================
# parse_range: 把 (left, right) 区间转 (op, val) 元组
# ============================================================
# 三种情况:
#   left ≤ minval        → ('<=', right)   (左端到边界, 退化成 <=)
#   right ≥ maxval       → ('>=', left)    (右端到边界, 退化成 >=)
#   都在中间             → ('[]', (left, right))  (真正的 range query)
# 注释掉的分支 (left == right) 故意没启用 — 让 '=' 走 wsf_equal, 这里只处理 range。
def parse_range(col: Column, left: Any, right: Any) -> Optional[Tuple[str, Any]]:
    #  if left <= col.minval and right >= col.maxval:
    #      return None
    #  if left == right:
    #      return ('=', left)
    if left <= col.minval:
        return ('<=', right)
    if right >= col.maxval:
        return ('>=', left)
    return ('[]', (left, right))

# ============================================================
# wsf_uniform: range width ~ uniform(0, col.maxval - col.minval)
# ============================================================
# 整个 domain 范围内均匀采宽度, 给 query 选择性谱写一个均匀分布的 workload。
# Categorical 列 / NaN center → 强制 '=' (range 在 categorical 上没意义)。
def wsf_uniform(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        width = random.uniform(0, col.maxval-col.minval)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query

# ============================================================
# wsf_exponential: range width ~ exponential(lambda = 10 / (maxval - minval))
# ============================================================
# 指数分布偏向小宽度 (= 大部分 query 选择性高, 少数选择性低) — 更贴近真实
# workload 形状 (大多数 query 是 narrow range)。
def wsf_exponential(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        lmd = 1 / ((col.maxval - col.minval) / 10)
        width = random.expovariate(lmd)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query

# ============================================================
# wsf_naru: 跟 Naru paper 一样, 每列随机选 op ('>=' / '<=' / '='); domain <10 强制 '='
# ============================================================
# 注意 wsf_naru 不构造 range 区间 — 它产生的是 point + half-bounded query (没 '[]'),
# 跟 wsf_uniform / wsf_exponential 在 range query 类型上不同。
# 见 [L0 SampleTupleThenRandom](../../../AllModels/Naru/eval_model.py).
def wsf_naru(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    ops = np.random.choice(['>=', '<=', '='], size=len(attrs))
    for a, c, o in zip(attrs, centers, ops):
        if table.columns[a].vocab_size >= 10:
            query.predicates[a] = (o, c)
        else:
            query.predicates[a] = ('=', c)
    return query

def wsf_equal(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        query.predicates[a] = ('=', c)
    return query

# ================================================================
# QueryGenerator: 把 asf / csf / wsf 三步组合, 加权采样输出 Query
# ================================================================
# 三个 dict 字段都是 {function: weight}: 每次 generate() 按权重抽一个函数用。
# 同一组配置混合多种启发式生成 query, 让 workload 更多样 (paper 实验通常
# 50% naru / 50% mscn 这种组合)。
class QueryGenerator(object):
    table: Table
    attr: Dict[AttributeSelFunc, float]
    center: Dict[CenterSelFunc, float]
    width: Dict[WidthSelFunc, float]
    attr_params: Dict[str, Any]
    center_params: Dict[str, Any]
    width_params: Dict[str, Any]

    def __init__(
            self, table: Table,
            attr: Dict[AttributeSelFunc, float],
            center: Dict[CenterSelFunc, float],
            width: Dict[WidthSelFunc, float],
            attr_params: Dict[str, Any],
            center_params: Dict[str, Any],
            width_params: Dict[str, Any]
            ) -> None:
        self.table = table
        self.attr = attr
        self.center = center
        self.width = width
        self.attr_params = attr_params
        self.center_params = center_params
        self.width_params = width_params

    # ============================================================
    # generate: 生成单条 Query (三步组合 + 加权采样)
    # ============================================================
    def generate(self) -> Query:
        # np.random.choice(keys, p=probs): 按 probs 加权采样选一个 key (function)。
        attr_func = np.random.choice(list(self.attr.keys()), p=list(self.attr.values()))
        #  L.info(f'start generate attr {attr_func.__name__}')
        attr_lst = attr_func(self.table, self.attr_params)

        center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
        #  L.info(f'start generate center points {center_func.__name__}')
        center_lst = center_func(self.table, attr_lst, self.center_params)

        width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
        #  L.info(f'start generate widths {width_func.__name__}')
        return width_func(self.table, attr_lst, center_lst, self.width_params)
