# ================================================================
# 教学注释 (annotation pass) — estimator.py 总览
# ================================================================
# lecarb 所有 estimator 的基类 + 共用算子表 + Oracle 真值 estimator。
# 这是 lecarb 框架的核心契约: 任何新 estimator 继承 `Estimator` 实现 `.query(query)`,
# 就能被 `lecarb test` 统一驱动 / 写结果 CSV / 算 q-error。
#
# 跟 Naru L0 [estimators.py:CardEst](../../../AllModels/Naru/estimators.py#L29) 的对应
# ----------------------------------------------------------------
#   L0 CardEst.Query(cols, ops, vals) → int
#   L2 Estimator.query(query)         → (card, dur_ms)
#   L0 CardEst 自己维护 query_starts/dur_ms/errs 数组
#   L2 Estimator 不维护, 由 run_test 在外面统一记录 (= 关注点分离更干净)
#
# 文件结构
# ----------------------------------------------------------------
#   - Estimator (base class)  : __init__ 存 table + params 字典, __repr__ 拼名字, query() 抽象
#   - OPS / in_between        : 共用比较算子表 (字符串 → numpy 函数)
#   - Oracle (Estimator)      : 扫全表算真值 (= L0 Oracle), 给 gen_label.py 用
# ================================================================
import time
import logging
import numpy as np
from typing import Tuple, Any
from ..workload.workload import Query, query_2_triple
from ..dataset.dataset import Table

L = logging.getLogger(__name__)

# ================================================================
# Estimator: 抽象基类
# ================================================================
# 子类必须实现 `.query(query) -> (card, dur_ms)`。
# __init__ 用 **kwargs 把超参存进 self.params, __repr__ 自动拼成
# "naru-version=v1;psample=2000" 这种字符串, 写进结果 CSV 给后续分析用。
class Estimator(object):
    """Base class for a cardinality estimator."""
    def __init__(self, table: Table, **kwargs: Any) -> None:
        self.table = table
        self.params = dict(kwargs)

    def __repr__(self) -> str:
        # 例: Sampling(table, ratio=0.01, seed=123) → "sampling-version=v1;ratio=0.01;seed=123"
        pstr = ';'.join([f"{p}={v}" for p, v in self.params.items()])
        return f"{self.__class__.__name__.lower()}-{pstr}"

    def query(self, query: Query) -> Tuple[float, float]:
        """return est_card, dur_ms"""
        raise NotImplementedError

# ================================================================
# in_between: 范围比较算子, 给 OPS['[]'] 用
# ================================================================
# numpy 没有内置 between, 自己写一个: lo ≤ data ≤ hi.
# 用 element-wise & (= bitwise AND on bool arrays) 而不是 Python `and`,
# 因为 `and` 不能 broadcast。
def in_between(data: Any, val: Tuple[Any, Any]) -> bool:
    assert len(val) == 2
    lrange, rrange = val
    return np.greater_equal(data, lrange) & np.less_equal(data, rrange)

# ================================================================
# OPS: 字符串算子 → numpy 比较函数的映射 (= lecarb 全局共用算子表)
# ================================================================
# 所有 estimator 都用 OPS[op_string](data, val) 调比较。
# np.greater / np.less 等都支持 broadcast: data 是 array, val 是 scalar
# → 返回 bool array。
# 注: L0 Naru 在 estimators.py 顶部有自己的 OPS, L2 提取到这里共用。
OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '[]': in_between
}

# ================================================================
# Oracle: 真值 estimator (= L0 Naru Oracle)
# ================================================================
# 直接扫全表算精确 cardinality, 给 gen_label.py 生成 ground truth 用。
# 单次 query 时间 O(N · ncols), 大表上慢, 所以一般离线生成 label 时调一次,
# 之后从 label.pkl 加载结果。
class Oracle(Estimator):
    def __init__(self, table):
        super(Oracle, self).__init__(table=table)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.table.row_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.table.data[c], v)
        card = bitmap.sum()
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

#  from pandasql import sqldf <- too slow
    #  def query(self, query):
    #      sql = query_2_sql(query, self.table)
    #      data = self.table.data
    #      start_stmp = time.time()
    #      df = sqldf(sql, locals())
    #      card = df.iloc[0, 0]
    #      dur_ms = (time.time() - start_stmp) * 1e3
    #      return card, dur_ms
