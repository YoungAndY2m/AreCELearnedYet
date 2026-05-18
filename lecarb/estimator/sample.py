
# ================================================================
# 教学注释 (annotation pass) — sample.py 总览
# ================================================================
# Lecarb 实现的 "uniform table sampling" baseline。跟 Naru L0
# [estimators.py:Sampling](../../../AllModels/Naru/estimators.py#L833) 思路一致, 但用
# lecarb 接口重写, 干净很多 (L0 ~30 行算法; 这里加 entry point 共 44 行)。
#
# 算法 (跟 L0 完全一样)
# ----------------------------------------------------------------
# 1. 离线: 从全表 uniform 随机抽 ratio · N 行存内存
# 2. 推理: query 在 sample 上算 bitmap, 数命中行数 s
# 3. 估计: Card ≈ (N / |sample|) · s  ("scale up" 回真实表大小)
#
# 优势 / 劣势
# ----------------------------------------------------------------
# 优势: 实现极简, 对独立性零假设 (整行采, 自然保留列间 correlation)
# 劣势: 高选择性 (rare) query 命中数少, 方差爆炸
#       e.g. ratio=0.01, sel=1e-5 → 期望命中 = 0.01 · N · 1e-5 行, 小表上常 = 0
# Naru paper Section 6 把它当 "经典 baseline" 对照, Naru 在 tail q-error 上完胜。
#
# 跟 L0 的具体差异
# ----------------------------------------------------------------
#   - 支持 random_state=seed → 可复现采样 (L0 有 TODO 没做)
#   - 接 lecarb Estimator 基类, .query() 返 (card, dur_ms)
#   - 用 query_2_triple 拆 query, 不再手写 column 提取循环
#   - 加 test_sample() entry point, 由 `lecarb test --estimator sample` 触发
#   - 支持 params['version']: 可以 "从 A 版本数据采样, 在 B 版本上测" (data shift 实验用)
# ================================================================

import time
import logging
from typing import Any, Dict
import numpy as np
from .estimator import Estimator, OPS
from .utils import run_test
from ..workload.workload import query_2_triple
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

# ================================================================
# Sampling: 表采样 estimator (= L0 Sampling 同算法, lecarb 接口)
# ================================================================
class Sampling(Estimator):
    def __init__(self, table, ratio, seed):
        # super 把 (version, ratio, seed) 作为元数据存入 self.params,
        # 用于 __repr__ 输出 + 结果 CSV 标识。
        super(Sampling, self).__init__(table=table, version=table.version, ratio=ratio, seed=seed)
        # pandas DataFrame.sample(frac=ratio, random_state=seed):
        # 不放回随机抽 ratio · N 行; seed 让结果可复现。
        # 注意 self.sample 是 DataFrame, 不是 numpy → query 时按列名 self.sample[c] 取列。
        self.sample = table.data.sample(frac=ratio, random_state=seed)
        self.sample_num = len(self.sample)

    def query(self, query):
        # query_2_triple(with_none=False): 只返回有 predicate 的列三元组, 没 predicate 的列直接跳过 (跟 L0 Sampling 一致, 它本来就只关心命中的列)。
        # split_range=False: 范围查询保留 '[]' 算子, 不拆成 '>=' AND '<='。
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        # bitmap 累乘 AND: 每条 sample 行是否满足所有 predicate。
        # 初值全 True (无 predicate 时整 sample 都算命中, sel=1)。
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            # OPS[o]: '=' / '<' 等 → numpy 比较算子; 对该列整列广播。
            # &=: 跟之前的 bitmap 做 element-wise AND。
            bitmap &= OPS[o](self.sample[c], v)
        # Scale up: card 估计 = (N / |sample|) · 命中数。
        # 例: ratio=0.01 → 缩放系数 = 100; sample 命中 5 行 → 估计 500 行。
        card = np.round((self.table.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

# ================================================================
# test_sample: lecarb CLI 调用入口
# ================================================================
# `lecarb test --estimator sample --dataset DS --workload W --params "{'ratio': 0.01}" --seed S`
# 会调到这里。
def test_sample(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the sample draw from, might not be the same with the one we test on
        ratio: the ratio of the sample
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    # 用户可以指定 sample 从 A 版本采、测试在 B 版本上跑 (data shift 实验)。
    # 没指定就用同版本。
    table = load_table(dataset, params.get('version') or version)

    L.info("construct sampling estimator...")
    # ratio 默认 0.01 (1% 采样, 跟 Naru paper 设置一致)。
    estimator = Sampling(table, ratio=params['ratio'] or 0.01, seed=seed)
    L.info(f"built sampling estimator: {estimator}")

    # run_test: lecarb 通用测试 loop (载入 test query → 调 estimator.query → 算 q-error → 写 CSV)。
    run_test(dataset, version, workload, estimator, overwrite)


