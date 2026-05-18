# ================================================================
# 教学注释 (annotation pass) — gen_dataset.py 总览
# ================================================================
# 合成 2D 数据集生成器, 给 CE 实验做"可控相关性 + 可控偏态"的玩具数据。
# 由 `lecarb gen-dataset --params "{'row_num': N, 'dom': D, 'corr': c, 'skew': s}"` 触发。
#
# 合成数据的两个轴 (= paper 报告 q-error 时常扫的两个维度)
# ----------------------------------------------------------------
#   corr (相关性): 第二列以概率 `corr` 复制第一列、以 (1-corr) 概率独立采样
#                  - corr=0.0 → 完全独立 (Heuristic / PG 都准)
#                  - corr=1.0 → 完全相关 (独立假设方法严重高估)
#   skew (偏态): 用 Pareto 分布生成第一列, c = skew - 1
#                  - skew=1.0 → 均匀 (etc)
#                  - skew→∞ → 极度偏态 (少数 hot value 占大比例)
# 限制: col_num 必须 = 2 (paper 工具脚本, 没扩展到任意维)。
#
# 输出
# ----------------------------------------------------------------
#   DATA_ROOT/{dataset}/{version}.csv  (raw)
#   DATA_ROOT/{dataset}/{version}.pkl  (DataFrame, load_table 用)
#   DATA_ROOT/{dataset}/{version}.table.pkl (Column 解析后的 cache)
# ================================================================
import random
import logging
import numpy as np
import pandas as pd
# scipy.stats:
#   - truncnorm: 截断正态 (定义在 [low, upp] 上, 落界外的概率重新归一化)
#   - truncexpon: 截断指数
#   - genpareto: 广义 Pareto (重尾分布, 给 skew 控制偏态用)
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any

from .dataset import load_table
from ..constants import DATA_ROOT

L = logging.getLogger(__name__)

# ================================================================
# get_truncated_normal / get_truncated_expon: 工具函数 (实际未在 generate_dataset 里调用)
# ================================================================
# 留作 future-use 接口, e.g. 想换分布族时改这里。
def get_truncated_normal(mean=0, sd=100, low=0, upp=1000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_truncated_expon(scale=100, low=0, upp=1000):
    return truncexpon(b=(upp-low)/scale, loc=low, scale=scale)

# ================================================================
# generate_dataset: 主入口, 生成一份合成 2D 数据集
# ================================================================
# 流程:
#   1. 用 genpareto 按 skew 参数生成第一列 (skew-1 是 Pareto shape 参数 c)
#   2. 第二列以 corr 概率复制第一列, 否则独立 uniform 采样
#   3. 输出 CSV + Pickle + 用 load_table 触发 vocab 解析
def generate_dataset(
    seed: int, dataset: str, version: str,
    params: Dict[str, Any], overwrite: bool
) -> None:
    path = DATA_ROOT / dataset
    path.mkdir(exist_ok=True)
    csv_path = path / f"{version}.csv"
    pkl_path = path / f"{version}.pkl"
    if not overwrite and csv_path.is_file():
        L.info(f"Dataset path exists, do not continue")
        return

    row_num = params['row_num']
    col_num = params['col_num']
    dom = params['dom']
    corr = params['corr']
    skew = params['skew']

    if col_num != 2:
        L.info("For now only support col=2!")
        exit(0)

    L.info(f"Start generate dataset with {col_num} columns and {row_num} rows using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # generate the first column according to skew
    # 第一列构造:
    # 1. 先撒一遍 [0, dom) 全部 domain 值, 保证每个值至少出现 1 次 (vocab 完整)
    # 2. 剩余 row_num - dom 行用 genpareto 采 → 缩放到 [0, dom) → 取整 + clip
    # 这样保证 col0 既覆盖整个 domain, 又按 skew 分布有偏态。
    col0 = np.arange(dom) # make sure every domain value has at least 1 value
    tmp = genpareto.rvs(skew-1, size=row_num-len(col0)) # c = skew - 1, so we can have c >= 0
    tmp = ((tmp - tmp.min()) / (tmp.max() - tmp.min())) * dom # rescale generated data to the range of domain
    col0 = np.concatenate((col0, np.clip(tmp.astype(int), 0, dom-1)))

    # generate the second column according to the first
    # 第二列构造: 按 corr 概率从 col0 抄 (= 完全相关), 否则独立采 → 控制相关性强度。
    # 例 corr=0.7: 70% 的行 col1 = col0, 30% 的行 col1 独立 uniform。
    col1 = []
    for c0 in col0:
        col1.append(c0 if np.random.uniform(0, 1) <= corr else np.random.choice(dom))

    df = pd.DataFrame(data={'col0': col0, 'col1': col1})

    L.info(f"Dump dataset {dataset} as version {version} to disk")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    load_table(dataset, version)
    L.info(f"Finish!")

