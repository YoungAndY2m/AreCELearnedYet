# ================================================================
# 教学注释 (annotation pass) — manipulate_dataset.py 总览
# ================================================================
# Data shift 实验工具: 在已有 dataset 上做 3 种"扰动"派生新版本, 然后 append
# 到原数据末尾, 给 update_naru / online-learning 实验做"环境变化"测试。
#
# 3 种扰动 (对应 paper "data update / drift" 实验)
# ----------------------------------------------------------------
#   ind  (Independence): 每列独立 shuffle → 破坏列间 correlation, 列边缘分布不变
#                        测 estimator 在 "失去相关性" 下的表现
#   cor  (Correlation):  每列 sort 后再整行 shuffle → 创造极强相关 (Spearman ≈ 1)
#                        测 estimator 在 "新增超强相关" 下的表现
#   skew (Skew):         挑全表最 rare 的 sample_ratio 行重复填充 → 数据分布
#                        高度偏向 rare value, 测 tail-skew 鲁棒性
#
# 流程
# ----------------------------------------------------------------
#   1. get_xxx_data() 派生新版本 → 存 {version}_{ind|cor|skew}.pkl
#   2. append_data() 把派生数据的前 batch_ratio 比例追加到 target 末尾
#      → 存 {version}+{派生}_{ratio}.pkl
#   3. update_naru 用这个新 version 增量微调 (= 模拟 data drift 后的 online update)
#
# 入口: `lecarb update --params "{'type': 'ind', 'batch_ratio': 0.2}"`
# ================================================================
import random
import logging
import pickle
import numpy as np
import math
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any, Tuple
from copy import deepcopy

from .dataset import load_table
from ..constants import DATA_ROOT, PKL_PROTO

L = logging.getLogger(__name__)

# ================================================================
# get_random_data: 每列独立 shuffle (= "independence" 扰动)
# ================================================================
# 列边缘分布保持不变 (= 每列出现频次不变), 但列之间任何相关性都被破坏。
# 例: 原数据 (gender, name) 有强相关 → shuffle 后变随机配对, "Bob, F" 会出现。
# Independence data: Random by each column
def get_random_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    rand_version = f"{version}_ind"
    random_file = DATA_ROOT / dataset / f"{rand_version}.pkl"
    if not overwrite and random_file.is_file():
        L.info(f"Dataset path exists, using it")
        return pd.read_pickle(random_file), rand_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, random_file, protocol=PKL_PROTO)
    return df, rand_version

# ================================================================
# get_sorted_data: 每列独立 sort 后整行 shuffle (= "max correlation" 扰动)
# ================================================================
# 先把每列单独 sort (= 第 i 行所有列都是各列的第 i 大), 然后整行 shuffle 打乱行顺序。
# 结果: 所有列的 rank 完全一致 → Spearman 相关系数 = 1 (最大相关)。
# 跟原数据的 actual correlation 比较, 测 estimator 在"出现新的极端相关"下的表现。
# Max Spearman correlation data: sort by each column
def get_sorted_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    sort_version = f"{version}_cor"
    sorted_file = DATA_ROOT / dataset / f"{sort_version}.pkl"
    if not overwrite and sorted_file.is_file():
        return pd.read_pickle(sorted_file), sort_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sort_values().reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, sorted_file, protocol=PKL_PROTO)
    return df, sort_version

# ================================================================
# get_skew_data: 挑最 rare 的行重复填充 (= "extreme skew" 扰动)
# ================================================================
# 算法:
#   1. 对每行算 "rank_sum" = 各列频次和 (= 该行有多 common; 低 = rare)
#   2. 挑 rank_sum 最小的 sample_ratio 比例行 (= 最 rare 的 ~0.05%)
#   3. 把这些 rare 行重复填充, 凑回原表大小
# 结果: 数据严重偏向 rare value, 测 estimator 在极端 skew 下的鲁棒性。
# Get skew data by tuple level frequent rank.
def get_skew_data(dataset: str = 'census', version: str = 'original', sample_ratio=0.0005, overwrite=False) -> Tuple[pd.DataFrame, str]:
    skew_version = f"{version}_skew"
    skew_file = DATA_ROOT / dataset / f"{skew_version}.pkl"
    if not overwrite and skew_file.is_file():
        return pd.read_pickle(skew_file), skew_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")


    rank_df = pd.DataFrame(0.0, index=range(len(df)), columns=['rank_sum']).astype(np.float32)
    for col in df.columns:
        rank_df['rank_sum'] += df[col].map(df[col].value_counts().div(len(rank_df))).astype(np.float32)
        print(f"{col} frequency calculation finished!")
    selected_id = rank_df.sort_values(by='rank_sum').head(round(len(df)*sample_ratio)).index
    sk_df = df.iloc[selected_id]
    sk_df = pd.concat([sk_df] * int(1/sample_ratio + 1), ignore_index=True).head(len(df))
    pd.to_pickle(sk_df, skew_file, protocol=PKL_PROTO)
    return sk_df, skew_version



# ================================================================
# append_data: 把派生数据的前 interval 比例追加到 target 末尾
# ================================================================
# 例: target=original (10000 行) + from=original_ind (10000 行) + interval=0.2
# → 输出 original+original_ind_0.2.pkl, 长度 12000 (原 + 2000 行扰动数据)。
# update_naru 用这个新 version 增量微调时, 看到的就是 "原数据基础上多了一批
# 偏离原分布的行" — 完美模拟 online data drift。
def append_data(dataset: str, version_target: str, version_from: str, interval=0.2):
    df_target = pd.read_pickle(DATA_ROOT / dataset / f"{version_target}.pkl")
    df_from = pd.read_pickle(DATA_ROOT / dataset / f"{version_from}.pkl")

    row_num = len(df_from)
    l = 0
    r = l + interval
    if r <= 1:
        L.info(f"Start appending {version_target} with {version_from} in [{l}, {r}]")
        df_target = df_target.append(df_from[int(l*row_num): int(r*row_num)], ignore_index=True, sort=False)
        pd.to_pickle(df_target, DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.pkl")
        df_target.to_csv(DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.csv", index=False)
        load_table(dataset, f"{version_target}+{version_from}_{r:.1f}")
    else:
        L.info(f"Appending Fail! Batch size is too big!")



# ================================================================
# gen_appended_dataset: CLI entry, 选扰动类型 + 派生 + append
# ================================================================
# `lecarb update --dataset DS --version V --params "{'type': 'ind', 'batch_ratio': 0.2}"`
# 三种 type (ind/cor/skew) 对应上面三种扰动方法。
def gen_appended_dataset(
    seed: int, dataset: str, version: str, 
    params: Dict[str, Any], overwrite: bool
    ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    update_type = params.get('type')
    batch_ratio = params.get('batch_ratio')
    L.info(f"Start generating appended data for {dataset}/{version}")

    if update_type == 'ind':
        _, rand_version = get_random_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, rand_version, interval=batch_ratio)
    elif update_type == 'cor':
        _, sort_version = get_sorted_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, sort_version, interval=batch_ratio)
    elif update_type == 'skew':
        _, skew_version = get_skew_data(dataset, version,
                                        sample_ratio=float(params['skew_size']), overwrite=overwrite)
        append_data(dataset, version, skew_version, interval=batch_ratio)
    else:
        raise NotImplementedError
    L.info("Finish updating data!")


