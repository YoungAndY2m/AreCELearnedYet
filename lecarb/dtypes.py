"""
Module for auxiliary type detection functions
"""
# ================================================================
# 教学注释 (annotation pass) — dtypes.py 总览
# ================================================================
# 给 Column / 各 estimator 判断列类型的辅助工具。三个谓词:
#   is_categorical(dtype) → bool : 字符串 / object / pandas Categorical
#   is_numerical(dtype)   → bool : int / float / datetime
#   is_discrete(dtype)    → bool : categorical 或 integer (≠ float)
#
# 使用场景:
#   - dataset.py Column.__init__: 决定是否做 vocab encoding (categorical 必须 encode)
#   - mhist.py / bayesnet.py: 判断列是否需要 discretize
#   - workload.py 的 predicate 验证: numerical 才允许 range query
#
# 实现关键: 用 issubclass / isinstance 检查 numpy + pandas 两套类型层次。
# np.dtype 是元数据, .type 拿真正的 Python type, 才能 issubclass 检查。
# ================================================================

from typing import Any

import numpy as np
import pandas as pd

# === categorical 类型清单 ===
# np.bool: 真假 (= 2 个 distinct value, 算 categorical 因为没有 "more/less" 语义)
# np.object: 字符串 / Python object (pandas 默认 str 列就是 object)
# pd.CategoricalDtype: pandas 主动 .astype('category') 标记的列 (省内存)
# pd.PeriodDtype: 时间段类型 (例 '2020Q1'), categorical 因为不连续
CATEGORICAL_NUMPY_DTYPES = [np.bool, np.object]
CATEGORICAL_PANDAS_DTYPES = [pd.CategoricalDtype, pd.PeriodDtype]
CATEGORICAL_DTYPES = CATEGORICAL_NUMPY_DTYPES + CATEGORICAL_PANDAS_DTYPES

# === numerical 类型清单 ===
# np.number: int / float / complex 总称 (issubclass 检查能匹配所有数值)
# np.datetime64: 日期 (连续, 支持 < / > / range)
# pd.DatetimeTZDtype: 带时区的 datetime
NUMERICAL_NUMPY_DTYPES = [np.number, np.datetime64]
NUMERICAL_PANDAS_DTYPES = [pd.DatetimeTZDtype]
NUMERICAL_DTYPES = NUMERICAL_NUMPY_DTYPES + NUMERICAL_PANDAS_DTYPES


# ================================================================
# is_categorical: 该 dtype 是否 categorical
# ================================================================
# 优先 fall-through: numerical 优先 (= 数值类型不归 categorical, 即使是整数)。
# numpy dtype 取 .type 后用 issubclass; pandas dtype 直接 isinstance。
def is_categorical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a categorical type
    """

    if is_numerical(dtype):
        return False

    if isinstance(dtype, np.dtype):
        dtype = dtype.type

        return any(issubclass(dtype, c) for c in CATEGORICAL_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in CATEGORICAL_PANDAS_DTYPES)


# ================================================================
# is_numerical: 该 dtype 是否数值 (int/float/datetime)
# ================================================================
def is_numerical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a numerical type
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
        return any(issubclass(dtype, c) for c in NUMERICAL_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in NUMERICAL_PANDAS_DTYPES)

# ================================================================
# is_discrete: categorical 或整数 (= 不连续可枚举的列)
# ================================================================
# 跟 is_categorical 的区别: 整数列 (int) 算 discrete 但不算 categorical。
# 用途: estimator 决定是否能直接当 vocab 处理 (例 mhist Partition.distinct 算法
# 假设 discrete 才有意义)。
def is_discrete(dtype: Any) -> bool:
    """
    Given a type, return if that type is a discrete type (categorical or integer)
    """
    if is_categorical(dtype):
        return True

    assert isinstance(dtype, np.dtype), dtype
    dtype = dtype.type
    return issubclass(dtype, np.integer)

