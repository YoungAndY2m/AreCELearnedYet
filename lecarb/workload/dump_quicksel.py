# ================================================================
# 教学注释 (annotation pass) — dump_quicksel.py 总览
# ================================================================
# QuickSel estimator (Park 2020 SIGMOD) 是 Java 实现的外部工具, 不在 lecarb
# Python 内跑。这个文件把 lecarb 的 workload + table 导出成 QuickSel 期望的
# CSV 格式, 让 Java 端能读。
#
# 两类导出
# ----------------------------------------------------------------
#   dump_quicksel_query_files: 把 train/test query 导成 CSV
#     每行: [col_0_lo, col_0_hi, col_1_lo, col_1_hi, ..., selectivity]
#     用 query_2_quicksel_vector 转 (discrete 列要对齐到 vocab 实际值的边界)
#
#   generate_quicksel_permanent_assertions: 生成"永久断言" (permanent assertion)
#     QuickSel 算法需要单列等深范围 query 当 prior. 对每列等分成 N 个区间,
#     用 Oracle 算真值, 写进 CSV。Java 端读这些当 QuickSel 模型的 anchor points.
#
# 输出路径: DATA_ROOT/{dataset}/quicksel/*.csv
# ================================================================
import csv
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .workload import load_queryset, load_labels, query_2_quicksel_vector, new_query
from ..dtypes import is_discrete, is_categorical
from ..dataset.dataset import load_table
from ..estimator.estimator import Oracle
from ..constants import DATA_ROOT

L = logging.getLogger(__name__)

# ================================================================
# dump_quicksel_query_files: 导出 query → CSV (Java 端读)
# ================================================================
# 每行格式: [normalized lo/hi 对] · ncols + [selectivity]
# Hard code: power dataset 的 3 列 Sub_metering_* 虽然 dtype 是 float,
# 实际语义是整数 (= electricity submeter readings), 强制当 discrete 处理。
def dump_quicksel_query_files(dataset: str, version: str, workload: str, overwrite: bool) -> None:
    result_path = DATA_ROOT / dataset / "quicksel"
    result_path.mkdir(exist_ok=True)
    if not overwrite and Path(result_path / f"{workload}-{version}-train.csv").is_file() and Path(result_path / f"{workload}-{version}-test.csv").is_file():
        L.info("Already has quicksel workload file dumped, do not continue")
        return

    table = load_table(dataset, version)
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)

    discrete_cols = set()
    for col_name, col in table.columns.items():
        # hard code for power dataset since all these columns are actually integers
        if dataset[:5] == 'power' and col_name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            discrete_cols.add(col_name)
            continue
        if is_discrete(col.dtype):
            discrete_cols.add(col_name)
    L.info(f"Detect discrete columns: {discrete_cols}")

    for group in ('train', 'test'):
        L.info(f"Start dump {workload} for {dataset}-{version}")
        result_file = result_path / f"{workload}-{version}-{group}.csv"
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            for query, label in zip(queryset[group], labels[group]):
                vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                vec.append(label.selectivity)
                writer.writerow(vec)
        L.info(f"File dumped to {result_file}")

# ================================================================
# generate_quicksel_permanent_assertions: 生成 QuickSel "永久断言"
# ================================================================
# QuickSel paper 算法需要一组 "permanent assertions" 当 prior 模型 anchor:
# 单列 range query + 真实 selectivity, 等分成 count 个区间。
# 这些 anchor 帮 QuickSel 在没有 workload 训练数据时也能有合理初始化。
#
# 处理三种列类型:
#   categorical: 如果 vocab 小, 每个值单独一个 '=' query;
#                vocab 大用 [lo, hi] range, 边界对齐到 vocab 实际值
#   integer    : 等分 [minval, maxval+1] 成 count 段, 每段一个 '[]' query
#   real-value : 等分 [minval, maxval], 每段一个 '[]' query (不对齐 vocab)
#
# 输出每行: [col_0_lo, col_0_hi, ..., col_n_lo, col_n_hi, selectivity]
# 只有当前列的 lo/hi 是真值, 其它列填 [0, 1] (= wildcard)。
def generate_quicksel_permanent_assertions(dataset: str, version: str, params: Dict[str, Dict[str, Any]], overwrite: bool) -> None:
    result_path = DATA_ROOT / dataset / "quicksel"
    result_path.mkdir(exist_ok=True)
    result_file = result_path / f"{version}-permanent.csv"
    if not overwrite and result_file.is_file():
        L.info("Already has permanent assertions generated, do not continue")
        return

    count = params['count']+1

    table = load_table(dataset, version)
    oracle = Oracle(table)

    discrete_cols = set()
    for col_name, col in table.columns.items():
        # hard code for power dataset since all these columns are actually integers
        if dataset[:5] == 'power' and col_name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            discrete_cols.add(col_name)
            continue
        if is_discrete(col.dtype):
            discrete_cols.add(col_name)
    L.info(f"Detect discrete columns: {discrete_cols}")

    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([0.0, 1.0] * table.col_num + [1.0])
        for col_id, col in enumerate(table.columns.values()):
            L.info(f"Start generate permanent queries on column {col.name}")
            # hard code for power dataset since all these columns are actually integers
            if is_discrete(col.dtype) or (dataset[:5] == 'power' and col.name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
                if is_categorical(col.dtype):
                    L.info("Categorical column")
                    if col.vocab_size <= count:
                        for i in range(col.vocab_size):
                            query = new_query(table, ncols=1)
                            query.predicates[col.name] = ('=', col.vocab[i])
                            card, _ = oracle.query(query)
                            #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                            #  vec.append(card/table.row_num)
                            vec = [0.0, 1.0] * table.col_num
                            vec.append(card/table.row_num)
                            vec[col_id*2] = i/col.vocab_size
                            vec[col_id*2+1] = (i+1)/col.vocab_size
                            writer.writerow(vec)
                            L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
                    else:
                        minval = 0
                        maxval = col.vocab_size
                        norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                        prange = minval + (maxval - minval) * norm_range
                        for i in range(len(prange)-1):
                            val0 = col.vocab[np.ceil(prange[i]).astype(int)]
                            val1 = col.vocab[np.ceil(prange[i+1]).astype(int)-1]
                            assert np.greater_equal(np.array(val1).astype(object), val0), (val1, val0)
                            query = new_query(table, ncols=1)
                            query.predicates[col.name] = ('[]', (val0, val1))
                            card, _ = oracle.query(query)
                            #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                            #  vec.append(card/table.row_num)

                            vec = [0.0, 1.0] * table.col_num
                            vec.append(card/table.row_num)
                            vec[col_id*2] = norm_range[i]
                            vec[col_id*2+1] = norm_range[i+1]
                            writer.writerow(vec)
                            L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
                else:
                    L.info("Integer column")
                    minval = col.minval
                    maxval = col.maxval + 1
                    norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                    prange = minval + (maxval - minval) * norm_range
                    for i in range(len(prange)-1):
                        val0 = np.ceil(prange[i])
                        val1 = np.ceil(prange[i+1])-1
                        assert val1 >= val0, (val0, val1)
                        query = new_query(table, ncols=1)
                        query.predicates[col.name] = ('[]', (val0, val1))
                        card, _ = oracle.query(query)
                        #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                        #  vec.append(card/table.row_num)

                        vec = [0.0, 1.0] * table.col_num
                        vec.append(card/table.row_num)
                        vec[col_id*2] = norm_range[i]
                        vec[col_id*2+1] = norm_range[i+1]
                        writer.writerow(vec)
                        L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
            else:
                L.info("Real-value column")
                norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                prange = col.minval + (col.maxval - col.minval) * norm_range
                for i in range(len(prange)-1):
                    query = new_query(table, ncols=1)
                    query.predicates[col.name] = ('[]', (prange[i], prange[i+1]))
                    card, _ = oracle.query(query)
                    #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                    #  vec.append(card/table.row_num)
                    vec = [0.0, 1.0] * table.col_num
                    vec.append(card/table.row_num)
                    vec[col_id*2] = norm_range[i]
                    vec[col_id*2+1] = norm_range[i+1]
                    writer.writerow(vec)
                    L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
