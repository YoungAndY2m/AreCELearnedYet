# ================================================================
# 教学注释 (annotation pass) — workload.py 总览
# ================================================================
# lecarb 的 query 抽象层 + 各种格式互转工具。所有 estimator 都通过 Query 对象
# 拿 query, 然后用 query_2_xxx 转成自己需要的形式 (triple / sql / vector / ...)。
#
# 核心数据类型
# ----------------------------------------------------------------
#   Query  (NamedTuple): predicates 字典 {col_name: (op, val) | None}, ncols
#                        e.g. Query({'age': ('>', 30), 'state': ('=', 'CA'), 'name': None}, ncols=3)
#   Label  (NamedTuple): (cardinality, selectivity), workload 的 ground truth
#
# 8 种 query 转换器 (每个 estimator 各取所需)
# ----------------------------------------------------------------
#   query_2_triple   → (cols, ops, vals) 三 list, 给 Naru / MHist / Sample / BayesNet 用
#   query_2_sql      → SQL string,        给 PG (postgres.py) / MySQL (mysql.py) 用
#   query_2_kde_sql  → KDE-PG SQL,        给 feedback_kde.py 用 (categorical 列要 discretize)
#   query_2_deepdb_sql → DeepDB 风格 SQL,  数据 normalize 后写进 SQL
#   query_2_sqls     → 每个 predicate 分一条 SQL (单列分析用)
#   query_2_vector   → [lo, hi] · ncols 的 float vector, 给 MSCN / lw_nn 这种 NN-based 用
#   query_2_quicksel_vector → QuickSel 专用 (vocab 离散化处理)
#
# 持久化
# ----------------------------------------------------------------
#   dump_queryset / load_queryset: workload (= train/valid/test 三 split 的 Query 列表)
#   dump_labels / load_labels    : 对应 ground truth
#   dump_sqls: 把 workload 导成 CSV (给 SQL Server / 外部工具用)
#
# 文件路径约定
# ----------------------------------------------------------------
#   queryset: DATA_ROOT/{dataset}/workload/{name}.pkl
#   labels:   DATA_ROOT/{dataset}/workload/{name}-{version}-label.pkl
#             (label 跟 version 绑定, 因为同 query 在不同 data version 上真值不同)
# ================================================================
import csv
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
import pickle
import numpy as np

from ..dtypes import is_categorical
from ..constants import DATA_ROOT, PKL_PROTO
from ..dataset.dataset import Table, load_table

# ================================================================
# Query: 不可变 query 对象
# ================================================================
# predicates: 用 OrderedDict 保持列顺序 (跟 table.columns 一致)。
# - 有 predicate 的列: (op, val), op ∈ {'=', '<', '<=', '>', '>=', '[]'}
# - 没 predicate 的列 (wildcard): None (= 该列接受任何值)
# ncols: 总列数 (= len(predicates))。
# 例: Query({'age': ('>', 30), 'state': None}, ncols=2)
class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int

class Label(NamedTuple):
    cardinality: int
    selectivity: float

# ================================================================
# new_query: 建一个所有列都是 None (wildcard) 的空 Query
# ================================================================
# 给 query generator 用 — 先建空 query, 然后选 ncols 列填 predicate。
def new_query(table: Table, ncols) -> Query:
    return Query(predicates=OrderedDict.fromkeys(table.data.columns, None),
                 ncols=ncols)

# ================================================================
# query_2_triple: 拆 Query 成 (cols, ops, vals) 三 list (= 最常用的格式)
# ================================================================
# 参数:
#   with_none=True : 没 predicate 的列也输出 (None 占位), 输出长度 = ncols
#                    (Naru / MHist 走这分支, 因为内部按 natural order 索引)
#   with_none=False: 只输出有 predicate 的列, 输出长度 = predicate 数量
#                    (Sampling / Oracle 走这分支, bitmap 累乘只关心命中列)
#   split_range=True: range query '[]' 拆成 '>=' AND '<=' 两个 predicate
#                    (有些 estimator 不支持 '[]' 算子, 用拆分版)
def query_2_triple(query: Query, with_none: bool=True, split_range: bool=False
               ) -> Tuple[List[int], List[str], List[Any]]:
    """return 3 lists with same length: cols(columns names), ops(predicate operators), vals(predicate literals)"""
    cols = []
    ops = []
    vals = []
    for c, p in query.predicates.items():
        if p is not None:
            if split_range is True and p[0] == '[]':
                cols.append(c)
                ops.append('>=')
                vals.append(p[1][0])
                cols.append(c)
                ops.append('<=')
                vals.append(p[1][1])
            else:
                cols.append(c)
                ops.append(p[0])
                vals.append(p[1])
        elif with_none:
            cols.append(c)
            ops.append(None)
            vals.append(None)
    return cols, ops, vals

# ================================================================
# query_2_sql: 转 SQL string, 给 postgres.py / mysql.py 用
# ================================================================
# 参数:
#   aggregate=True : 用 COUNT(*) 包 (= 真跑出 cardinality)
#   aggregate=False: 用 SELECT * (= 只看优化器估计, 不真扫表 — PG EXPLAIN 用这个)
#   split=False    : '[]' range query 用 BETWEEN (PG / MySQL 都支持)
#   split=True     : 拆成 >= AND <= 两个 predicate
#   dbms='postgres': 表名用 "..." 引号 (PG 标准); dbms='mysql' 用 `...` 反引号
# Categorical (string) 列的 val 自动加单引号 (= SQL 字符串字面量)。
def query_2_sql(query: Query, table: Table, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)
        if op == '[]':
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            preds.append(f"{col} {op} {val}")

    if dbms == 'mysql':
        return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM `{table.name}` WHERE {' AND '.join(preds)}"
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

# ================================================================
# query_2_kde_sql: KDE-PG 专用 SQL (categorical 列必须 discretize)
# ================================================================
# 跟 query_2_sql 区别:
#   1. categorical 列的 val 不加引号, 而是 discretize 成整数 bin_id
#      (KDE 在 C 内核里只处理数值, 不识别字符串字面量)
#   2. assert op='=' 不允许 range, 不允许 tuple val (KDE 对 categorical 列只支持 =)
# 必须配 KDE-patched PG (KDE_DATABASE_URL 那个) 才能跑。
def query_2_kde_sql(query: Query, table: Table):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            assert op =='=' and not isinstance(val, tuple), val
            val = table.columns[col].discretize(val).item()
        if op == '[]':
            preds.append(f"{col} >= {val[0]}")
            preds.append(f"{col} <= {val[1]}")
        else:
            preds.append(f"{col} {op} {val}")

    return f"SELECT * FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

# ================================================================
# query_2_deepdb_sql: DeepDB 专用 (val 全部 normalize 到 [0, 1])
# ================================================================
# DeepDB 内部数据已经 normalize, query 也得 normalize 才能对齐。
# 例: col.age min=0, max=100, query age > 50 → 50 normalized = 0.5 → SQL: `age > 0.5`。
def query_2_deepdb_sql(query: Query, table: Table, aggregate=True, split=False):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if op == '[]':
            val = table.columns[col].normalize(list(val))
            assert len(val) == 2, val
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            val = table.columns[col].normalize(val).item()
            preds.append(f"{col} {op} {val}")

    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

# ================================================================
# query_2_sqls: 每个 predicate 单独一条 SQL (= predicate-by-predicate)
# ================================================================
# 给"列独立分析"用 (e.g. 算单列 selectivity), 主流程不用。
def query_2_sqls(query: Query, table: Table):
    sqls = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)

        if op == '[]':
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} between {val[0]} and {val[1]}")
        else:
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} {op} {val}")
    return sqls


# ================================================================
# query_2_vector: 把 Query 编码成定长 float vector, 给 NN-based estimator 用
# ================================================================
# 每列贡献 [lo, hi] 两个数 (∈ [0, upper]), 总长度 = 2 · ncols。
# 编码规则 (val 都先 normalize 到 [0, 1]):
#   None (wildcard): [0, 1]  (= 全 domain)
#   '=' val        : [val, val]
#   '<=' val       : [0, val]
#   '>=' val       : [val, 1]
#   '[]' (lo, hi)  : [lo, hi]
# upper 参数: 通常 1, 但 MSCN 等 model 有时用 100 (放大避免太多 0/1 极值)。
# 这个 vector 直接喂 MSCN / lw_nn 的 query encoder, 拿到 cardinality 估计。
def query_2_vector(query: Query, table: Table, upper: int=1):
    vec = []
    for col, pred in query.predicates.items():
        if pred is None:
            vec.extend([0.0, 1.0])
            continue
        op, val = pred
        if op == '[]':
            vec.extend([table.columns[col].normalize(val[0]).item(), table.columns[col].normalize(val[1]).item()])
        elif op == '>=':
            vec.extend([table.columns[col].normalize(val).item(), 1.0])
        elif op == '<=':
            vec.extend([0.0, table.columns[col].normalize(val).item()])
        elif op == '=':
            vec.extend([table.columns[col].normalize(val).item()] * 2)
        else:
            raise NotImplementedError
    return np.array(vec) * upper

# ================================================================
# query_2_quicksel_vector: QuickSel 专用 vector (离散列特殊处理)
# ================================================================
# QuickSel paper (Park 2020) 是基于"selectivity functional" 的 CE 方法。
# 这函数比 query_2_vector 复杂, 因为 QuickSel 要求 discrete 列的 predicate
# bounds 必须对齐到 vocab 中实际存在的 value (而不是连续区间), 用 vocab 数组
# + argmax 找最近的合法边界。
# 连续列直接 normalize, 跟 query_2_vector 同。
def query_2_quicksel_vector(query: Query, table: Table, discrete_cols=set()):
    vec = []
    for col_name, pred in query.predicates.items():
        if pred is None:
            vec.extend([0.0, 1.0])
            continue
        op, val = pred
        col = table.columns[col_name]

        # adjust predicate to a proper range for discrete columns
        if col_name in discrete_cols:
            if is_categorical(col.dtype):
                val = col.discretize(val)
                minval = 0
                maxval = col.vocab_size
                vocab = np.arange(col.vocab_size)
            else: # integer values
                minval = col.minval
                maxval = col.maxval + 1
                vocab = col.vocab

            if op == '=':
                val = (val, val)
            elif op == '>=':
                val = (val, maxval)
            elif op == '<=':
                val = (minval, val)
            else:
                assert op == '[]'

            vocab = np.append(vocab, maxval)
            # argmax return 0 if no value in array satisfies
            val0 = vocab[np.argmax(vocab >= val[0])] if val[0] < maxval else maxval
            val1 = vocab[np.argmax(vocab > val[1])] if val[1] < maxval else maxval
            assert val0 <= val1, (val0, val1)
            assert val0 >= minval and val0 <= maxval, (val0, minval, maxval)
            assert val1 >= minval and val1 <= maxval, (val1, minval, maxval)
            # normalize to [0, 1]
            vec.extend([(val0-minval)/(maxval-minval), (val1-minval)/(maxval-minval)])

        # directly normalize continous columns
        else:
            if op == '>=':
                vec.extend([col.normalize(val).item(), 1.0])
            elif op == '<=':
                vec.extend([0.0, col.normalize(val).item()])
            elif op == '[]':
                vec.extend([col.normalize(val[0]).item(), col.normalize(val[1]).item()])
            else:
                raise NotImplementedError
    return np.array(vec)


# ================================================================
# dump_queryset / load_queryset: workload (= query 集) 持久化
# ================================================================
# queryset 结构: {'train': [Query, ...], 'valid': [Query, ...], 'test': [Query, ...]}
# 文件路径: DATA_ROOT/{dataset}/workload/{name}.pkl
# 一个 workload 可被多个 estimator 用 (label 跟具体 dataset version 绑定)。
def dump_queryset(dataset: str, name: str, queryset: Dict[str, List[Query]]) -> None:
    query_path = DATA_ROOT / dataset / "workload"
    query_path.mkdir(exist_ok=True)
    with open(query_path / f"{name}.pkl", 'wb') as f:
        pickle.dump(queryset, f, protocol=PKL_PROTO)

def load_queryset(dataset: str, name: str) -> Dict[str, List[Query]]:
    query_path = DATA_ROOT / dataset / "workload"
    with open(query_path / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)

# ================================================================
# dump_labels / load_labels: ground truth (cardinality + selectivity) 持久化
# ================================================================
# 文件路径: DATA_ROOT/{dataset}/workload/{name}-{version}-label.pkl
# 跟 queryset 分开存因为同一组 query 在不同 data version 上真值不同
# (例如 data shift 实验, 同一个 'age > 50' query 在 v1 / v2 上 cardinality 不一样)。
# 真值由 gen_label.py 在数据上扫一遍算出。
def dump_labels(dataset: str, version: str, name: str, labels: Dict[str, List[Label]]) -> None:
    label_path = DATA_ROOT / dataset / "workload"
    with open(label_path / f"{name}-{version}-label.pkl", 'wb') as f:
        pickle.dump(labels, f, protocol=PKL_PROTO)

def load_labels(dataset: str, version: str, name: str) -> Dict[str, List[Label]]:
    label_path = DATA_ROOT / dataset / "workload"
    with open(label_path / f"{name}-{version}-label.pkl", 'rb') as f:
        return pickle.load(f)

# ================================================================
# dump_sqls: 把 workload 导成 CSV (SQL + 真值), 给外部 estimator 用
# ================================================================
# 例: 想用 SQL Server / 自己写的 estimator 跑同一组 query, 导出后让它读 CSV。
# 输出 './test.csv', 每行 (sql_string, true_card)。
def dump_sqls(dataset: str, version: str, workload: str, group: str='test'):
    table = load_table(dataset, version)
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)

    with open('test.csv', 'w') as f:
        writer = csv.writer(f)
        for query, label in zip(queryset[group], labels[group]):
            sql = query_2_sql(query, table, aggregate=False, dbms='sqlserver')
            writer.writerow([sql, label.cardinality])
