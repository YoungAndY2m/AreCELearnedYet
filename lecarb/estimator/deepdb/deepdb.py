# ============================================================================
# deepdb.py (L2 wrapper) — lecarb 适配层: 把 L0 DeepDB 接进 ARELY benchmark
# ============================================================================
# (教学注释 by Claude, 不动原代码)
#
# 这是 ARELY/lecarb 框架集成 DeepDB 的 *adapter / wrapper*. 角色对比:
#   - L0 ([AllModels/DeepDB/](../../../../AllModels/DeepDB/)): 完整研究系统, ~10,781 行 / 55 files
#   - L1 (ARELY standalone): **∅ 不存在** (LOG_STRUCTURE.md §10.1)
#   - L2 (本文件 + 大部分 byte-identical vendored DeepDB 代码): ~8,468 行 / 36 files
#
# L0 → L2 改造 (LOG_STRUCTURE.md §10.2, 7 类改动):
#   1. 删 `rspn/` 整树, 全部 inline 到 `aqp_spn/aqp_spn.py` (从双继承变单继承)
#   2. 删 `schemas/`, 改用 `construct_schema(table)` 动态构造 SchemaGraph
#   3. 删 `aqp_spn/code_generation/` (C++ codegen 不需要)
#   4. `maqp.py` 4-stage CLI → 单文件 `deepdb.py` 4 函数 (本文件)
#   5. SQL 字符串边界: lecarb Query → `query_2_sql` → `parse_query` → SPN
#   6. 加 `get_deepdb_size(spn_ensemble)` (本文件) 算模型大小给 sizelimit 用
#   7. 状态保存: `spn_ensemble.state` 加 train_time / args / metrics / version
#
# lecarb 调用接口 (跟其它 estimator 完全一致):
#   - `train_deepdb(seed, dataset, version, workload, params, sizelimit)`
#   - `test_deepdb(dataset, version, workload, params, overwrite)`
#   - `load_deepdb(dataset, model_name)`
#   - `update_deepdb(seed, dataset, new_version, workload, params, overwrite)`
#     ← 增量学习入口 (paper §7, 调 AQPSPN.add_dataset)
# ============================================================================
import time
import logging
from typing import Dict, Any, Tuple
import numpy as np
from ..estimator import Estimator
from ..utils import run_test, evaluate
from ...constants import DATA_ROOT, MODEL_ROOT, NUM_THREADS, VALID_NUM_DATA_DRIVEN
from ...dataset.dataset import load_table
# query_2_sql: lecarb Query → SQL str (paper §5.1 字符串边界协议)
from ...workload.workload import load_queryset, load_labels, query_2_sql

import pathlib
import sys
# sys.path 注入: 让 vendored deepdb 内部的 `from aqp_spn.aqp_spn import ...`
# 这种绝对 import 能找到本目录下的 aqp_spn/ etc. 路径
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from ensemble_compilation.graph_representation import SchemaGraph, Table, QueryType
from ensemble_compilation.spn_ensemble import SPNEnsemble, read_ensemble
from data_preparation.join_data_preparation import JoinDataPreparator
from data_preparation.prepare_single_tables import prepare_all_tables
from aqp_spn.aqp_spn import AQPSPN
from aqp_spn.aqp_leaves import Sum, Categorical, IdentityNumericLeaf
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.structure.Base import get_nodes_by_type, Product
from evaluation.utils import parse_query

L = logging.getLogger(__name__)

class Args:
    """Hyperparameter container. 只 4 个 SPN-relevant 参数, 比 L0 maqp.py 大幅简化."""
    def __init__(self, **kwargs):
        self.max_rows_per_hdf_file = 20000000
        self.hdf_sample_size = 1000000
        self.rdc_threshold = 0.3
        self.ratio_min_instance_slice = 0.01

        # overwrite parameters from user
        self.__dict__.update(kwargs)

def construct_schema(table):
    """从 lecarb Table 动态构造 SchemaGraph (10 行). 替代 L0 hardcoded
    schemas/<ds>/schema.py 工厂 (LOG_STRUCTURE.md §10.2.2).

    L2 单表场景: 只 add 1 table, 无 relationship (无 join 路径).
    """
    # construct a schema that has one table only
    csv_file = DATA_ROOT / table.dataset / f"{table.version}.csv"
    schema = SchemaGraph()
    schema.add_table(Table(f'"{table.name}"', # use table name in postgres since deepdb deal with sql directly
                           attributes=table.data.columns.values.tolist(),
                           csv_file_location=csv_file,
                           table_size=table.row_num))
    return schema

def get_deepdb_size(spn_ensemble):
    """遍历 SPN tree 节点估 MB. L0 没这个概念; L2 给 sizelimit 检查用 (LOG_STRUCTURE.md §10.2.6).

    每节点统计:
      - Product: children 数 + scope size
      - Sum: 2*children + scope + cluster_centers (children × scope)
      - Categorical: 2 + p 长度 (scope + cardinality + p)
      - IdentityNumericLeaf: 3 + unique_vals + prob_sum
    最后 × 4 bytes/int 转 MB.
    """
    # only deal with single table, only have one spn
    spn = spn_ensemble.spns[0].mspn
    size = 0
    nodes = get_nodes_by_type(spn, Product)
    for node in nodes:
        size += len(node.children) + len(node.scope)

    nodes = get_nodes_by_type(spn, Sum)
    for node in nodes:
        assert len(node.children) == len(node.weights) == len(node.cluster_centers)
        assert len(node.cluster_centers[0]) == len(node.scope)
        num_child = len(node.children)
        num_var = len(node.scope)
        size += 2*num_child + num_var + num_var*num_child # children, weights, scope, cluster_centers

    nodes = get_nodes_by_type(spn, Categorical)
    for node in nodes:
        assert len(node.scope) == 1
        size += 2 + len(node.p) # scope, cardinality, p

    nodes = get_nodes_by_type(spn, IdentityNumericLeaf)
    for node in nodes:
        assert len(node.scope) == 1
        assert len(node.unique_vals) + 1 == len(node.prob_sum)
        size += 3 + len(node.unique_vals) + len(node.prob_sum) # scope, cardinality, null_value_prob, uniqe_vals, prob_sum

    # assume use 4 bytes to store all integers and floats
    return size * 4 / 1024 / 1024 #MB

def train_deepdb(seed, dataset, version, workload, params, sizelimit):
    """L2 训练入口 (替代 L0 maqp.py 的 4-stage CLI).

    流程:
      1. construct_schema(table) (10 行替代 L0 schemas/<ds>/schema.py)
      2. (如 hdf 不存在) prepare_all_tables 把 csv 转 HDF5
      3. JoinDataPreparator(single_table=table_name) 单表 sampling (跳过 join)
      4. AQPSPN(...).learn(...) → 单 SPN
      5. spn_ensemble.add_spn(spn); 写 .pt (含 spn_ensemble.state 元信息)
    """
    L.info(f"params: {params}")
    args = Args(**params)

    # for sampling
    np.random.seed(seed)

    table = load_table(dataset, version)
    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    schema = construct_schema(table)

    # convert data from csv to hdf
    hdf_path = DATA_ROOT / dataset / 'deepdb' / f"hdf-{version}"
    if hdf_path.is_dir():
        L.info('Use existing hdf file!')
    else:
        hdf_path.mkdir(parents=True)
        prepare_all_tables(schema, str(hdf_path), csv_seperator=',', max_table_data=args.max_rows_per_hdf_file)

    # generate SPN for table
    prep = JoinDataPreparator(hdf_path / 'meta_data.pkl', schema, max_table_data=args.max_rows_per_hdf_file)
    spn_ensemble = SPNEnsemble(schema)
    table_obj = schema.tables[0]
    L.info(f"table name: {table_obj.table_name}")
    df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(args.hdf_sample_size,
                                                                                 single_table=table_obj.table_name,
                                                                                 post_sampling_factor=1.0)
    assert len(df_samples) == min(args.hdf_sample_size, table.row_num), '{} != min({}, {})'.format(len(df_samples), args.hdf_sample_size, table.row_num)


    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-spn_sample{len(df_samples)}_rdc{args.rdc_threshold}_ms{args.ratio_min_instance_slice}-{seed}.pkl"

    # learn spn
    L.info(f"Start learning SPN for {table_obj.table_name}.")
    start_stmp = time.time()
    aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema, relationship_list=None,
                     full_sample_size=len(df_samples), table_set={table_obj.table_name},
                     column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
    min_instance_slice = args.ratio_min_instance_slice * len(df_samples)
    aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=False,
                  rdc_threshold=args.rdc_threshold)
    spn_ensemble.add_spn(aqp_spn)
    dur_min = (time.time() - start_stmp) / 60

    mb = get_deepdb_size(spn_ensemble)
    L.info(f"SPN built finished, time spent since start: {dur_min:.1f} mins with {mb:.2f}MB size of memory")
    L.info(f'Final SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    if sizelimit > 0 and mb > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mb:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue!")
        return

    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = DeepDB(spn_ensemble, table, schema, 'valid')
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    spn_ensemble.state = {
        'train_time': dur_min,
        'model_size': mb,
        'args': args,
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': table.dataset,
        'version': table.version,
        'valid_error': {workload: metrics}
    }

    # save spn to file
    spn_ensemble.save(model_file)
    L.info(f'Training finished! Save model to {model_file} Time spent since start: {dur_min:.2f} mins')

class DeepDB(Estimator):
    """lecarb Estimator 实现 — 跟其它 estimator (MSCN / Naru / CoLSE) 同接口.

    持有: spn_ensemble (训好的 SPNEnsemble), schema (动态构造的 SchemaGraph).
    .query(q) → (est_card, dur_ms).
    """
    def __init__(self, spn_ensemble, table, schema, model_name):
        super(DeepDB, self).__init__(table=table, model=model_name)
        self.spn_ensemble = spn_ensemble
        self.schema = schema

    def query(self, query):
        """lecarb 标准 query 入口. 4 步:
          1. query_2_sql: lecarb Query AST → SQL string
          2. parse_query: SQL string → deepdb Query AST (LOG_STRUCTURE.md §10.2.5)
          3. spn_ensemble.cardinality: paper §5.1 主推理
          4. round + dur_ms 返回
        """
        sql = query_2_sql(query, self.table, aggregate=True, split=True)
        #  print(sql)
        query = parse_query(sql.strip(), self.schema)
        assert query.query_type == QueryType.CARDINALITY

        start_stmp = time.time()
        formula, factors, card, factor_values = self.spn_ensemble.cardinality(query, return_factor_values=True)
        dur_ms = (time.time() - start_stmp) * 1e3
        #  print(factors)
        #  print(factor_values)
        #  print(formula)
        return np.round(card), dur_ms

def load_deepdb(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    """加载 .pkl SPN ensemble + 重建 DeepDB Estimator.

    `build_reverse_dict=True`: read_ensemble 加载 SPN 后多建一个 reverse lookup dict
    (查找加速). 详 [spn_ensemble.py](spn_ensemble.py).
    """
    model_file = MODEL_ROOT / dataset /f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    spn_ensemble = read_ensemble(model_file, build_reverse_dict=True)
    L.info(f'Get SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    state = spn_ensemble.state
    table = load_table(state['dataset'], state['version'])
    schema = construct_schema(table)
    estimator = DeepDB(spn_ensemble, table, schema, model_name)
    return estimator, state

def test_deepdb(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
    """

    model_file = MODEL_ROOT / dataset /f"{params['model']}.pkl"
    L.info(f"load model from {model_file} ...")
    spn_ensemble = read_ensemble(model_file, build_reverse_dict=True)
    L.info(f'Get SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    state = spn_ensemble.state
    table = load_table(state['dataset'], state['version'])
    schema = construct_schema(table)
    estimator = DeepDB(spn_ensemble, table, schema, params['model'])

    run_test(dataset, version, workload, estimator, overwrite)

def update_deepdb(seed: int, dataset: str, new_version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    # for sampling
    np.random.seed(seed)
    # load old model
    new_table = load_table(dataset, new_version)
    model_path = MODEL_ROOT / new_table.dataset
    model_file = model_path /f"{params['model']}.pkl"
    L.info(f"load model from {model_file} ...")
    estimator, state = load_deepdb(dataset, params['model'])
    spn_ensemble = estimator.spn_ensemble

    old_version = state['version']
    args = state['args']
    old_table = load_table(dataset, old_version)
    # load updated data and save to csv
    updated_dataset = load_table(dataset, new_version)
    updated_dataset.data = updated_dataset.data.iloc[len(old_table.data):].sample(frac=0.01)
    updated_dataset.data.reset_index(drop=True)
    updated_dataset.version += '_cut'
    updated_dataset.name += '_cut'
    updated_dataset.data.to_csv(DATA_ROOT / dataset / f"{updated_dataset.version}.csv", index=False)
    updated_dataset.row_num = len(updated_dataset.data)
    
    L.info(f"Updated size {updated_dataset.row_num}")

    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, new_version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    schema = construct_schema(updated_dataset)
    L.info(f"{schema}")
    # convert data from csv to hdf
    hdf_path = DATA_ROOT / dataset / 'deepdb' / f"hdf-{updated_dataset.version}"
    if hdf_path.is_dir():
        L.info('Use existing hdf file!')
    else:
        hdf_path.mkdir(parents=True)
    prepare_all_tables(schema, str(hdf_path), csv_seperator=',', max_table_data=args.max_rows_per_hdf_file)

    # generate SPN for table
    prep = JoinDataPreparator(hdf_path / 'meta_data.pkl', schema, max_table_data=args.max_rows_per_hdf_file)
    table_obj = schema.tables[0]
    L.info(f"table name: {table_obj.table_name}")
    L.info(f"table attributes: {schema.tables[0].attributes}")
    df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(args.hdf_sample_size,
                                                                                 single_table=table_obj.table_name,
                                                                                 post_sampling_factor=1.0)
    # assert len(df_samples) == min(args.hdf_sample_size, old_table.row_num), '{} != min({}, {})'.format(len(df_samples), args.hdf_sample_size, old_table.row_num)

    
    # Update model
    L.info(f"Start learning SPN for {table_obj.table_name}.")
    start_stmp = time.time()
    spn_ensemble.spns[0].learn_incremental(df_samples.to_numpy())
    dur_min = (time.time() - start_stmp) / 60

    L.info(f"SPN update finished, time spent since start: {dur_min:.4f} mins")
    L.info(f'Final SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    # L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    # estimator = DeepDB(spn_ensemble, new_table, schema, 'valid')
    # preds = []
    # for q in valid_queries:
    #     est_card, _ = estimator.query(q)
    #     preds.append(est_card)
    # _, metrics = evaluate(preds, [l.cardinality for l in labels])

    spn_ensemble.state['update_time'] = dur_min
    args = state['args']
    # save spn to file
    sample_size = min(args.hdf_sample_size, old_table.row_num)
    new_model_file = model_path / f"{new_table.version}-spn_sample{sample_size}_rdc{args.rdc_threshold}_ms{args.ratio_min_instance_slice}-{seed}.pkl"

    spn_ensemble.save(new_model_file)
    L.info(f'Updating finished! Save model to {new_model_file} Time spent since start: {dur_min:.4f} mins')


