# ================================================================
# 教学注释 (annotation pass) — bayesnet.py 总览
# ================================================================
# lecarb 的 Bayesian Network estimator (pomegranate-based)。
# 算法部分 = L0 [estimators.py:BayesianNetwork](../../../AllModels/Naru/estimators.py#L929) 的搬运 (文件第 19 行
# 明写 "copied from naru with slight modification")。
# **完整算法解释看 L0 那边的教学注释** (BN 是什么 / pomegranate API /
# 跟 MADE 对比 / discretize 必要性 / VariableElimination)。
#
# 文件结构 (= L0 算法 + lecarb 包装层)
# ----------------------------------------------------------------
#   1. BayesianNetworkWorker  : L0 BayesianNetwork 类的搬运 (改 table API)
#                              — 跑 progressive sampling 的真正算法
#   2. Result (NamedTuple)    : 异步 query 结果容器 (i, est_card, dur_ms)
#   3. Bayes(Estimator)       : lecarb Estimator 接口 wrapper
#                              用 Ray 分布式并行调度 N 个 Worker 跑 query
#   4. test_bayesnet          : lecarb CLI entry point
#
# 为什么要分两层 (Worker vs Estimator wrapper)?
# ----------------------------------------------------------------
# BN inference 慢 (单 query 几百 ms 到几秒, 跟 MADE 完全不在一个量级):
#   - pgmpy.VariableElimination 是精确推理, 复杂度跟 graph treewidth 相关
#   - pomegranate.predict_proba 是近似但仍然慢
# 单进程跑 2000 query 要小时级。所以 L2 用 Ray 启 parallelism 个 worker
# 进程, 各自持有一个 BN 副本, 并发跑不同 query。L0 也用了类似 Ray 设计
# (eval_model.py:RunNParallel) 但代码风格不一样。
#
# 跟 L0 的具体差异 (BayesianNetworkWorker)
# ----------------------------------------------------------------
#   - L0 接收 NaruTable / dataset 对象; L2 直接收 lecarb table
#   - L0 调 dataset.tuples.numpy(); L2 调 col.discretize(table.data[cname])
#     拼成 [N, ncols] 整数矩阵 (跟 NaruTableDataset 一致)
#   - L0 用 print, L2 用 L.info / L.debug
#   - .Query() 方法签名: L0 Query(columns, operators, vals); L2 Query(query)
#     内部 query_2_triple 拆解
#   - L2 n_jobs=NUM_THREADS (而不是 L0 硬编码 8)
# ================================================================
import time
import copy
import json
import pickle
import logging
import collections
from typing import Any, Dict, NamedTuple
import numpy as np
import pandas as pd
from .estimator import Estimator, OPS
from .utils import run_test
from ..workload.workload import query_2_triple
from ..dataset.dataset import load_table
from ..constants import NUM_THREADS
from ..dtypes import is_categorical

L = logging.getLogger(__name__)

"""The below implementation is copied from https://github.com/naru-project/naru with some slight modification"""

# ================================================================
# BayesianNetworkWorker: L0 BayesianNetwork 类的 lecarb 适配版
# ================================================================
# 算法 100% = L0 (见 L0 BayesianNetwork 类的详注), 仅 table API 改:
#   - col.vocab / col.vocab_size 替代 L0 col.all_distinct_values / col.DistributionSize()
#   - col.discretize(series) 替代 L0 Discretize(col)
#   - table.row_num 替代 L0 table.cardinality
# 跑 progressive sampling 用 BN 推理替代 MADE forward (= 同样的 ∏Zᵢ 估计 sel(R))。
class BayesianNetworkWorker(object):
    """Progressive sampling with a pomegranate bayes net."""

    # ============================================================
    # build_discrete_mapping: 给高基数列再分粗 (= L0 同名方法)
    # ============================================================
    # 理由: BN 的 CPT 大小 = ∏|D_i|, 高基数列爆炸。
    # 两种分桶方法 (equal_size / equal_freq) 见 L0 教学注释详解。
    def build_discrete_mapping(self, table, discretize, discretize_method):
        assert discretize_method in ["equal_size",
                                     "equal_freq"], discretize_method
        self.max_val = collections.defaultdict(lambda: None)
        if not discretize:
            return {}
        table = table.copy()
        mapping = {}
        for col_id in range(len(table[0])):
            col = table[:, col_id]
            if max(col) > discretize:
                if discretize_method == "equal_size":
                    denom = (max(col) + 1) / discretize
                    fn = lambda v: np.floor(v / denom)
                elif discretize_method == "equal_freq":
                    per_bin = len(col) // discretize
                    counts = collections.defaultdict(int)
                    for x in col:
                        counts[int(x)] += 1
                    assignments = {}
                    i = 0
                    bin_size = 0
                    for k, count in sorted(counts.items()):
                        if bin_size > 0 and bin_size + count >= per_bin:
                            bin_size = 0
                            i += 1
                        assignments[k] = i
                        self.max_val[col_id] = i
                        bin_size += count
                    assignments = np.array(
                        [assignments[i] for i in range(int(max(col) + 1))])

                    def capture(assignments):

                        def fn(v):
                            return assignments[v.astype(np.int32)]

                        return fn

                    fn = capture(assignments)
                else:
                    assert False

                mapping[col_id] = fn
        return mapping

    def apply_discrete_mapping(self, table, discrete_mapping):
        table = table.copy()
        for col_id in range(len(table[0])):
            if col_id in discrete_mapping:
                fn = discrete_mapping[col_id]
                table[:, col_id] = fn(table[:, col_id])
        return table

    def apply_discrete_mapping_to_value(self, value, col_id, discrete_mapping):
        if col_id not in discrete_mapping:
            return value
        return discrete_mapping[col_id](value)

    # ============================================================
    # __init__: 载入数据 → discretize → 学 BN 结构 → 填 CPT (+可选 pgmpy)
    # ============================================================
    # 完整参数语义 (algorithm / max_parents / topological_sampling_order /
    # use_pgm / discretize) 见 L0 BayesianNetwork.__init__ 的详注。
    # L2 改动: table 直接是 lecarb Table 对象 (L0 是 NaruTableDataset)。
    def __init__(self,
                 #  dataset,
                 table,
                 num_samples,
                 algorithm="greedy",
                 max_parents=-1,
                 topological_sampling_order=True,
                 use_pgm=True,
                 discretize=None,
                 discretize_method="equal_size",
                 root=None):

        # pomegranate 用时再 import (重依赖, 没装 BN 也能 import 别的 estimator)。
        # 注意名字冲突: pomegranate.BayesianNetwork 是真正的 BN 类,
        # BayesianNetworkWorker 是本文件外层 wrapper, 互不影响。
        from pomegranate import BayesianNetwork
        self.discretize = discretize
        self.discretize_method = discretize_method
        # 深拷贝 lecarb Table, 避免训练过程被外部修改影响。
        self.table = copy.deepcopy(table)
        # 把整张表 discretize 成 [N, ncols] 的 int 矩阵 (= L0 dataset.tuples.numpy())。
        # col.discretize(series) = vocab 编码 + NaN 永远 bin_id=0 那套逻辑。
        self.dataset = np.stack([col.discretize(self.table.data[cname]) for cname, col in self.table.columns.items()], axis=1)
        self.algorithm = algorithm
        self.topological_sampling_order = topological_sampling_order
        self.num_samples = num_samples
        self.discrete_mapping = self.build_discrete_mapping(
            self.dataset, discretize, discretize_method)
        self.discrete_table = self.apply_discrete_mapping(
            self.dataset, self.discrete_mapping)
        L.info('calling BayesianNetwork.from_samples...')
        t = time.time()
        # pomegranate "一键学": 自动跑 algorithm (greedy / chow-liu / exact)
        # 搜 DAG 结构 + 用最大似然填 CPT。n_jobs=NUM_THREADS (= lecarb 统一线程
        # 上限, 让 BN 跟其它 estimator 在同样 CPU 预算下比)。
        # 详见 L0 的 detailed 教学注释。
        self.model = BayesianNetwork.from_samples(self.discrete_table,
                                                  algorithm=self.algorithm,
                                                  max_parents=max_parents,
                                                  n_jobs=NUM_THREADS,
                                                  root=root)
        L.info(f'done! took {(time.time() - t)/60:.2f} mins')

        def size(states):
            n = 0
            for state in states:
                if "distribution" in state:
                    dist = state["distribution"]
                else:
                    dist = state
                if dist["name"] == "DiscreteDistribution":
                    for p in dist["parameters"]:
                        n += len(p)
                elif dist["name"] == "ConditionalProbabilityTable":
                    for t in dist["table"]:
                        n += len(t)
                    if "parents" in dist:
                        for parent in dist["parents"]:
                            n += size(dist["parents"])
                else:
                    assert False, dist["name"]
            return n

        self.size = 4 * size(json.loads(self.model.to_json())["states"])
        L.info(f'model size is {self.size/1024/1024:.2f}MB')

        # print('json:\n', self.model.to_json())
        self.json_size = len(self.model.to_json())
        self.use_pgm = use_pgm
        #        print(self.model.to_json())

        # ============================================================
        # 决定列采样顺序 (拓扑排序)
        # ============================================================
        # 父节点先采, 子节点后采。简单 Kahn 算法雏形:
        # 每轮挑 "所有 parents 都已 ordered" 的节点加进去。
        # 详见 L0 同段代码注释。
        if topological_sampling_order:
            self.sampling_order = []
            while len(self.sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in self.sampling_order:
                        continue  # already ordered
                    if all(d in self.sampling_order for d in deps):
                        self.sampling_order.append(i)
                L.debug(f"Building sampling order {self.sampling_order}")
        else:
            self.sampling_order = list(range(len(self.model.structure)))
        L.info(f"Using sampling order {self.sampling_order} {str(self)}")

        # ============================================================
        # 可选: pgmpy 同结构重新 fit (精确推理用)
        # ============================================================
        # pomegranate.predict_proba 走近似 (loopy BP), pgmpy.VariableElimination
        # 是精确推理 — 慢但更准。这里把 pomegranate 学到的 DAG 结构搬到 pgmpy,
        # 用同一份数据重 fit (= 重填 CPT, 因为两个库 CPT 表示不同)。
        if use_pgm:
            from pgmpy.models import BayesianModel
            data = pd.DataFrame(self.discrete_table.astype(np.int64))
            # spec = list of (parent, child) tuples (DAG edge list)。
            # orphans = 无 parents 的节点 (要单独 add_node, 否则 pgmpy 跳过)。
            spec = []
            orphans = []
            for i, parents in enumerate(self.model.structure):
                for p in parents:
                    spec.append((p, i))
                if not parents:
                    orphans.append(i)
            L.info(f"Model spec {spec}")
            model = BayesianModel(spec)
            for o in orphans:
                model.add_node(o)
            L.info('calling pgm.BayesianModel.fit...')
            t = time.time()
            # pgmpy 用最大似然估计填 CPT。
            model.fit(data)
            L.info(f'done! took {(time.time() - t)/60:.2f} mins')
            self.pgm_model = model

    # ============================================================
    # __str__: estimator 名字 (写进结果 CSV / log)
    # ============================================================
    def __str__(self):
        return "bn-{}-{}-{}-{}-bytes-{}-{}-{}".format(
            self.algorithm,
            self.num_samples,
            "topo" if self.topological_sampling_order else "nat",
            self.size,
            # self.json_size,
            self.discretize,
            self.discretize_method if self.discretize else "na",
            "pgmpy" if self.use_pgm else "pomegranate")

    # ============================================================
    # Query: progressive sampling on BN (= L0 同方法)
    # ============================================================
    # 思路: 跟 ProgressiveSampling 一样, 逐列截断采样累乘 Zᵢ。
    # 差别仅在 condition 分布的来源:
    #   - MADE: model.forward 一次出所有列 logits
    #   - BN  : BN.predict_proba 或 VariableElimination 拿条件分布 (慢得多)
    # 完整流程 + draw_conditional / draw_conditional_pgm 两个内部函数解释
    # 见 L0 [estimators.py:BayesianNetwork.Query](../../../AllModels/Naru/estimators.py#L1095) 的详注。
    # 注意 L2 返回 (card, dur_ms) tuple, L0 只返回 card。
    def Query(self, query):
        # query_2_triple 替代 L0 的 FillInUnqueriedColumns; with_none=True 让
        # 没 predicate 的列补 None 占位。
        columns, operators, vals = query_2_triple(query, with_none=True)

        start_stmp = time.time()
        ncols = len(columns)
        nrows = self.discrete_table.shape[0]
        assert ncols == self.discrete_table.shape[1], (
            ncols, self.discrete_table.shape)

        def adjust_literals(col_id, op, val):
            col = list(self.table.columns.values())[col_id]
            if is_categorical(col.dtype):
                return col.discretize([val])[0]
            if op == '>=':
                assert val <= col.maxval, (col.name, val, col.maxval)
                val = col.vocab[np.argmax(col.vocab >= val)]
                return col.discretize([val])[0]
            elif op == '<=':
                assert val >= col.minval, (col.name, val, col.minval)
                val = col.vocab[::-1][np.argmax(col.vocab[::-1] <= val)]
                return col.discretize([val])[0]
            elif op == '[]':
                assert val[0] <= col.maxval, (col.name, val[0], col.maxval)
                assert val[1] >= col.minval, (col.name, val[1], col.minval)
                val0 = col.vocab[np.argmax(col.vocab >= val[0])]
                val1 = col.vocab[::-1][np.argmax(col.vocab[::-1] <= val[1])]
                return col.discretize([val0, val1])
            elif op == '=':
                assert val in col.vocab
                return col.discretize([val])[0]
            else:
                L.error(f"unknown operator: {op}")
                raise NotImplementedError

        def draw_conditional_pgm(evidence, col_id):
            """PGM version of draw_conditional()"""

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = adjust_literals(col_id, operators[col_id], vals[col_id])
                if operators[col_id] == '[]':
                    val = [self.apply_discrete_mapping_to_value(v, col_id, self.discrete_mapping) for v in val]
                else:
                    val = self.apply_discrete_mapping_to_value(val, col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if operators[col_id] == "<" and val == 0:
                        val += 1
                    elif operators[col_id] == ">" and val == self.max_val[col_id]:
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in enumerate(distribution):
                    if op(k, val):
                        p += v
                return p

            from pgmpy.inference import VariableElimination
            model_inference = VariableElimination(self.pgm_model)
            xi_distribution = []
            for row in evidence:
                e = {}
                for i, v in enumerate(row):
                    if v is not None:
                        e[i] = v
                result = model_inference.query(variables=[col_id], evidence=e)
                xi_distribution.append(result[col_id].values)

            xi_marginal = [prob_match(d) for d in xi_distribution]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in enumerate(d):
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        def draw_conditional(evidence, col_id):
            """Draws a new value x_i for the column, and returns P(x_i|prev).
            Arguments:
                evidence: shape [BATCH, ncols] with None for unknown cols
                col_id: index of the current column, i
            Returns:
                xi_marginal: P(x_i|x0...x_{i-1}), computed by marginalizing
                    across the range constraint
                match_rows: the subset of rows from filtered_rows that also
                    satisfy the predicate at column i.
            """

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = adjust_literals(col_id, operators[col_id], vals[col_id])
                if operators[col_id] == '[]':
                    val = [self.apply_discrete_mapping_to_value(v, col_id, self.discrete_mapping) for v in val]
                else:
                    val = self.apply_discrete_mapping_to_value(val, col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                            col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in distribution.items():
                    if op(k, val):
                        p += v
                return p

            xi_distribution = self.model.predict_proba(evidence,
                                                       max_iterations=1,
                                                       n_jobs=-1)
            xi_marginal = [
                prob_match(d[col_id].parameters[0]) for d in xi_distribution
            ]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in d[col_id].parameters[0].items():
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        p_estimates = [1. for _ in range(self.num_samples)]
        evidence = [[None] * ncols for _ in range(self.num_samples)]
        for col_id in self.sampling_order:
            if self.use_pgm:
                xi_marginal, xi_samples = draw_conditional_pgm(evidence, col_id)
            else:
                xi_marginal, xi_samples = draw_conditional(evidence, col_id)
            for ev_list, xi in zip(evidence, xi_samples):
                ev_list[col_id] = xi
            for i in range(self.num_samples):
                p_estimates[i] *= xi_marginal[i]

        dur_ms = (time.time() - start_stmp) * 1e3
        return np.round(np.mean(p_estimates) * nrows).astype(dtype=np.int32, copy=False), dur_ms

# ================================================================
# Result: 单条 query 结果的轻量容器
# ================================================================
# NamedTuple: 不可变 record, 字段 = (query index, 估计 cardinality, 耗时 ms)。
# 每个 Ray Worker 跑完一条 query 就 append 一个 Result 进 self.stats。
class Result(NamedTuple):
    i: int
    est_card: int
    dur_ms: float

# ================================================================
# Bayes: lecarb Estimator wrapper, 把 query 分发给 Ray Worker
# ================================================================
# 设计原因: BN 单 query 慢, 用 ray.remote 启 parallelism 个 worker 进程并行。
# 每个 worker 持有一个完整 BayesianNetworkWorker (= 包括独立的 BN model 副本),
# Driver 端只负责派发 query, 不真做 inference。
class Bayes(Estimator):
    def __init__(self, table, samples, discretize, parallelism):
        super(Bayes, self).__init__(table=table, version=table.version, samples=samples, discretize=discretize)
        self.num_workers = parallelism
        self.workers = []
        self.start_workers(parallelism)

    # ============================================================
    # start_workers: 启动 N 个 ray actor, 每个独立训一份 BN
    # ============================================================
    # 关键: 每个 worker 独立 build BayesianNetworkWorker (= 各自跑一遍 from_samples),
    # *没有* shared model. 这样 forward 不用 inter-process locking, 也不用
    # 把 BN model pickle 传过去 (pomegranate model 难 pickle)。
    # 代价: 启动慢 + 内存翻 parallelism 倍, 但 query 跑得起来。
    def start_workers(self, parallelism):
        import ray
        # ray.init: 启动 ray runtime; redis_password 是协调用认证 (开发用 dummy)。
        ray.init(redis_password='xxx')

        # @ray.remote: 把 class 变成 distributed actor。
        # 实例方法调用要写 worker.run_query.remote(...) (返 ObjectRef, 异步)。
        @ray.remote
        class Worker(object):
            def __init__(self, table, samples, discretize, i):
                # 默认硬编码用 chow-liu 算法 + max_parents=2 + equal_freq discretize。
                # ARELY 里实测这套配置在 DMV / Census 上效果最好 (跟 Naru paper 比 q-error)。
                self.estimator = BayesianNetworkWorker(table,
                                                       samples,
                                                       'chow-liu',
                                                       topological_sampling_order=True,
                                                       root=0,
                                                       max_parents=2,
                                                       use_pgm=False,
                                                       discretize=discretize,
                                                       discretize_method='equal_freq')
                self.i = i
                # stats: 累积本 worker 跑过的所有 Result。
                self.stats = []

            def run_query(self, query, j):
                # 收到 pickled query → 反序列化 → 跑 inference → 存 Result。
                # pickle 因为 ray.remote 跨进程传 args 需要序列化。
                query = pickle.loads(query)
                card, dur_ms = self.estimator.Query(query)
                self.stats.append(Result(i=j, est_card=card, dur_ms=dur_ms))
                if (j+1) % 10 == 0:
                    L.info(f'Finished {j+1} queries')

            def get_stats(self):
                # Driver 端调这个把所有 worker 的 stats 收回来合并。
                return self.stats

        L.info(f"construct {parallelism} bayesian network workers...")
        for i in range(parallelism):
            # Worker.remote(...): 在 ray 集群上创建一个 actor (= 一个进程)。
            self.workers.append(Worker.remote(self.table, self.params['samples'], self.params['discretize'], i))

    def query(self, query):
        # 同步接口故意空着: BN 太慢, 必须走 query_async。
        # lecarb 的 run_test(..., query_async=True) 会调 query_async 替代 query。
        pass

    def query_async(self, query, i):
        # 把 query 派给 i % num_workers 号 worker (round-robin 负载均衡)。
        # .remote(...) 立即返 ObjectRef, 不阻塞 driver。
        # run_test 之后会调 ray.get([w.get_stats.remote() for w in workers]) 收结果。
        self.workers[i % self.num_workers].run_query.remote(pickle.dumps(query), i)

# ================================================================
# test_bayesnet: lecarb CLI entry point
# ================================================================
# `lecarb test --estimator bayesnet --params "{'samples': 200, 'discretize': 100, 'parallelism': 50}"`
# - samples: progressive sampling 路径数 (= L0 args.bn_samples, paper 默认 200)
# - discretize: 二次粗化 bucket 数 (paper 默认 100)
# - parallelism: ray worker 进程数 (paper 默认 50; 内存允许就开大点)
def test_bayesnet(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the bayesian network is built from, might not be the same with the one we test on
        samples: # progressive samples of each inference
        discretize: # bins for each column
        parallelism: # threads to inference in parallel
    """
    np.random.seed(seed)

    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    estimator = Bayes(table, samples=params['samples'], discretize=params['discretize'], parallelism=params['parallelism'])
    L.info(f"built bayesian network estimator: {estimator}")

    # query_async=True 告诉 run_test 走 estimator.query_async 而不是 .query。
    # run_test 内部会做: 派发所有 query → 等所有 worker 完成 → 收集合并 stats → 写 CSV。
    run_test(dataset, version, workload, estimator, overwrite, query_async=True)
