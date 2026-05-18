# ================================================================
# 教学注释 (annotation pass) — feedback_kde.py 总览
# ================================================================
# Feedback-aware Kernel Density Estimation (KDE) baseline。
# 注意: **不在 Naru L0 的 estimators.py 里出现**, 因为 KDE 不是 Naru paper
# 比较的 baseline (Naru paper 主要对照传统 DB 方法: PG / MaxDiff / Sampling /
# BayesNet)。KDE 是另一条研究线。
#
# 来源 paper
# ----------------------------------------------------------------
# - Heimel, Markl 2013 SIGMOD: "Self-Tuning, GPU-Accelerated Kernel Density
#   Models for Multidimensional Selectivity Estimation"
# - Kiefer, Heimel, Breß, Markl 2017 SIGMOD: "Estimating Join Selectivities
#   using Bandwidth-Optimized KDE" (= "feedback-aware" 版本, 用 query feedback
#   做带宽优化)
# ARELY 用的是 2017 版的 PG fork (= 把 KDE estimator 实装进 PostgreSQL).
#
# KDE 是什么 (一句话)
# ----------------------------------------------------------------
# 非参数密度估计: 给一组样本点 {x_1, ..., x_n}, 估计 PDF
#     p(x) ≈ (1/n) Σᵢ K_h(x - x_i)
# 其中 K_h 是 kernel 函数 (常用 Gaussian), h 是 bandwidth (= 带宽).
# 估 cardinality: card(R) ≈ N · ∫_R p(x) dx, 数值积分实现。
# 多维 KDE 用乘积 kernel: K_h(x-x_i) = ∏_d K_h(x_d - x_id).
# Bandwidth h 决定平滑度 — 小 h 过拟合, 大 h 过平滑。
# Feedback-aware 部分: 用历史 query 的真值反馈在线调 h, 让估计在该 workload 上更准。
#
# 为什么这文件才 97 行?
# ----------------------------------------------------------------
# **Python 不实现 KDE 算法**, 算法在外部 PG fork 的 C 代码里 (= AllModels/KDE/
# 那个 fork). 本文件只是 psycopg2 client: 把 query 喂给 KDE-enabled PG 实例,
# 读回 EXPLAIN 的 row 估计。
# 跟普通 postgres.py 区别: 连的是 KDE-patched PG (KDE_DATABASE_URL), 该 PG
# 有 `kde_*` 系列 GUC 参数 + `pg_kdemodels` / `pg_kdefeedback` 系统表。
#
# 训练流程 (跟 PG / Sampling 不同, KDE 需要训练!)
# ----------------------------------------------------------------
# 1. 构造 estimator: 开 kde_collect_feedback, 设 sample_num
# 2. train_batch(queries): 跑一批训练 query 让 KDE 收集 (query → 真值) feedback
# 3. 训完关 feedback collection, 开 bandwidth optimization → ANALYZE
#    (这一步让 KDE 用收集到的 feedback 调 bandwidth h)
# 4. test: 跟 postgres.py 一样 EXPLAIN 拿 Plan Rows
#
# 依赖
# ----------------------------------------------------------------
# - 一个跑 KDE-patched PG 的实例 (源码在 AllModels/KDE/ 的 PG fork, 要编译装)
# - KDE_DATABASE_URL 指向它 (跟普通 PG 是两个不同 instance)
# - GPU 推荐: 论文 paper 是 "GPU-Accelerated KDE", `ocl_use_gpu TO true` 启用 OpenCL
# ================================================================
import time
import logging
from typing import Any, Dict
import psycopg2

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_kde_sql, load_queryset
from ..dataset.dataset import load_table
from ..constants import KDE_DATABASE_URL

L = logging.getLogger(__name__)

# ================================================================
# FeedbackKDE: KDE estimator wrapper (= 外部 KDE-PG 的 client)
# ================================================================
class FeedbackKDE(Estimator):
    def __init__(self, table, ratio, train_num, seed):
        super(FeedbackKDE, self).__init__(table=table, version=table.version, ratio=ratio, train_num=train_num, seed=seed)
        # KDE 内部要从全表抽 ratio 比例当 kernel 中心点 (= 样本)。
        # 比 simple sampling 复杂一点: KDE 不止用 sample 算 selectivity,
        # 而是把 sample 当 PDF 的 "kernel centers" 用 (越多越准, 越多越慢)。
        self.sample_num = int(table.row_num * ratio)
        L.info(f"Going to collect {self.sample_num} samples")

        # 连接 KDE-patched PG (注意! 跟普通 postgres.py 的 DATABASE_URL 不同)。
        self.conn = psycopg2.connect(KDE_DATABASE_URL)
        # 'read uncommitted' + autocommit: 不要事务隔离, 让训练 query 立即可见。
        self.conn.set_session('read uncommitted', autocommit=True)
        self.cursor = self.conn.cursor()

        # Make sure that debug mode is deactivated and that all model traces are removed (unless we want to reuse the model):
        # 随机种子可复现 (PG 的 setseed, 同 postgres.py)。
        self.cursor.execute(f"SELECT setseed({1/seed});")
        # self.cursor.execute("SET kde_debug TO true;")
        # ========= KDE-specific GUC 参数 (这些都是 PG fork 加的) =========
        self.cursor.execute("SET kde_debug TO false;")
        # GPU 加速 (paper 卖点之一): OpenCL 实现的 kernel sum, 比 CPU 快 10x+
        self.cursor.execute("SET ocl_use_gpu TO true;")
        # bandwidth optimization 时用的 loss 函数: Quadratic = MSE
        # (KDE paper 也支持 KL / absolute, Quadratic 是默认推荐)
        self.cursor.execute("SET kde_error_metric TO Quadratic;")

        # Remove all existing model traces if we don't reuse the model.
        # 清空 KDE 系统表 (KDE fork 引入的两张表):
        #   - pg_kdemodels: 已建好的 KDE 模型注册
        #   - pg_kdefeedback: 历史 query 反馈
        # 不清的话上次的 model + feedback 会影响这次, 实验不干净。
        self.cursor.execute("DELETE FROM pg_kdemodels;")
        self.cursor.execute("DELETE FROM pg_kdefeedback;")
        # pg_stat_reset: 重置 PG 统计信息 (跟 ANALYZE stats 不同, 这是运行时统计)
        self.cursor.execute("SELECT pg_stat_reset();")

        # KDE-specific parameters.
        # 真正激活 KDE estimator 的 3 个参数:
        # 1. kde_samplesize: 几个 kernel center (= self.sample_num)
        # 2. kde_enable: 让 optimizer 用 KDE 替代默认 histogram 估计
        # 3. kde_collect_feedback: 训练阶段开 (true), 让 KDE 在每次 query 时
        #    收集 (predicate, 真值) 反馈写进 pg_kdefeedback。
        self.cursor.execute(f"SET kde_samplesize TO {self.sample_num};")
        self.cursor.execute("SET kde_enable TO true;")
        self.cursor.execute("SET kde_collect_feedback TO true;")

    # ============================================================
    # train_batch: 跑一批训练 query 让 KDE 收集反馈 + 优化 bandwidth
    # ============================================================
    # 关键流程:
    #   1. 把每条训练 query 真跑一遍 (= EXPLAIN ANALYZE 风格执行, KDE 自动收集
    #      (predicate, 真行数) 写进 pg_kdefeedback)
    #   2. 关 feedback collection
    #   3. 开 bandwidth optimization, 让 KDE 用收集到的 feedback 调每维的 h
    #   4. 设 statistics 桶数 + ANALYZE → 让 KDE 选 sample (= 哪些行当 kernel center)
    #   5. 把 sample dump 到 /tmp 文件 (实验可复现 / 调试用)
    def train_batch(self, queries):
        for i, query in enumerate(queries):
            # query_2_kde_sql: lecarb 工具, 跟 query_2_sql 类似但 KDE fork 可能要不同语法。
            self.cursor.execute(query_2_kde_sql(query, self.table))
            if (i + 1) % 100 == 0:
                L.info(f"{i+1} queries done")
        L.info("Finishing running all training queries")

        # 切到 "用 feedback 优化 bandwidth" 模式。
        self.cursor.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.
        self.cursor.execute("SET kde_enable_bandwidth_optimization TO true;")
        # 用最近 N=len(queries) 条 feedback 算 bandwidth (= 全部训练 query)。
        self.cursor.execute(f"SET kde_optimization_feedback_window TO {len(queries)};")

        # 设单列 histogram 桶数 = 100 (兜底, 万一 KDE fallback 到普通 histogram 时用)。
        stat_cnt = 100
        for c in self.table.columns.values():
            self.cursor.execute(f"alter table \"{self.table.name}\" alter column {c.name} set statistics {stat_cnt};")

        # ANALYZE 触发 KDE 真正建模 (= 选 sample 点 + 算 bandwidth)。
        self.cursor.execute(f"analyze \"{self.table.name}\"({','.join(self.table.columns.keys())});")

        # 把 KDE 选出的 kernel center sample 写到磁盘文件 (调试 / 复现用)。
        sample_file = f"/tmp/sample_{self.table.name}.csv"
        self.cursor.execute(f"SELECT kde_dump_sample('{self.table.name}', '{sample_file}');")

    # ============================================================
    # query: 跟 postgres.py 一样 EXPLAIN 拿 Plan Rows
    # ============================================================
    # 区别: PG 后端走 KDE estimator (因为 kde_enable=true), 不是默认 histogram。
    def query(self, query):
        sql = f"explain(format json) {query_2_kde_sql(query, self.table)}"

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        return card, dur_ms

# ================================================================
# test_kde: lecarb CLI entry point
# ================================================================
# `lecarb test --estimator feedback_kde --params "{'ratio': 0.01, 'train_num': 1000}"`
# 触发。跟 sample / postgres 不同的是这里 *训练 + 测试* 都在一个 entry 里
# (KDE 需要训练 query 来调 bandwidth)。
def test_kde(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that postgres construct statistics, might not be the same with the one we test on
        ratio: ratio of the sample size
        train_num: number of queries use to train
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    table = load_table(dataset, params.get('version') or version)
    train_num = params['train_num']

    # ========= 载入训练 query =========
    # workload 的 'train' split 是给 query-driven 方法用的 (MSCN / KDE 都要)。
    # data-driven 方法 (Naru / DeepDB) 不用这部分; 它们只看数据。
    L.info("load training workload...")
    queries = load_queryset(dataset, workload)['train'][:train_num]

    L.info("construct postgres estimator...")
    estimator = FeedbackKDE(table, ratio=params['ratio'], train_num=train_num, seed=seed)

    # ========= 训练 KDE =========
    # train_batch 真跑训练 query (收集 (predicate, 真值) feedback) → 调 bandwidth。
    # 训练时间跟 train_num 成正比, paper 实验通常 1000-10000 query。
    L.info(f"start training with {train_num} queries...")
    start_stmp = time.time()
    estimator.train_batch(queries)
    dur_min = (time.time() - start_stmp) / 60
    L.info(f"built kde estimator: {estimator}, using {dur_min:1f} minutes")

    # 测试阶段跟其它 estimator 一样走 run_test。
    run_test(dataset, version, workload, estimator, overwrite)


