# ================================================================
# 教学注释 (annotation pass) — postgres.py 总览
# ================================================================
# 把 query 喂给真实 PostgreSQL 拿优化器估计 (= EXPLAIN 输出 "Plan Rows")
# 当 CE baseline。跟 Naru L0 [estimators.py:Postgres](../../../AllModels/Naru/estimators.py#L868) 同思路, lecarb 重写。
#
# PG 内部怎么估?
# ----------------------------------------------------------------
# PG 优化器有以下统计 (ANALYZE 命令生成, 存在 pg_stats 视图):
#   - 单列 histogram (等深, 桶数由 stat_target 控制, 默认 100)
#   - Most Common Values (MCV) 列表 (该列前 N 个最常见值的频次)
#   - distinct value 数估计
#   - null 比例
# Cardinality 估算公式 (简化):
#   sel(col op v) = lookup histogram + MCV → 该列 selectivity
#   joint sel = ∏ 各列 sel  (**默认假设列独立!**)
# 所以 PG 在相关列上吃亏跟 [estimators.py:Heuristic](../../../AllModels/Naru/estimators.py#L729) 一个原理 (独立假设)。
# Naru paper 用 PG 当 "生产系统真实表现" 代表。
#
# 跟 L0 Naru Postgres 的差异
# ----------------------------------------------------------------
#   - L0: 写死 database='dmv', port=None, 直接 ANALYZE 默认桶数
#   - L2: 用 DATABASE_URL (env 变量), 灵活配置
#   - L2 新加 stat_target 参数 → `alter column SET STATISTICS N`,
#     控制 PG 单列 histogram 桶数 (10-10000), 桶数越多越准但内存越大。
#     ARELY 用它做 "model footprint vs accuracy" 扫描 (跟 mhist limit 角色一样)。
#   - L2 用 query_2_sql (lecarb util) 生成 SQL, L0 自己写 QueryToPredicate
#   - L2 测 statistics 大小 (查 pg_stats 表的字节数), 给 paper 报 "histogram size"
#   - L2 setseed 让 ANALYZE 可复现 (PG 采样收集 stats 默认随机)
#   - L2 多一个 query_sql() 方法接受 raw SQL 字符串 (调试用)
#   - L2 加 test_postgres() entry point
#
# 依赖
# ----------------------------------------------------------------
#   - 需要一个运行中的 PG server, DATABASE_URL 指向它
#   - 数据要预先 COPY 进 PG (lecarb 有 import_pg 命令)
#   - psycopg2 (Python ↔ PG 客户端)
# ================================================================
import time
import psycopg2
import logging
from typing import Any, Dict

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_sql
from ..dataset.dataset import load_table
from ..constants import DATABASE_URL

L = logging.getLogger(__name__)

# ================================================================
# Postgres: PG EXPLAIN estimator
# ================================================================
class Postgres(Estimator):
    def __init__(self, table, stat_target, seed):
        # stat_target / seed 进 self.params 给 __repr__ + 结果 CSV 用。
        super(Postgres, self).__init__(table=table, version=table.version, stat=stat_target, seed=seed)

        # psycopg2 标准连接 + cursor。autocommit=True 每条 SQL 立即提交, 省 transaction 管理。
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        # construct statistics
        start_stmp = time.time()
        # ========= setseed: 让后面 ANALYZE 的随机采样可复现 =========
        # PG ANALYZE 内部从表里随机采几千行算 stats; setseed(x), x ∈ [-1, 1]
        # 让伪随机序列固定. 这里 1/seed 是为了把外部 seed 整数映射到这个区间。
        # (注意 seed=0 会报错, 一般 seed > 0)
        self.cursor.execute('select setseed({});'.format(1 / seed))
        # ========= 设每列的 histogram 桶数上限 =========
        # PG `ALTER COLUMN ... SET STATISTICS N`: 让 ANALYZE 给该列建 N 个等深桶
        # (1 ≤ N ≤ 10000, 默认 100). 桶多 → 精度高 + 内存大. 全部列设同一个 N 让
        # benchmark 控制变量公平。
        for c in table.columns.values():
            self.cursor.execute('alter table \"{}\" alter column {} set statistics {};'.format(
                table.name, c.name, stat_target))
        # ========= ANALYZE: 触发 PG 重新扫表算 stats =========
        # 没 ANALYZE 过的表 PG 用默认 cardinality (~1000), 估计离谱。
        # ANALYZE 写完 pg_stats 表, 之后 EXPLAIN 才能用上。
        self.cursor.execute('analyze \"{}\";'.format(self.table.name))
        self.conn.commit()
        dur_min = (time.time() - start_stmp) / 60

        # get size
        # ========= 测 PG histogram + MCV 占多少字节 =========
        # 查 pg_stats 视图, pg_column_size 给每行的字节数, sum 累加。
        # 给 paper 报 "Postgres baseline 占多少内存", 对照 Naru / MHIST 的 model size。
        self.cursor.execute('select sum(pg_column_size(pg_stats)) from pg_stats where tablename=\'{}\''.format(self.table.name))
        size = self.cursor.fetchall()[0][0]
        #  self.cursor.execute('select sum(pg_column_size(pg_stats_ext)) from pg_stats_ext where tablename=\'{}\''.format(self.table.name))
        #  res = self.cursor.fetchall()[0][0]
        # might not have content in ext table
        # ↑ 注释掉的代码是想加上 multivariate statistics (PG 10+ 支持的扩展统计)
        # 的大小, 但 ARELY 没用扩展统计 (= 强制 PG 走 "纯独立假设" 路径).
        #  if res is not None:
        #      size += res
        size = size / 1024 / 1024 # MB

        L.info(f"construct statistics finished, using {dur_min:.4f} minutes, All statistics consumes {size:.2f} MBs")

    def query(self, query):
        # ========= 构造 EXPLAIN SQL =========
        # query_2_sql(query, table, aggregate=False) 生成不带 COUNT(*) 的 SELECT * 语句,
        # 因为只要优化器的 row 估计, 不真执行。
        # explain(format json) 让 PG 输出结构化 JSON 而不是文本表格, 解析更稳。
        sql = 'explain(format json) {}'.format(query_2_sql(query, self.table, aggregate=False))
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        # JSON 三层嵌套 (rows × cols × outer-array), [0][0][0] 取唯一的 plan dict。
        # 'Plan' → 'Plan Rows' = 优化器的 row 估计 (= 我们要的 cardinality)。
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        return card, dur_ms

    # ============================================================
    # query_sql: 接受 raw SQL 字符串 (调试 / 自定义 query 用, 主流程不走)
    # ============================================================
    def query_sql(self, sql):
        sql = 'explain(format json) {}'.format(sql)
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

# ================================================================
# test_postgres: lecarb CLI entry point
# ================================================================
# `lecarb test --estimator postgres --params "{'stat_target': 100}"` 触发。
# stat_target 是必填参数 (paper 实验扫 10 / 100 / 1000 / 10000)。
def test_postgres(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that postgres construct statistics, might not be the same with the one we test on
        stat_target: size of the statistics limit
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    # data shift 实验: stats 在 A 版本上建、test 在 B 版本上跑 (看 PG 的 stats 失效有多快)。
    table = load_table(dataset, params.get('version') or version)

    L.info("construct postgres estimator...")
    estimator = Postgres(table, stat_target=params['stat_target'], seed=seed)
    L.info(f"built postgres estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


