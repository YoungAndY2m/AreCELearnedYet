# ================================================================
# 教学注释 (annotation pass) — mysql.py 总览
# ================================================================
# MySQL 优化器估计 baseline (跟 [postgres.py](postgres.py) 同思路, 换数据库)。
# **L0 Naru 没这个 baseline**, L2 新加的 — 让 paper 同时报 PG / MySQL 两个生产
# 系统的 CE 表现 (有时候它们差别挺大)。
#
# MySQL CE 内部原理
# ----------------------------------------------------------------
# MySQL 8.0+ 引入显式 histogram (跟 PG 类似), 用 `ANALYZE TABLE ... UPDATE
# HISTOGRAM ON cols WITH N BUCKETS` 命令建。
# Histogram 类型 (MySQL 自动选): equi-height (等深) / singleton (低基数列)。
# 推理: EXPLAIN 返回 "filtered" 列 (= selectivity, 单位是百分比) +
#       "rows" 列 (= 该 plan 节点的估计行数)。
#
# 跟 postgres.py 的差异
# ----------------------------------------------------------------
#   - 连接: mysql.connector vs psycopg2
#   - 直方图构造: `ANALYZE TABLE ... UPDATE HISTOGRAM` vs PG `ALTER COLUMN SET STATISTICS + ANALYZE`
#   - SQL 表名引号: ` (反引号) vs " (双引号), 通过 query_2_sql(dbms='mysql') 切换
#   - EXPLAIN 输出: MySQL 返回固定字段 (res[0][10] = filtered百分比), PG 返回 JSON
#   - 没有 setseed (MySQL 不暴露这接口)
#
# 估计公式
# ----------------------------------------------------------------
#   filtered (= 0.01 · percent) · row_num
# 0.01 因子: MySQL 返回的 filtered 是 0-100 的整数, 不是 [0, 1] 比例。
# ================================================================
import time
import mysql.connector
import logging
from typing import Any, Dict
import numpy as np

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_sql
from ..dataset.dataset import load_table
from ..constants import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PSWD

L = logging.getLogger(__name__)

# ================================================================
# MySQL: MySQL EXPLAIN estimator
# ================================================================
class MySQL(Estimator):
    def __init__(self, table, bucket, seed):
        super(MySQL, self).__init__(table=table, version=table.version, bucket=bucket, seed=seed)

        # mysql.connector: 官方 Python MySQL driver。
        self.conn = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PSWD, host=MYSQL_HOST, port=MYSQL_PORT, database=MYSQL_DB)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        # construct statistics
        # ANALYZE TABLE ... UPDATE HISTOGRAM ON c1, c2, ... WITH N BUCKETS:
        # 在指定列上建立 N 个 bucket 的 equi-height histogram。
        # 跟 PG `ALTER COLUMN SET STATISTICS + ANALYZE` 等价但更显式。
        # bucket 通常扫 16 / 64 / 256 / 1024 当 paper 实验参数。
        start_stmp = time.time()
        self.cursor.execute(f"analyze table `{self.table.name}` update histogram on "
                            f"{','.join([c.name for c in table.columns.values()])} "
                            f"with {bucket} buckets;")
        rows = self.cursor.fetchall()
        L.info(f"{rows}")
        dur_min = (time.time() - start_stmp) / 60

        L.info(f"construct statistics finished, using {dur_min:.4f} minutes")

    # ============================================================
    # query: EXPLAIN → 解析 row 估计
    # ============================================================
    # MySQL EXPLAIN 返回固定 12 列, res[0][10] 是 "filtered" (百分比, 0-100)。
    # 估计 cardinality = filtered/100 · table.row_num。
    # 注: 注释掉的 "test 2" 用 res[0][9] (= rows 列) 替代 row_num — 当 query
    # 走的不是全表扫描时才区分, 通常两个等价。
    def query(self, query):
        sql = 'explain {}'.format(query_2_sql(query, self.table, aggregate=False, dbms='mysql'))
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        assert len(res) == 1, res
        # test 1
        card = np.round(0.01 * res[0][10] * self.table.row_num)
        # test 2
        #  card = np.round(0.01 * res[0][10] * res[0][9])
        #  L.info(card)
        return card, dur_ms

# ================================================================
# test_mysql: lecarb CLI entry point
# ================================================================
# `lecarb test --estimator mysql --params "{'bucket': 256}"` 触发。
# 需要环境变量 MYSQL_HOST/PORT/DB/USER/PSWD 都设好 (见 constants.py)。
def test_mysql(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that mysql construct statistics, might not be the same with the one we test on
        bucket: number of bucket for each histogram
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    L.info("construct mysql estimator...")
    estimator = MySQL(table, params['bucket'], seed=seed)
    L.info(f"built mysql estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


