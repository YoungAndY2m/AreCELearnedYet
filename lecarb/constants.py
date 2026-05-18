# ================================================================
# 教学注释 (annotation pass) — lecarb 全局常量
# ================================================================
# 所有路径 / database URL / 资源预算都从这里读, 让代码不出现 hardcode。
# 强约束: 多个 env 变量必须在 `source export_env.sh` 后才能 import 这个文件,
# 否则 KeyError (= 比 .get(...) 早报错好, 防止意外用默认值)。
#
# 文件被 import 时的副作用 (top-level): 读 env + 推 DEVICE + 算 NUM_THREADS。
# ================================================================
import os
from pathlib import Path
import torch

# === 路径常量 (export_env.sh 设的) ===
# DATA_ROOT: 数据集 CSV / parquet / discrete table pickle 存的地方
DATA_ROOT = Path(os.environ["DATA_ROOT"])
# OUTPUT_ROOT: 训出的 model / 测出的 result / log 都放这下面
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"])
# MODEL_ROOT/{dataset}/{version}-*.pt or *.pkl
MODEL_ROOT = OUTPUT_ROOT / "model"
# 每次 run_test 输出的 q-error CSV
RESULT_ROOT = OUTPUT_ROOT / "result"
# 训练 / 推理日志
LOG_ROOT = OUTPUT_ROOT / "log"

# === 数据库连接 ===
# DATABASE_URL: 普通 Postgres (给 postgres.py estimator + mysql.py 用)
DATABASE_URL = os.environ["DATABASE_URL"]
# KDE_DATABASE_URL: 单独的 KDE-patched PG fork (给 feedback_kde.py 用)
KDE_DATABASE_URL = os.environ["KDE_DATABASE_URL"]
# MySQL 连接信息 (mysql.py 用; 拆 host/port/db/user/pswd 是因为 PyMySQL 接口要分开传)
MYSQL_HOST = os.environ["MYSQL_HOST"]
MYSQL_PORT = os.environ["MYSQL_PORT"]
MYSQL_DB = os.environ["MYSQL_DB"]
MYSQL_USER = os.environ["MYSQL_USER"]
MYSQL_PSWD = os.environ["MYSQL_PSWD"]

# pickle 协议版本: 4 支持大文件 + Python 3.4+, 比默认 protocol 兼容性好
PKL_PROTO = 4

# === 资源预算 ===
# DEVICE: cuda 可用就用 (Naru / MSCN 这种 nn-based estimator 走 GPU 快得多)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# NUM_THREADS: torch / numpy / Bayes net 的并发上限, 默认全部 CPU 核
# 实验时通常用 env CPU_NUM_THREADS 限到 4 / 8 (公平 benchmark, 不同 estimator 在同样预算下比)
NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count()))

# data-driven estimator (Naru / DeepDB) 训练时在线 valid q-error 用的 query 数
# 100 是平衡 valid 精度 vs 训练开销的折中值
VALID_NUM_DATA_DRIVEN = 100
