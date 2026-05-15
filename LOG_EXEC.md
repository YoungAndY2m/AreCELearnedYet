# AreCELearnedYet — 执行命令手册（LOG_EXEC）

> 此文件汇总所有「实际执行」用的命令：环境准备 → 数据 → 实验跑测 → 进度查看 → 结果分析。
> 配合 [LOG.md](LOG.md) 一起读：LOG.md 解释「为什么/改了什么」，本文件解释「怎么跑」。
> 所有命令以最新 [week-3 PLAN](../UniMelb/Vault/wiki/in-progress/0_workplan/week-3/PLAN.md) Step 0 为准。

---

## 0. 前置：每次新终端必做

```bash
cd ~/Desktop/AreCELearnedYet
export PATH=~/.local/bin:$PATH       # just / uv 安装在用户态
source .venv/bin/activate            # 激活 uv 创建的虚拟环境
source export_env.sh                 # 把 .env 变量 export 给 lecarb（DATA_ROOT 等）
```

跳过任一行后续命令会报：
- `command not found: just/uv` → 没 export PATH
- `ModuleNotFoundError: torch/pandas/...` → 没 activate venv
- `KeyError: 'DATA_ROOT'` → 没 source export_env.sh

---

## 1. 从零环境配置（Fresh clone 后第一次）

### 1.1 系统依赖（一次性 sudo）

```bash
# 一次性装好 ARELY 所需的 apt 系统库
sudo apt update
sudo apt install -y unixodbc-dev libhdf5-dev postgresql postgresql-contrib

# Postgres 起服务并建 card 用户/db（lecarb 的 PostgreSQL 估计器用）
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE USER card WITH SUPERUSER LOGIN PASSWORD 'card';"
sudo -u postgres psql -c "CREATE DATABASE card OWNER card;"

# 验证
PGPASSWORD=card psql -h localhost -U card -d card -c "SELECT version();"
```

### 1.2 用户态工具链（一次性）

```bash
# just（task runner）
mkdir -p ~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin

# uv（Python 包管理器；自带 python download）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证
~/.local/bin/just --version
~/.local/bin/uv --version
```

### 1.3 Clone 项目

```bash
cd ~/Desktop
git clone https://github.com/YoungAndY2m/AreCELearnedYet.git
cd AreCELearnedYet

# 添加 upstream（可选，方便后续拉 sfu-db 更新）
git remote add upstream https://github.com/sfu-db/AreCELearnedYet.git
git remote -v
```

### 1.4 装 Python 依赖

```bash
cd ~/Desktop/AreCELearnedYet

# 必须带 SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# 否则 uv.lock 中锁住的 sklearn==0.0.post12（deprecated placeholder）构建失败
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True ~/.local/bin/uv sync

# 验证 409 packages
ls .venv/lib/python3.8/site-packages/ | wc -l

# 验证关键 import
.venv/bin/python -c "import torch, pandas, pomegranate, sklearn, psycopg2, tables, pyodbc, ray, xgboost, spn; print('all OK')"
```

### 1.5 下载并解压数据

```bash
cd ~/Desktop/AreCELearnedYet

# 方法 A：用 Justfile 内置 target（推荐）
export PATH=~/.local/bin:$PATH
just download-data    # 等价于 wget + tar -xzf

# 方法 B：手动
wget -O data.tar.gz "https://www.dropbox.com/scl/fi/ghx8wh117tpr2rcf3y6le/data.tar.gz?rlkey=h4bdblx75ktdy5uolibo0ldpm&st=mvltaztn&dl=1"
tar -xzf data.tar.gz

# 验证
ls data/      # 应有 census13/ dmv11/ forest10/ power7/
ls data/census13/  # 应有 original.csv 和 workload/
```

### 1.6 `.env` 端口对齐（**必做**）

```bash
# 检查当前 .env 中 DATABASE_URL 的端口
grep "^DATABASE_URL" .env

# Lankadinee 默认 6667，但本机 PostgreSQL 装在 5432 → 改成 5432
sed -i 's|@localhost:6667/card|@localhost:5432/card|' .env

# 确认
grep "^DATABASE_URL" .env
# 期望：DATABASE_URL=postgres://card:card@localhost:5432/card

# 验证连通
PGPASSWORD=card psql -h localhost -U card -d card -c "SELECT 1;"
```

### 1.7 烟测：建 Census Table 对象

```bash
cd ~/Desktop/AreCELearnedYet
export PATH=~/.local/bin:$PATH
source .venv/bin/activate
source export_env.sh

just csv2pkl data/census13/original.csv    # → data/census13/original.pkl
just pkl2table census13 original           # → output/.../census13_original Table object

# 期望最后看到：
#   build finished: Table census13_original (48842 rows, 4.84MB, columns: ...)
#   dump table to disk...
```

到此环境完整。

---

## 2. Day 1 实验执行

> 参照 [week-3/LOG.md Day 1 计划](../UniMelb/Vault/wiki/in-progress/0_workplan/week-3/LOG.md#L205-L214)：
> 「ARELY 环境可用 + Census 上至少 3 个 estimator 有 Q-Error 数字」
> 推荐顺序：PostgreSQL → Sample → MHist（quick）→ Naru/MSCN/DeepDB/LW-NN/BayesNet（learned）

### 2.1 准备日志目录

```bash
cd ~/Desktop/AreCELearnedYet
mkdir -p logs
```

### 2.2 三个 quick estimator（无需训练，直接 test）

```bash
# 前置：保证已 source 完毕（见 §0）

# --- Sample (~秒级) ---
just test-sample census13 original base 0.015 original 123 2>&1 | tee logs/sample.log

# --- MHist (建议先用小 bins 验证管线，再跑大的) ---
just test-mhist census13 original base 1000 original 123 2>&1 | tee logs/mhist-1k.log
# 验证通过后再跑论文用的 num_bins=30000（可能 5-10 分钟）
just test-mhist census13 original base 30000 original 123 2>&1 | tee logs/mhist-30k.log

# --- PostgreSQL (依赖 §1.6 端口对齐 + PG 服务在跑) ---
# 注意：test-postgres 需要先把数据加载到 Postgres 表里
just census2postgres original census13              # 加载数据
just test-postgres census13 original base 10000 original 123 2>&1 | tee logs/postgres.log
```

### 2.3 三个 learned estimator（先训练再 test）

```bash
# --- BayesNet (**实测慢**：~30 queries/min on Census, 10K queries ≈ 5h. 建议进 tmux 后台跑，或降 samples=50 加速 4x) ---
# 完整精度（论文配置）：
just test-bayesnet census13 original base 200 100 50 original 123 2>&1 | tee logs/bayesnet.log
# Day 1 快通版（精度下降，~1h）：
just test-bayesnet census13 original base 50 100 50 original 123 2>&1 | tee logs/bayesnet-s50.log

# --- Naru (训练慢，CPU 上 30+min；GPU 推荐) ---
just train-naru census13 original 4 32 4 embed embed True 0 0 100 base 123 2>&1 | tee logs/naru-train.log
# 训练完模型名格式如 naru-... 在 output/model/census13/ 下
just test-naru <model_filename> 2000 census13 original base 123 2>&1 | tee logs/naru-test.log

# --- MSCN (训练 ~10min) ---
just train-mscn census13 original base 1000 16 200 1024 100000 0 123 2>&1 | tee logs/mscn-train.log
just test-mscn <model_filename> census13 original base 123 2>&1 | tee logs/mscn-test.log

# --- DeepDB (训练 ~5-15min) ---
just train-deepdb census13 original 1000000 0.3 0.01 0 base 123 2>&1 | tee logs/deepdb-train.log
just test-deepdb <model_filename> census13 original base 123 2>&1 | tee logs/deepdb-test.log

# --- LW-NN (训练 ~10min) ---
just train-lw-nn census13 original base 128_64_32 200 10000 32 0 123 2>&1 | tee logs/lw-nn-train.log
just test-lw-nn <model_filename> census13 original base True 123 2>&1 | tee logs/lw-nn-test.log
```

> ⚠️ `<model_filename>` 要点：**只传文件名主干**，去掉路径前缀（`output/model/{ds}/`）和扩展名（`.pt`）。Lecarb 内部会拼成 `output/model/{ds}/{model}.pt`，你多传 → 它叠加 → 报 `No such file: output/model/census13/output/model/census13/...pt.pt`。
> 文件名带逗号时（如 Naru 的 `resmade_hid32,32,32,32_...`）**必须用单引号**包起来防 shell 误处理。
> 看实际可用 model：`ls output/model/census13/ | sed 's/\.pt$//'`

### 2.4 收集 q-error

```bash
# 列出所有结果 csv（文件名格式实测：{version}-{workload}-{est}-{params}.csv）
ls -lt output/result/census13/ | head -20

# 计算每个结果文件的 q-error 统计（max / p99 / p95 / median / mean）
just report-error <实际csv文件名> census13

# 实测样例（注意：含 `;` 或 `=` 的文件名必须单引号包起来）
just report-error original-base-mhist-bins=1000.csv census13
just report-error original-base-mhist-bins=30000.csv census13
just report-error 'original-base-postgres-version=original;stat=10000;seed=123.csv' census13
just report-error 'original-base-sampling-version=original;ratio=0.015;seed=123.csv' census13
```

一次性全跑所有结果文件：
```bash
for f in output/result/census13/*.csv; do
    echo "=== $(basename "$f") ==="
    just report-error "$(basename "$f")" census13
done | tee logs/all-qerror.txt
```

---

## 3. 进度查看（lecarb 无内置 tqdm）

| 场景 | 命令 |
|------|------|
| **前台 + stdout 流** | 直接跑命令（不加 `&`），DEBUG log 自动打印每个 query |
| **后台 + tail** | `... 2>&1 \| tee logs/X.log &` 然后 `tail -f logs/X.log` |
| **看输出文件长大** | `watch -n 5 'ls -la output/result/census13/'` |
| **CPU/内存** | `htop -F lecarb` 或 `top -p $(pgrep -d, -f lecarb)` |
| **杀掉跑飞的进程** | `pkill -f "lecarb test"` |
| **行级进度条**（可选改 lecarb） | 在 `lecarb/estimator/utils.py` 的 test loop 把 `for q in queries` 改为 `from tqdm import tqdm; for q in tqdm(queries)`。`tqdm` 已经装好了 |

---

## 4. 常见错误速查

| 报错 | 原因 | 解法 |
|------|------|------|
| `KeyError: 'DATA_ROOT'` | 没 `source export_env.sh` | 重新 source |
| `ModuleNotFoundError: torch` | 没 activate venv | `source .venv/bin/activate` |
| `sql.h: No such file` 跑 uv sync 时 | 缺 `unixodbc-dev` | `sudo apt install unixodbc-dev` |
| `Failed to build sklearn==0.0.post12` | 没设环境变量 | `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True uv sync` |
| `connection refused on port 6667/5432` | Postgres 没起 / 端口没对齐 | 见 §1.6 + `sudo systemctl start postgresql` |
| `Cython.Compiler.Errors.CompileError: pomegranate/utils.pyx` | 用 poetry 装老 pomegranate | 必须用 uv（本 repo 已迁移） |
| `ImportError: No module named spflow` | spflow 包名是 spn | 改 `import spn` |

---

## 5. 与 Week 3 PLAN 的映射

| LOG_EXEC 章节 | PLAN.md 对应 | LOG.md Day 1 对应 |
|---------------|-------------|------------------|
| §1 环境配置 | Step 0 准备 | "ARELY 环境可用" |
| §2.2 quick estimator | "先跑快速 estimator: PostgreSQL + Sample + MHist" | "至少 3 个 estimator 有 Q-Error" |
| §2.3 learned estimator | "再跑 learned: Naru → MSCN → DeepDB → LW-NN → BayesNet" | Day 2+ |
| §2.4 收 q-error | "记录: 每个方法的 Q-Error (max, p99, p95, median, mean)" | Day 1 产出 |
