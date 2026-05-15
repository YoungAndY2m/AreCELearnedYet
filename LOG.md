# AreCELearnedYet — 修改与配置日志

> 此文件记录该 repo 从 clone 后所有重要修改，供未来回溯参考。遵循 [[Desktop/CLAUDE.md]] 规范。

---

## 元数据

- **Fork 来源**: `YoungAndY2m/AreCELearnedYet`（origin），fork 自 `sfu-db/AreCELearnedYet`（upstream）
- **Clone 时间**: 2026-05-15
- **初始 commit**: `aa52da7` — "Update README.md"（原 sfu-db 最后 commit）
- **关联工作计划**: [[UniMelb/Vault/wiki/in-progress/0_workplan/week-3/LOG]], [[week-3/PLAN]], [[week-3/THOUGHT]], [[week-3/week-2-conclude]]
- **平行参考 repo**: `Desktop/CoLSE_AreCELearnedYet/`（= `Lankadinee/AreCELearnedYet`，22 commits ahead；本 repo 借用了它的 uv 配置，**Lankadinee fork 不可修改**）

---

## 修改记录

> 最新在最上面。

### [2026-05-15] Poetry → uv 构建系统迁移（vendored from Lankadinee fork）

- **目的**: 原 `sfu-db/AreCELearnedYet` 的 poetry 配置在现代 Python 环境上装不通 —— `sklearn==0.0.post12`（已废弃 placeholder）和 `pomegranate 0.13.5`（Cython 编译失败）两个雷使 `just install-dependencies` 跑不动。Lankadinee 的 fork 把构建系统改为 uv + 全 `==` 精确锁，解决了所有依赖问题。本次将 Lankadinee 的 uv 配置 vendored 到本 fork。
- **修改文件**:
  - `pyproject.toml` — 从 poetry 1.x 格式重写为 PEP 621 + uv 格式，所有版本约束改为 `==` 精确锁
  - `Justfile` — 所有 `poetry run` → `uv run`；新增 `wget-data` / `extract-data` / `download-data` target；`csv2pkl` script shebang 改为 `#!/usr/bin/env -S uv run --script`
- **新增文件**:
  - `pyproject.bk.toml` — 原 poetry 版 pyproject 备份（保留用于回溯对比）
  - `.python-version` — 内容 `3.8`，uv/pyenv 标识
  - `uv.lock` — 6053 行完全锁定的依赖图
  - `export_env.sh` — 3 行 bash：`export PYTHONPATH=$(pwd)/lecarb` + 从 `.env` 加载环境变量
- **影响**:
  - 安装方式从 `just install-dependencies`（poetry）→ `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True uv sync` + 装系统依赖 `unixodbc-dev libhdf5-dev`
  - 运行前需要 `source .venv/bin/activate && source export_env.sh`（否则 lecarb 找不到 `DATA_ROOT` 环境变量）
  - **lecarb/ 源代码不动**（仅 `manipulate_dataset.py` 与 Lankadinee 微差，不影响 Day 1）
- **来源**: 完整对比详见下方 "配置文件 —— 关键差异" 章节

### [2026-05-15] 环境配置完成

- **uv sync 成功**: 409 packages 装好；torch 1.5.0 / pandas 1.0.3 / pomegranate 0.13.4 / sklearn 0.23.0 / psycopg2 / tables / pyodbc / ray / xgboost / spflow(spn) 全部 import OK
- **数据**: `tar -xzf data.tar.gz` 解开 → `data/{census13, dmv11, forest10, power7}/`，含 csv + workload + 预生成 labels
- **PostgreSQL**: 本机已装 PostgreSQL 14.22（apt install），创建 `card`/`card` 用户和 db；但默认在 5432 端口，`.env` 的 `DATABASE_URL=postgres://card:card@localhost:6667/card` 仍指向 6667 —— **未对齐**，跑 PG 估计器前需要二选一调整
- **烟测通过**: `just csv2pkl data/census13/original.csv` + `just pkl2table census13 original` 生成 Census Table 对象（48,842 rows, 13 cols）

---

## 当前环境配置

- **Python**: 3.8.20（`uv` 自动管理，位于 `.venv/bin/python`）
- **包管理器**: `uv 0.11.14`（`~/.local/bin/uv`）
- **Task runner**: `just 1.51.0`（`~/.local/bin/just`）
- **系统依赖**（apt）: `unixodbc-dev`, `libhdf5-dev`, `postgresql`, `postgresql-contrib`
- **PostgreSQL**: 14.22 active on `localhost:5432`，用户 `card` 密码 `card`，db `card`
- **环境变量**: `.env` + `export_env.sh`（必须 source 才能跑 lecarb）

---

## 已知 pitfall

1. **`uv sync` 第一次必须带 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True`** —— 否则 `sklearn==0.0.post12`（uv.lock 锁住的 deprecated placeholder）构建失败
2. **必须先装 `unixodbc-dev`** —— 否则 `pyodbc 4.0.30` 因缺 `sql.h` 编译失败。`pyodbc` 实际代码中没人 import，但 pyproject.toml 列着，所以必须装系统依赖让 uv 能装
3. **必须先 `source export_env.sh`** —— 否则 `lecarb/constants.py` 在 `os.environ["DATA_ROOT"]` 处抛 `KeyError`
4. **`.env` 端口与本机 Postgres 不一致** —— 用户保留了 `DATABASE_URL=...:6667/card` 但本机 Postgres 跑在 5432。Day 1 三个非 DB estimator（MHist/Sample/BayesNet）不受影响，但跑 PostgreSQL 估计器（`test-postgres`）前必须对齐
5. **`spflow` 包名 ≠ 模块名** —— 装的是 `spflow==0.0.34`，但 `import` 用 `spn`（不要找 `import spflow`）
6. **MHist 慢** —— `num_bins=30000` 在 13 列 census 上跑 5+ 分钟仍未出结果（user 中断过一次）；测试时可降低 num_bins 或跑别的 estimator 先
7. **BayesNet 极慢** —— Census 10K test queries 实测 ~30 queries/min，全跑完 ~5 小时（pomegranate progressive sampling + ray 并行）。Day 1 想凑数：降 `samples` 从 200 到 50（速度 4x，精度下降）；或直接换 Postgres 当第 3 个 estimator
8. **test-naru/test-deepdb/... 的 model 参数**：只传文件名主干，去掉路径前缀和 `.pt` 扩展名。Lecarb 内部 path = `output/model/{ds}/{model_arg}.pt`，多传任何前缀/后缀会叠加报 `No such file`。带逗号的文件名（如 `resmade_hid32,32,32,32_...`）记得加单引号

---

## 与 Week 3 PLAN 的对应

PLAN Step 0 检查项（详见 [[week-3/PLAN.md#Step 0]]）：
- [x] Clone `sfu-db/AreCELearnedYet` → fork 到 `YoungAndY2m/`
- [x] 安装依赖（**uv 路线，不走 poetry**）
- [x] 配置 `.env`（端口需对齐后才完整）
- [x] Census 上跑 `csv2pkl` + `pkl2table`（Table 对象生成）
- [ ] Census 上跑 ≥3 estimator → q-error 数字（Day 1 目标）
- [ ] 跑 quick estimator: PostgreSQL + Sample + MHist
- [ ] 跑 learned: Naru → MSCN → DeepDB → LW-NN → BayesNet
- [ ] 记录 Q-Error 表

---

## 1. 两个 Repo 的关系

- **`AreCELearnedYet/`** = 原版 `sfu-db/AreCELearnedYet`（VLDB 2021 paper code，最后维护停留在 2020 年）
- **`CoLSE_AreCELearnedYet/`** = `Lankadinee/AreCELearnedYet`（fork，22 commits ahead，2024-2025 维护）

Lankadinee 的修改全部在 `feature/fixed-dependancy-installation` 分支上，主要是**构建系统现代化**，不是研究方向上的修改。

---

## 2. 文件级别差异

### 2.1 lecarb/（源代码）—— 基本一致

`diff -rq AreCELearnedYet/lecarb CoLSE_AreCELearnedYet/lecarb` 输出：

```
Files .../manipulate_dataset.py differ
```

只有 **1 个文件**有差异：`lecarb/dataset/manipulate_dataset.py`
- 修改源：commit `0d2f765` "Enhance append_data function to include detailed logging and save combined datasets in both pickle and CSV formats"
- 影响：仅 `append_data` / `gen_appended_dataset` 增加 logging 和 csv 备份
- **对 Day 1 无影响**（不做 dataset appending）

另外 2 个 lecarb 内 commit（在 git log 中但不影响当前对比）：
- `bcfac33`: `deepdb.py` sample rate `1.0 → 0.01`（后又被 commit `07b36b0` 在 Justfile 里改成 0.3）
- `3a6f511`: `lecarb/__main__.py` 改一个 typo

### 2.2 dynamic-exp/、hyper-params.md、LICENSE、.pylintrc、.gitignore

全部 100% 一致。

### 2.3 配置文件 —— 关键差异

| 文件 | 原版 | Lankadinee | 备注 |
|------|------|-----------|------|
| `pyproject.toml` | poetry 1.x 格式，caret 约束 | PEP 621 + uv，`==` 精确锁 | 重写 |
| `pyproject.bk.toml` | 不存在 | 原 poetry 版备份 | Lankadinee 良习 |
| `.python-version` | 不存在 | `3.8` | uv/pyenv 标识 |
| `uv.lock` | 不存在 | 6053 行 | 完全可复现 |
| `export_env.sh` | 不存在 | 3 行：`PYTHONPATH=$(pwd)/lecarb` + `export $(grep -v '^#' .env \| xargs)` | 激活 venv 后 source |
| `Justfile` | 所有命令 `poetry run ...` | 改为 `uv run ...` + 新增 `wget-data` / `extract-data` / `download-data` | ~30 处替换 |
| `README.md` | `pip install poetry` + `just install-dependencies` | `curl ... uv install` + `source export_env.sh` + `just download-data` | 安装流程改写 |
| `.env` | DATABASE_URL=6667 | 同 | 一致 |

### 2.4 `pyproject.toml` 依赖差异详表

| 包 | 原版（poetry） | Lankadinee（uv） | 关键点 |
|----|---------------|------------------|-------|
| python | `^3.7` | `>=3.8` | 提升下限，与 `.python-version` 对齐 |
| pomegranate | `^0.13.4` | **`==0.13.4`** | caret 解到 0.13.5 会触发 Cython 编译失败 |
| Cython | `^0.29.21` | `==0.29.21` | 防止解到 3.x |
| ray | `^0.8.7` | `==0.8.7` | 老版只有窄范围 wheel |
| pyodbc | `^4.0.30` | `==4.0.30` | 同 |
| mysql-connector-python | `^8.0.21` | `==8.0.21` | 同 |
| tables | `3.5.1` | **`3.6.1`** | 3.6.1 cp38 wheel 更干净 |
| networkx | 未锁 | **`==2.8.8`** | 让 protobuf/ray 上下游一致 |
| protobuf | 未锁 | **`==3.20.3`** | ray 0.8.7 在 py3.8 上需要 |
| setuptools | 未锁 | **`>=75.3.2`** | uv 构建后端要求 |
| scikit-learn | `0.23.0` | `==0.23.0` | 同；但 uv resolver 不拽 `sklearn==0.0.post12` 那个 placeholder |
| 其它包 | caret/松约束 | 一律 `==` | 同 |
| dev-dependencies (mypy/black/pylint/ipython) | 有 | **被删** | 简化构建 |

---

## 3. 为什么 poetry 路线踩雷、uv 路线通畅

### Poetry 失败原因

1. **caret 约束让 resolver 升级到边缘版本**
   - `pomegranate = "^0.13.4"` 解到 0.13.5；0.13.5 在 Cython 0.29 上的 `.pyx` 编译失败
2. **传递依赖拖入已废弃包**
   - 某个老依赖把 `sklearn==0.0.post12`（已 deprecated 的 placeholder）拽进来，需要 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` 才肯装
3. **poetry 2.x 的 PEP 517 强制 source build**
   - 即使 PyPI 有 cp38 manylinux wheel，poetry 也可能去重新构建

### uv 优势

1. **`==` 精确锁绕过 caret 升级路径** —— 直接锁到有 wheel 的版本
2. **`uv.lock` 完全可复现** —— 任何机器装出同一棵树
3. **resolver 更智能** —— 优先用 wheel，少走 source build
4. **二进制 binary 形式，速度快** —— 比 poetry 快 10-100x
5. **不需要 venv 预设** —— `uv sync` 自动管理 venv 和 Python 版本

---

## 4. 迁移方案（Lankadinee 风格 → 原版 AreCELearnedYet/）

**约束**: 不动 `CoLSE_AreCELearnedYet/`；把构建系统现代化应用到 `AreCELearnedYet/`。

### 步骤
1. **备份原 poetry 配置**: `AreCELearnedYet/pyproject.toml` → `pyproject.bk.toml`
2. **复制 Lankadinee 的 5 个文件到原版**:
   - `pyproject.toml`（PEP 621 + uv 格式）
   - `.python-version`（`3.8`）
   - `uv.lock`（6053 行锁文件）
   - `export_env.sh`
   - `Justfile`（uv run 版本）
3. **保留 lecarb/ 原版源码**（仅 `manipulate_dataset.py` 与 Lankadinee 微差，且不影响 Day 1）
4. **安装 uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv`
5. **解压数据**: `cd AreCELearnedYet && tar -xzf <path>/data.tar.gz`（或借用 `CoLSE_AreCELearnedYet/data/` 软链接）
6. **`uv sync`** 装依赖（uv.lock 确保走 wheel 路径）
7. **激活 + 跑**:
   ```bash
   source .venv/bin/activate
   source export_env.sh
   just pkl2table census13 original  # 实际是 uv run python -m lecarb dataset table ...
   ```

### .env 端口对齐
当前 `.env` 中 `DATABASE_URL=postgres://card:card@localhost:6667/card`，但本机刚装的 Postgres 14 默认在 5432。两个选项：
- (A) 改 `.env` 端口 6667 → 5432（简单）
- (B) 改 Postgres 端口 5432 → 6667（保留原配置）

Day 1 跑 PostgreSQL 估计器前必须先解决。

---

## 5. 与 Week 3 PLAN 的对应

PLAN Step 0 检查项：
- [x] Clone `sfu-db/AreCELearnedYet`
- [x] 安装依赖（**通过 uv 路线，不走 poetry**）
- [x] 配置 `.env`
- [ ] Census 上跑通全流程: csv2pkl → table → workload gen → train → test → report
- [ ] 跑 quick estimator: PostgreSQL + Sample + MHist
- [ ] 跑 learned: Naru → MSCN → DeepDB → LW-NN → BayesNet
- [ ] 记录 Q-Error 表

Day 1 目标（[LOG.md Day 1 计划](UniMelb/Vault/wiki/in-progress/0_workplan/week-3/LOG.md#L205-L214)）：「ARELY 环境可用 + Census 上至少 3 个 estimator 有 Q-Error 数字」—— 用 MHist + Sample + (PostgreSQL or BayesNet) 即可达标。

---

## 6. 后续提示

- **CoLSE_AreCELearnedYet/ 用途**: 当 PLAN Step 1 进入 CoLSE 测试阶段时，这个 fork 可能就是 CoLSE 组改造过的版本（命名暗示），不要丢；但本次 Day 1 暂不动它。
- **uv 学习曲线**: `uv pip install X` / `uv run X` / `uv sync` / `uv add X` / `uv lock` —— 基本对齐 npm 的 mental model。
- **Justfile 命令**: 迁移后所有 `just <target>` 行为不变，底层从 `poetry run` 改成了 `uv run`。
