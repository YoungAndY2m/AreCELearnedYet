# AreCELearnedYet — 修改与配置日志

> 此文件记录该 repo 从 clone 后所有重要修改，供未来回溯参考。遵循 [[Desktop/CLAUDE.md]] 规范。

---

## 元数据

- **Fork 来源**: `YoungAndY2m/AreCELearnedYet`（origin），fork 自 `sfu-db/AreCELearnedYet`（upstream）
- **Clone 时间**: 2026-05-15
- **初始 commit**: `aa52da7` — "Update README.md"（原 sfu-db 最后 commit）
- **关联工作计划**: [[UniMelb/Vault/wiki/in-progress/0_workplan/week-3/LOG]], [[week-3/PLAN]], [[week-3/THOUGHT]], [[week-3/week-2-conclude]]
- **uv 配置来源**: 借鉴自 [Lankadinee/AreCELearnedYet](https://github.com/Lankadinee/AreCELearnedYet) 这个 2024 fork（22 commits ahead of sfu-db，把构建系统改为 uv + `==` 精确锁）。曾经在 `Desktop/CoLSE_AreCELearnedYet/` 有本地副本，已于 2026-05-16 删除（所需 5 个 uv 文件已 vendored 进本 repo，副本无再用途）

---

## 修改记录

> 最新在最上面。

### [2026-05-16] 清理 `CoLSE_AreCELearnedYet/` 残留引用（本地副本已删）

- **目的**: 用户删除了 `Desktop/CoLSE_AreCELearnedYet/`（Lankadinee fork 的本地副本）。迁移已完成，uv 文件全部 vendored 进本 repo，副本无再用途。所有跨文件的引用要么收紧成"历史 reference"，要么删
- **改动**:
  - **`Desktop/CLAUDE.md`** §0 目录树：移除 `CoLSE_AreCELearnedYet/` 节点，加 `AllModels/` 节点 + 删除时间历史备注
  - **本文件**：合并原 §1（两个 repo 关系）+ §2（文件级差异）→ 一段 §1 brief reference；§3-§5 渐次降级 + 清理 `CoLSE_AreCELearnedYet/` 字样；末尾加澄清"PLAN Step 1 的 CoLSE 走 `AllModels/CoLSE/` 不是这个 fork"
  - **`LOG_EXEC.md`** §1.6：把 "Lankadinee 默认 6667" 改为 "上游 fork 默认 6667"
  - **`Desktop/SESSION_HANDOFF.md`**：移除"CoLSE_AreCELearnedYet read-only 副本"bullet，加历史备注；memory 文件描述同步
  - **`AllModels/MSCN/LOG.md`** "与其它 repo 的关系"：删 `CoLSE_AreCELearnedYet/` bullet，修 `AreCELearnedYet/` 链接的相对路径（多了一层 `../`）
  - **memory**：
    - `MEMORY.md` 索引条目改为 "Lankadinee fork (historical) — ... local copy deleted"
    - `reference_colse_arely_fork.md` 重写为历史 reference（GitHub URL 仍有效）
    - `reference_arely_env.md` 把"借用本地副本"改为"借鉴 GitHub fork"
- **保留**:
  - LOG.md §1 仍记录"uv 配置来源 = Lankadinee fork"+ 关键移植点 + GitHub URL（方便未来 lock 文件坏掉时回溯）
  - 2026-05-15 那条 changelog "Poetry → uv 构建系统迁移（vendored from Lankadinee fork）"未动，是历史事实
- **影响**: 纯文档；无代码改动

### [2026-05-16] LOG_STRUCTURE §6.0.1 新增 estimator 代码出处对照表

- **目的**: 之前 §6 只有 paper taxonomy（按 Methodology/Input/Model 分类），但没有回答"这些代码是 ARELY 改编自上游还是从论文自实现"。研究方向上要决定改哪个 estimator 最省力，必须先看 attribution
- **新增内容**: LOG_STRUCTURE.md §6.0.1
  - 15 行表格覆盖所有 estimator：5 个学习方法（Naru/MSCN/DeepDB/LW-NN/LW-XGB） + 8 个传统/native（BayesNet/QuickSel/KDE-FB/MHIST/Sample-A/B/Postgres/MySQL/DBMS-A） + 2 个 paper-only（DQM-D/Q）
  - 每行列出：类型 / paper 引用 / 上游 GitHub / ARELY 内状态 / 详细位置
  - 末尾"实践启示"：哪些 estimator 改造最省力（LW + MHIST 无跨 repo 同步压力）
- **依据**:
  - [README §Code References](README.md)（作者明确列的上游）
  - 各子目录 `README.md`（`lecarb/estimator/{naru,mscn,deepdb,lw}/README.md`）
  - Paper §2 Related Work + §3 Methodology + Table 1 taxonomy
  - Paper Table 3+ 结果表（验证 DQM-D/Q 不在 benchmark 内）
- **影响**: 纯文档；无代码改动

### [2026-05-16] 修正 LOG_EXEC §2.3 learned estimator 参数 → 对齐论文 Selected Models

- **目的**: 早期写 `LOG_EXEC.md §2.3` 时把 Justfile target 的 default 值当成了论文配置，但 default 是 dmv / forest 量级，census13 的论文配置在 [`hyper-params.md` "Selected Models"](hyper-params.md#L41-L160) 里单独覆写过。再跑会产出非可比的 q-error，必须修正。
- **错配对照**:
  | Estimator | 错配（旧 LOG_EXEC = Justfile default） | 论文 Selected（已改成此值） |
  |-----------|--------------------------------------|--------------------------|
  | Naru      | `layers=4 fc=32 embed=4`              | `layers=4 fc=16 embed=8` |
  | MSCN      | `samples=1000 hid=16 ep=200 bs=1024`  | `samples=500 hid=8 ep=100 bs=256` |
  | DeepDB    | `rdc=0.3`                             | `rdc=0.4`                |
  | LW-NN     | `hid=128_64_32 train=10000 bs=32`     | `hid=64_64_64 train=100000 bs=128` |
- **未受影响**:
  - BayesNet `samples=200 discretize=100 parallelism=50` — `hyper-params.md` 未单列 BayesNet，等价于 Justfile default 即论文配置 ✅
  - Sample / MHist / PostgreSQL — 同上，default 即论文 ✅
- **修改文件**:
  - `LOG_EXEC.md` §2.3 — 4 个 learned estimator 命令重写到论文配置，并预填了 test 阶段需要的 model 文件名（参考 `hyper-params.md` 命名）
  - `LOG_EXEC.md` 新增 §2.5 — 一键 tmux 4 window 并行训练 + 测试脚本（本机 16 核 / 62 GB 足够同时跑 Naru/MSCN/DeepDB/LW-NN）
- **影响**:
  - 已用旧配置训出的 model 文件（如有）应视为废弃，重训即可（census13 数据小，全部 4 个重训预计 30-60 min）
  - BayesNet `samples=200` 那条不受影响，可独立继续跑
- **来源依据**: [hyper-params.md "Selected Models · census"](hyper-params.md#L41-L160)（与原 sfu-db 仓库 100% 一致，本次 diff 中此文件未动）

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

## 1. uv 配置参考来源（仅作 reference）

本 repo 的 uv 配置（`pyproject.toml` / `uv.lock` / `.python-version` / `export_env.sh` / `Justfile`）最早 vendored 自 [Lankadinee/AreCELearnedYet](https://github.com/Lankadinee/AreCELearnedYet) —— 是 `sfu-db/AreCELearnedYet` 的 2024 fork，主要做了**构建系统现代化**（poetry → uv，所有依赖 `==` 精确锁，新增 `uv.lock`，Justfile 改 `uv run`）。修改在 `feature/fixed-dependancy-installation` 分支。

**关键移植点**（已固化到本 repo 现有文件，无需再去原 fork 翻）：
- `pomegranate==0.13.4`（caret 解到 0.13.5 会触发 Cython 编译失败）
- `Cython==0.29.21` / `ray==0.8.7` / `pyodbc==4.0.30` / `tables==3.6.1`
- `networkx==2.8.8` / `protobuf==3.20.3`（让 ray / protobuf 上下游一致）
- `scikit-learn==0.23.0` —— 但 `uv.lock` 仍锁了 `sklearn==0.0.post12` placeholder，所以**首次 sync 必须带 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True`**
- lecarb/ 源码与 sfu-db 上游基本一致（仅 `lecarb/dataset/manipulate_dataset.py` 加了 logging + 双备份，对 Day 1 无影响）

> 历史本地副本 `Desktop/CoLSE_AreCELearnedYet/` 已于 2026-05-16 删除，迁移已完成无再用途。需要回溯具体差异时去上面那个 GitHub URL。

---

## 2. 为什么 poetry 路线踩雷、uv 路线通畅

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

## 3. 迁移历史（已完成；步骤留作参考）

> 2026-05-15 完成，无需重跑。仅作 reference，方便未来在别的 fork 上复用此模式。

### 步骤
1. **备份原 poetry 配置**: `AreCELearnedYet/pyproject.toml` → `pyproject.bk.toml`
2. **从 [Lankadinee fork](https://github.com/Lankadinee/AreCELearnedYet) 复制 5 个文件**:
   - `pyproject.toml`（PEP 621 + uv 格式）
   - `.python-version`（`3.8`）
   - `uv.lock`（6053 行锁文件）
   - `export_env.sh`
   - `Justfile`（uv run 版本）
3. **保留 lecarb/ 原版源码**（仅 `manipulate_dataset.py` 微差，不影响 Day 1）
4. **安装 uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv`
5. **解压数据**: `cd AreCELearnedYet && tar -xzf <path>/data.tar.gz`
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

## 4. 与 Week 3 PLAN 的对应

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

## 5. 后续提示

- **uv 学习曲线**: `uv pip install X` / `uv run X` / `uv sync` / `uv add X` / `uv lock` —— 基本对齐 npm 的 mental model。
- **Justfile 命令**: 迁移后所有 `just <target>` 行为不变，底层从 `poetry run` 改成了 `uv run`。
- **CoLSE 实验路径**: PLAN Step 1 的 CoLSE 实验走 `Desktop/AllModels/CoLSE/` 这个独立 repo（Rathuwadu et al. copula-based CE），与历史 `CoLSE_AreCELearnedYet/` 完全无关——后者只是 Lankadinee fork 的本地副本，已删除。
