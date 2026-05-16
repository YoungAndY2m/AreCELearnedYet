# AreCELearnedYet — 代码结构说明（LOG_STRUCTURE）

> 此文件是 repo 代码结构的导览，配合 paper **"Are We Ready For Learned Cardinality Estimation?"** (Wang et al., VLDB 2021, [PDF in repo root](./Wang%20等%20-%202021%20-%20Are%20we%20ready%20for%20learned%20cardinality%20estimation.pdf)) 一起读。
>
> **阅读顺序建议**:
> 1. 先读 paper §1-§3（背景 + 方法分类法 + 实验框架）
> 2. 读本文件 §1-§3（看 repo 整体怎么映射 paper）
> 3. 跳到 paper §4-§5 + 本文 §6（每个 estimator 实现细节）
> 4. 想扩展功能 → 本文 §7（如何加新 estimator）

---

## 0. 元数据

- **Paper**: Xiaoying Wang, Changbo Qu, Weiyuan Wu, Jiannan Wang, Qingqing Zhou. *Are We Ready For Learned Cardinality Estimation?* PVLDB 14(9), 2021. [arxiv](https://arxiv.org/abs/2012.06743) | [vldb](http://www.vldb.org/pvldb/vol14/p1640-wang.pdf)
- **核心问题**: 单表 cardinality estimation。给定关系 `R` 上含 `d` 个谓词的查询 `SELECT COUNT(*) FROM R WHERE θ_1 AND ... AND θ_d`，估计满足的元组数。
- **Paper 4 大贡献**:
  1. Static 环境：学习方法 vs 9 个传统方法，统一 workload + 4 数据集
  2. Dynamic 环境（数据更新）：学习方法是否跟得上
  3. 学习方法什么时候出错（按 correlation/skewness/domain size 变化）
  4. 提出两个未来方向（控成本 + 提可信度）
- **Methods 覆盖**: 5 个学习方法（Naru, MSCN, DeepDB, LW-XGB, LW-NN）+ 9 个传统方法（Postgres, MySQL, DBMS-A, Sample-A/B, MHIST, QuickSel, Bayes, KDE-FB）

---

## 1. Repo 顶层布局

```
AreCELearnedYet/
├── Justfile                ← 所有命令入口（pkl2table / train-X / test-X / report-error）
├── pyproject.toml          ← uv 依赖配置（PEP 621 + ==）
├── pyproject.bk.toml       ← 原 poetry 配置备份
├── uv.lock                 ← 完全锁定依赖（6053 行）
├── .python-version         ← Python 3.8
├── .env                    ← DATA_ROOT/OUTPUT_ROOT/DATABASE_URL 等
├── export_env.sh           ← 把 .env 变量 export 出来（lecarb 启动前必 source）
├── README.md
├── hyper-params.md         ← 论文用的超参数实测值（每个数据集 × 每个方法）
├── LOG.md                  ← 修改与配置日志（changelog）
├── LOG_EXEC.md             ← 执行命令手册（runbook）
├── LOG_STRUCTURE.md        ← 本文件（代码结构导览）
├── Wang等-2021-paper.pdf   ← 原 paper PDF
├── data/                   ← 数据集（git ignored），结构: {dataset}/{version}.csv|.pkl + workload/
│   ├── census13/           ← 49K 行 (US Census 13 列 1994)
│   ├── forest10/           ← 581K 行 (UCI Forest Cover 10 列)
│   ├── power7/             ← 2.1M 行 (UCI Power Consumption 7 列)
│   └── dmv11/              ← 11.6M 行 (NYC DMV 11 列)
├── output/                 ← 模型 + 结果产物（git ignored）
│   ├── model/{dataset}/    ← 训练好的模型文件
│   └── result/{dataset}/   ← 测试 CSV（每个 query 一行：est_card, dur_ms, q_error）
├── dynamic-exp/            ← Dynamic 实验脚本（paper §5）
└── lecarb/                 ← 主代码包（详见下面）
    ├── __main__.py         ← CLI 入口（docopt）
    ├── constants.py        ← 路径 + DB 凭据（环境变量驱动）
    ├── dtypes.py           ← pandas/numpy 类型判断
    ├── dataset/            ← 数据加载/合成/扰动
    ├── workload/           ← Query 生成 + ground truth 标注
    └── estimator/          ← 10+ 个 CE 方法（重点）
```

---

## 2. 端到端 Pipeline

```
[原始 CSV]
   │ just csv2pkl
   ▼
[原始 PKL DataFrame]                    data/{ds}/{ver}.pkl
   │ just pkl2table
   ▼
[Table 对象]                             columns + vocab + stats
   │
   ├──→ workload gen ──→ [Queryset]    随机 d 个谓词的合成查询
   │                        │
   │                        ▼
   │                    workload label
   │                        │
   │                        ▼
   │                    [Labels]        Oracle scan 算精确 cardinality
   │
   └──→ estimator train ──→ [Model]    （仅 learned 方法需要）
                              │
                              ▼
                          estimator test
                              │
                              ▼
                          [Result CSV]   每 query: est_card, dur_ms, q_error
                              │
                              ▼
                          report-error
                              │
                              ▼
                          [Q-Error 统计]  max / 99th / 95th / median / mean / gmean
```

每个 `[ ]` 节点对应 `data/` 或 `output/` 下的一个落盘文件。

---

## 3. lecarb 模块映射到 paper

| Paper 章节 | 对应 lecarb 子目录/文件 | 责任 |
|-----------|------------------------|------|
| §2.1 Problem Statement (单表 + 合取谓词) | `workload/workload.py` 中的 `Query` namedtuple | 抽象 |
| §2.2-§2.4 Taxonomy（Regression vs Joint Dist） | `estimator/` 下的子目录划分 | 实现 |
| §2.3 Regression workflow（Fig 1a） | `estimator/{mscn,lw}/` | 实现 |
| §2.4 Joint Distribution workflow（Fig 1b） | `estimator/{naru,deepdb,bayesnet}.py` | 实现 |
| §3 Experimental Setup (q-error 指标) | `estimator/utils.py::qerror()` `evaluate()` | 评测 |
| §3 Workload generator（query center + range width，d∈[1,\|D\|]） | `workload/generator.py` 中 asf_/csf_/wsf_ | 实现 |
| §3 Dataset (Census/Forest/Power/DMV) | `data/{ds}/{ver}.csv` + Justfile `*2postgres` 加载 | 数据 |
| §4 Static environment 实验 | Justfile `train-*` + `test-*` + `report-error` | 跑实验 |
| §5 Dynamic environment 实验 | `dynamic-exp/` + Justfile `update-*` `dynamic-*` | 跑实验 |
| §6 When learned methods go wrong | `dataset/manipulate_dataset.py` + `dataset/gen_dataset.py` | 合成数据扰动 |

---

## 4. lecarb/ 文件级说明（CORE）

### 4.1 lecarb/__main__.py — CLI Dispatcher

**职责**: docopt 解析 + 路由到具体函数。

**子命令**（与 Justfile target 对应）:
```
dataset    table|gen|update|dump    数据准备
workload   gen|label|update-label|merge|dump|quicksel   workload 准备
train      -e {naru|mscn|deepdb|lw_nn|lw_tree}    训练（仅 learned）
test       -e {naru|mscn|deepdb|lw_nn|lw_tree|sample|postgres|mysql|mhist|bayesnet|kde}    测试
update-train  -e {naru|deepdb}    增量训练
report     --params "{'file': X}"    汇总 q-error
report-dynamic                    dynamic 实验报告
```

**关键参数**: `--seed`, `--dataset`, `--dataset-version`, `--workload`, `--estimator`, `--params <JSON>`, `--sizelimit`（模型大小相对数据的预算，默认 0.015 = 1.5%）。

### 4.2 lecarb/constants.py

环境变量映射到全局 Path/凭据。**必须先 `source export_env.sh` 否则 `KeyError: 'DATA_ROOT'`**（见 [LOG_EXEC.md §4](LOG_EXEC.md)）。导出：

- `DATA_ROOT` / `OUTPUT_ROOT` / `MODEL_ROOT=OUTPUT_ROOT/model` / `RESULT_ROOT=OUTPUT_ROOT/result` / `LOG_ROOT`
- `DATABASE_URL`（Postgres）/ `KDE_DATABASE_URL` / `MYSQL_*`
- `DEVICE`（auto cuda/cpu via torch）
- `NUM_THREADS`、`PKL_PROTO=4`、`VALID_NUM_DATA_DRIVEN=100`

### 4.3 lecarb/dtypes.py

`is_categorical(dtype)` / `is_numerical(dtype)` / `is_discrete(dtype)`：判 pandas/numpy 类型。在 `workload/generator.py` 决定用 equality 还是 range 时调用。

---

## 5. lecarb/ 文件级说明（DATA + WORKLOAD）

### 5.1 lecarb/dataset/dataset.py — **核心抽象**

最重要的两个类，几乎所有 estimator 都吃 `Table`：

#### `Column`
- **Fields**: `name`, `dtype`, `vocab`（排序后唯一值数组）, `vocab_size`, `minval`, `maxval`, `has_nan`
- **Methods**:
  - `discretize(data) -> int32`：把值映射到 `[0, vocab_size)` 的 bin ID（categorical 必经；range 估计也常用）
  - `normalize(data) -> float32`：把数值列归一到 `[0, 1]`（learned 方法的输入预处理）

#### `Table`
- **Fields**: `data: pd.DataFrame`, `row_num`, `col_num`, `data_size_mb`, `columns: OrderedDict[str→Column]`
- **Methods**:
  - `parse_columns()`：构造 `columns` 字典
  - `get_minmax_dict()`：每列 (min, max) 元组
  - `normalize(scale=1)`：返回归一化的 DataFrame（learned 方法常用）
  - `digitalize()`：返回 discretize 后的 int DataFrame
  - `get_max_muteinfo_order()`：互信息排序列（Naru 用，paper §2.4）
- **I/O**: 模块顶层 `load_table(dataset, version)` 反序列化 `data/{ds}/{ver}-table.pkl`

### 5.2 lecarb/dataset/gen_dataset.py — 合成数据（Paper §6）

`generate_dataset(seed, dataset, version, params, overwrite)`：合成 2 列数据测试学习方法的 robustness。

**Params**: `row_num`, `col_num`（强制 =2）, `dom`（domain 大小）, `corr`（列相关性 ∈ [0,1]）, `skew`（Pareto 形状参数）。

**输出**: `data/{dataset}/{version}.csv` + `.pkl`。Justfile `data-gen 1.0 1.0 1000` 调用此。

### 5.3 lecarb/dataset/manipulate_dataset.py — 数据扰动（Paper §5 dynamic）

为 dynamic 实验生成"更新后"的数据：
- `get_random_data()` — 每列独立打乱（**independence**，Spearman ρ=0）
- `get_sorted_data()` — 每列排序后用同样 row index（**max correlation**，Spearman ρ→1）
- `get_skew_data()` — 按 tuple-frequency 重采样（**max skew**）
- `append_data(...)` — 把 update 版本的尾部 20% 拼到 original

Justfile target: `append-data-{ind|cor|skew}`。

### 5.4 lecarb/workload/workload.py — Query/Label 抽象

**类**:
- `Query(NamedTuple)`: `predicates: Dict[col_name → Optional[(op, val)]]`, `ncols: int`（实际 filter 列数）
- `Label(NamedTuple)`: `cardinality: int`, `selectivity: float`（= card / row_num）

**Query → 其他格式**（每个 estimator 拿不同 view）:
- `query_2_triple(query, with_none, split_range)` → `(cols, ops, vals)` 三元组列表
- `query_2_sql(query, table, aggregate, split, dbms)` → SQL 字符串（送 Postgres/MySQL）
- `query_2_vector(query, table, upper)` → 归一化向量 [0,1]（learned 方法）
- `query_2_quicksel_vector(...)` → QuickSel 输入格式

**I/O**: `dump_queryset()` / `load_queryset()` / `dump_labels()` / `load_labels()`（pickle 到 `data/{ds}/workload/{name}*.pkl`）。

### 5.5 lecarb/workload/generator.py — **Paper §3 Workload Generator**

> Paper 把 query 看成 d 维空间的超矩形：`query_center` + `range_width`。 generator 拆成 3 类函数。

**Attribute Selection Functions (asf_)** — 选哪些列加谓词：
- `asf_pred_number(table, params)` — params['number'] 个谓词，从 whitelist/blacklist 抽
- `asf_comb(table, params)` — 固定列组合 params['comb']
- `asf_naru(table, params)` — Naru 论文的 5-12 列随机

**Center Selection Functions (csf_)** — 选谓词中心值（paper Fig 3 + §3 "Query Center"）：
- `csf_domain(table, attrs, params)` — **从真实 tuple 抽**（Paper §3 ①，保证 card>0）
- `csf_distribution(table, attrs, params)` — 分布抽样（缓存的 row）
- `csf_ood(table, attrs, params)` — **out-of-distribution**：每列独立随机（Paper §3 ②，可能 card=0）
- `csf_vocab_ood(table, attrs, params)` — 从 column vocab 抽
- `csf_domain_ood(table, attrs, params)` — uniform 抽整个 domain
- `csf_naru(table, attrs, params)` — Naru 风格

**Width Selection Functions (wsf_)** — 选 range 宽度（paper §3 "Range Width"）：
- `wsf_uniform(table, attrs, centers, params)` — Paper §3 ❶：均匀 [0, size_i]
- `wsf_exponential(table, attrs, centers, params)` — Paper §3 ❷：指数衰减（λ=10/size_i 默认）
- `wsf_equal(table, attrs, centers, params)` — 全 equality
- `wsf_naru(table, attrs, centers, params)` — Naru：>=, <=, =

**核心类**:
- `QueryGenerator`：组合 asf × csf × wsf，按概率分布抽。`generate() → Query`。

**Paper §3 默认配置**（"Our Workload"，Table 2 最后一行）：
- `asf_pred_number`（d ∈ [1, |D|]）
- 90% `csf_domain` + 10% `csf_ood`
- 50% `wsf_uniform` + 50% `wsf_exponential`
- equality 和 range 都覆盖；考虑 OOD

### 5.6 lecarb/workload/gen_workload.py

编排：`generate_workload(seed, dataset, version, name, no_label, old_version, win_ratio, params)`：
- 实例化 `QueryGenerator`，按 params 配置 asf/csf/wsf 概率
- 生成 train/valid/test 三段 query（每段 size 由 params['number'] 给）
- 如有 `old_version + win_ratio` → 把 query center 集中在更新数据的尾部
- 落到 `data/{ds}/workload/{name}.pkl`

### 5.7 lecarb/workload/gen_label.py — Ground Truth Oracle

- `generate_labels(dataset, version, workload)`：load Table + queryset → 对每个 query 调 `Oracle.query()` 全扫一遍计算精确 card → dump 到 `data/{ds}/workload/{name}-{version}-label.pkl`
- `update_labels(...)`：用 sampling 估算（dynamic 实验里 dataset 频繁更新时用，避免全扫）

### 5.8 lecarb/workload/merge_workload.py / dump_quicksel.py

- `merge_workload`：把 `{name}_0..{name}_9` 10 个并行生成的 workload 合并成一个（大数据集用）
- `dump_quicksel`：导出 QuickSel 估计器需要的特殊格式（离散 bin 向量 + selectivity）

---

## 6. lecarb/ 文件级说明（ESTIMATOR — 重点）

### 6.0 Paper 分类总图（Table 1 + Fig 1）

| Method | Methodology | Input | Model | File |
|--------|------------|-------|-------|------|
| **MSCN** | Regression | Query+Data | Neural Network (set conv) | `mscn/mscn.py` |
| **LW-XGB** | Regression | Query+Data | Gradient Boosted Tree | `lw/lw_tree.py` |
| **LW-NN** | Regression | Query+Data | Neural Network (MLP) | `lw/lw_nn.py` |
| **Naru** | Joint Distribution | Data | Autoregressive (MADE/Transformer) | `naru/naru.py` |
| **DeepDB** | Joint Distribution | Data | Sum Product Network | `deepdb/deepdb.py` |
| **BayesNet** | Joint Distribution | Data | Bayesian Network (pomegranate/pgmpy) | `bayesnet.py` |

> Regression 思路（Fig 1a）：query → featurize → learned regressor → cardinality。需要 (query, true_card) 训练对。
> Joint Distribution 思路（Fig 1b）：data → learned density model；inference 时 query 转一组对模型的概率请求。不需要 query training data。

### 6.0.1 代码出处与改编状态（Attribution）

> 此表回答："lecarb 里这堆 estimator 是 ARELY 团队改编自上游，还是从论文自己实现？"
> 来源：[README §"Code References"](README.md#code-references) + 各子目录的 `README.md`（`lecarb/estimator/{naru,mscn,deepdb,lw}/README.md`）+ paper §2-§3。

| Estimator | 类型 | Paper 出处 | 上游公开代码 | ARELY 内状态 | 详细位置 |
|-----------|------|-----------|------------|------------|---------|
| **Naru** | Learned (autoregressive) | Yang VLDB 2020 [95] | [naru-project/naru](https://github.com/naru-project/naru) | 改编自上游 ✅ | §6.5 `naru/` |
| **BayesNet** | Learned (PGM + progressive sampling) | Tzoumas 2013 [13]，**实现 bundled 在 naru repo 内** | [naru-project/naru](https://github.com/naru-project/naru) | 改编自 naru repo ✅ | §6.5 `bayesnet.py` |
| **MSCN** | Learned (set-conv / Deep Sets) | Kipf CIDR 2019 [34] | [andreaskipf/learnedcardinalities](https://github.com/andreaskipf/learnedcardinalities) | 改编自上游 ✅ | §6.4 `mscn/` |
| **DeepDB** | Learned (SPN) | Hilprecht VLDB 2020 [30] | [DataManagementLab/deepdb-public](https://github.com/DataManagementLab/deepdb-public) | 改编自上游 ✅ | §6.5 `deepdb/` |
| **LW-NN** | Learned (regression on CE features) | Dutt VLDB 2019 [18] | **❌ 无公开代码** | **ARELY 团队从论文自实现** ⚠️ | §6.4 `lw/lw_nn.py` |
| **LW-XGB** | Learned (xgboost on CE features) | Dutt VLDB 2019 [18] | **❌ 无公开代码** | **同上，自实现** ⚠️ | §6.4 `lw/lw_tree.py` |
| **QuickSel** | Traditional (query-driven uniform mix) | Park SIGMOD 2020 [66] | [illinoisdata/quicksel](https://github.com/illinoisdata/quicksel)（Java） | **sfu-db 自己 fork** → [sfu-db/quicksel](https://github.com/sfu-db/quicksel)（加 test class） | **不在 lecarb/** —— 是独立 Java repo |
| **KDE-FB** | Traditional (KDE + query feedback) | Kiefer SIGMOD 2017 | [martinkiefer/feedback-kde](https://github.com/martinkiefer/feedback-kde) | **sfu-db 自己 fork** → [sfu-db/feedback-kde](https://github.com/sfu-db/feedback-kde)（列数限制 10→15） | §6.3 `feedback_kde.py`（wrapper），实际计算走 Postgres KDE 扩展（外部 repo） |
| **MHIST (MHIST-2)** | Traditional (multi-dim histogram, Maxdiff) | Poosala SIGMOD 1996 [73] | **❌ 无公开代码** | **ARELY 团队从论文自实现** ⚠️ | §6.3 `mhist.py` |
| **Sample-A / Sample-B** | Traditional baseline | classic | n/a | 原创（uniform random sampling，两个 ratio 变种） | §6.3 `sample.py` |
| **Postgres** | Native DB estimator | n/a | n/a | wrapper（调 `EXPLAIN`） | §6.3 `postgres.py` |
| **MySQL** | Native DB estimator | n/a | n/a | wrapper | §6.3 `mysql.py` |
| **DBMS-A** | Native（匿名商业 DB） | n/a | n/a | **不在此 repo 代码内** —— paper 外部跑的，作者匿名化未透露 | n/a |
| ~~DQM-D~~ | Learned (autoregressive) | Hasan SIGMOD 2020 [28] | ❌ 无公开代码 | **❌ 仅 paper §2-§3 提及**（与 Naru "propose similar ideas"），无实现 | n/a |
| ~~DQM-Q~~ | Learned (regression) | Hasan SIGMOD 2020 [28] | ❌ 无公开代码 | **❌ 仅 paper §2-§3 提及**，无实现，结果表 Table 3+ 不出现 | n/a |

**总结**:
- **改编自上游（5）**：Naru、BayesNet、MSCN、DeepDB、KDE-FB（KDE-FB 的核心计算在 sfu-db fork 的 PostgreSQL 扩展里）
- **sfu-db 自己 fork 改造（2）**：QuickSel（加 test class）、KDE-FB（扩列数限制）
- **从论文自实现（3）**：LW-NN、LW-XGB、MHIST
- **wrapper 原创（4）**：Postgres、MySQL、Sample-A/B
- **paper 提及但没实现（2）**：DQM-D（被 Naru 涵盖）、DQM-Q（无公开代码）
- **不在此 repo（1）**：DBMS-A（匿名商业 DB，外部跑）

**实践启示**:
1. **想深度改 Naru/MSCN/DeepDB/BayesNet** → 先去上游 repo 理解原始版本，再回 lecarb 看 ARELY 的改造点（典型改造：去掉 join，专做单表；去除 GPU 强依赖；统一 `query()` 接口）
2. **想改 LW-NN / LW-XGB / MHIST** → 直接改 lecarb 即可，没有跨 repo 同步压力 —— 这是做"semantic injection / 新 estimator 实验"最低摩擦的入口
3. **想跑 QuickSel / KDE-FB** → 单独 clone `sfu-db/quicksel` 或 `sfu-db/feedback-kde`，**不在 lecarb 内**；KDE-FB 还需要装他们 fork 的 PostgreSQL 扩展
4. **想加 DQM-D 实现** → 因为 paper §3 明说 "Naru 和 DQM-D propose similar ideas"，可基于 `lecarb/estimator/naru/` 改 featurization（一阶离散化 + 训练 query workload augment）

---

### 6.1 lecarb/estimator/estimator.py — 抽象基类

**`class Estimator`**:
- `__init__(self, table, **kwargs)`
- `query(self, query: Query) -> (est_cardinality: int, duration_ms: float)`：**所有 estimator 必须实现的接口**

**`class Oracle(Estimator)`**: 真值版本，全扫表算 card。用于生成 ground truth label。底层用 NumPy bitmap 操作 + `OPS` 字典（`=`, `<`, `>`, `<=`, `>=`, `[a,b]` 6 种 op）。

### 6.2 lecarb/estimator/utils.py — 统一测试循环（关键）

**所有 estimator 测试都过这里**：
- `qerror(est_card, true_card)` → `max(est/true, true/est)`（paper §3 定义）
- `evaluate(preds, labels, total_rows)` → `{max, 99th, 95th, 90th, median, mean, gmean, rms}` 统计字典
- `run_test(dataset, version, workload, estimator, overwrite, lazy=False, lw_vec=False, query_async=False)`:
  - 加载 test queryset + labels
  - 循环 `estimator.query(q) for q in queries`
  - 写 `output/result/{ds}/test-{estimator_name}-...csv`（每行 `card,gt,sel,gt_sel,err,dur_ms`）
  - 同时打印 `evaluate()` 统计

**lazy 模式**：训练版本和测试版本不同时，按 row_num 比例缩放预测 card（dynamic 实验）。

---

### 6.3 BASELINE（无需训练）

| 文件 | 类 | 入参 params | 算法要点 | Paper 章节 |
|------|-----|------------|---------|-----------|
| `postgres.py` | `Postgres` | `stat_target`（histogram buckets，默认 10000） | `EXPLAIN` 查 planner 估计 | §3 Trad: Postgres |
| `mysql.py` | `MySQL` | `bucket`（默认 1024） | MySQL 直方图 | §3 Trad: MySQL |
| `sample.py` | `Sampling` | `ratio`（默认 0.01） | 简单随机采样 → 按比例放大 card | §3 Trad: Sample-A |
| `mhist.py` | `Partition` | `num_bins`（默认 30000） | **MaxDiff 多维直方图**：递归找方差大的列切分，直到 bin 数耗尽 | §3 Trad: MHIST |
| `feedback_kde.py` | `FeedbackKDE` | `ratio`, `train_num` | Postgres KDE 扩展 + query feedback bandwidth 调优 | §3 Trad: KDE-FB |
| `bayesnet.py` | `BayesianNetworkWorker` | `samples`, `discretize`, `parallelism` | Pomegranate BN，**progressive sampling 推断**（Naru 论文比较的 baseline） | §3 Trad: Bayes |

> ⚠️ MHIST 慢：30000 bins × 13 列 × 48K 行的递归切分实测 5-10min（见 LOG.md pitfall #6）。

---

### 6.4 LEARNED REGRESSION（query+data → feature → model）

#### lecarb/estimator/lw/ — Lightweight (Dutt et al. 2019)

**文件**:
- `common.py`：`encode_query(table, query, pg_est)` → 特征向量
  - **Range features**: 每列归一后的 [lo, hi]
  - **CE features**: AVI / EBO / MinSel（基于 Postgres 单列估计 + 独立性假设）
- `model.py`：`LWNNModel(input_len, hid_units)` — 几层 ReLU FC
- `lw_nn.py`：
  - `train_lw_nn(seed, dataset, version, workload, params, sizelimit)`：Adam ~500 epochs
  - Params: `hid_units`（e.g. `'128_64_32'`）, `bins`（200）, `train_num`（10000）, `bs`（32）, `lr`（0.001）
  - **Loss**: MSE on `log(card+1)`（paper §2.3：等价于 q-error 几何均值，大误差更重）
- `lw_tree.py`：XGBoost regressor，`train_lw_tree(...)`. Params: `trees`（16）, `bins`, `train_num`.

**I/O**: `output/model/{ds}/{ver}-{workload}-lw{nn|tree}-*.pkl`

**Paper §2.3 摘录**: "LW-NN 用 NN, LW-XGB 用 GBT。两者最小化 log-transformed label 的 MSE（= q-error 几何均值），所以对大 q-error 更敏感。"

#### lecarb/estimator/mscn/ — Multi-Set CNN (Kipf et al. 2019)

**文件**:
- `model.py`：`SetConv` 模块
  - Sample set encoder：MLP(per-tuple) → sum-pool
  - Predicate set encoder：MLP(per-predicate) → sum-pool
  - 拼接后过两层 NN → sigmoid 输出（归一化 card）
- `mscn.py`：
  - `train_mscn(...)`：Adam 200 epochs，特殊 q-error loss（不是 MSE）
  - Params: `num_samples`（1000，sample size 进 NN）, `hid_units`（256）, `bs`（1024）, `train_num`（100000）, `epochs`（200）
  - 训练数据增强：每个 query 附带 sample table 的 bitmap（每位表示对应 tuple 是否满足该 predicate）

**I/O**: `output/model/{ds}/{ver}-mscn-*.pt`

**Paper §2.3 摘录**: "MSCN 在训练数据里把 query 喂模型时，附带一个 materialized sample 的 bitmap，每位标记 sample 中第 i tuple 是否过 predicate。这增强已被证明显著提性能（[34, 95]）。"

---

### 6.5 LEARNED JOINT DISTRIBUTION（data → density → 查询时推断）

#### lecarb/estimator/naru/ — Naru (Yang et al. 2019)

**文件**:
- `made.py`：MADE 网络（Masked Autoencoder for Density Estimation）
  - `MADE(nin, hidden_sizes, nout, input_bins, ...)`：mask 强制 autoregressive `P(A_i | A_1...A_{i-1})`
  - 支持多 ordering、residual、embedding
- `transformer.py`：可选 Transformer 变体（若 `params['heads'] > 0`）
- `naru.py`：
  - `train_naru(seed, dataset, version, workload, params, sizelimit)`：
    - 训练数据：原始 `Table.digitalize()` 后的离散 tuple
    - **Loss**: NLL（per-column conditional 累加）
    - Params: `layers`（4）, `fc_hiddens`（128）, `embed_size`（64）, `input_encoding`（'embed'）, `output_encoding`（'embed'）, `bs`（2048）, `epochs`（20）, `warmups`（LR warmup steps）, `residual`（True）, `num_orderings`（训练几组列序）, `column_masking`（True 时支持通配符）
  - `class Naru(Estimator)`：
    - `query(query, return_probs=False)`：**Progressive Sampling 推断**（paper §2.4）
      - 按列序逐列采样：满足约束的取值用 inverse CDF 选 + 边算概率
      - 输出每个 sample 的概率乘积 → 均值 × `row_num` = cardinality
    - `_sample_n(num_samples, ordering, columns, ops, vals, ...)`：核心采样函数
  - `test_naru(...)`：Params: `model`（pt 文件名）, `psample`（progressive samples 数，默认 2000）
  - `update_naru(...)`：dynamic 更新——加载老模型，在新数据上 fine-tune 1 epoch

**I/O**: `output/model/{ds}/{ver}-{model_name}_warm{warmups}-{seed}.pt`

**Paper §2.4 摘录**: "Naru 用 autoregressive 把 joint 分解为 $\prod P(A_i | A_1..A_{i-1})$。MADE 或 Transformer。点查询直接乘；range 查询用 progressive sampling：按 inverse CDF 在每列 conditional 上采样，sample 数 = `psample`。"

#### lecarb/estimator/bayesnet.py — Bayesian Network

**算法**: 学一个 Bayesian network（默认 pomegranate；可选 pgmpy via `use_pgm=True`），推断也是 progressive sampling（和 Naru 同框架，只是底层 model 不同）。

**Params**: `samples`（progressive samples）, `discretize`（每列离散 bin 数）, `parallelism`（pomegranate 并行度）。

**Paper §3 摘录**: "我们用 [13, 95] 的实现：progressive sampling 估 range 查询。Bayes 在所有数据集都很 promising。"

#### lecarb/estimator/deepdb/ — DeepDB (Hilprecht et al. 2019)

**结构**: 上游 DeepDB 代码 drop 在这。子目录:
- `aqp_spn/` — SPN learning：`AQPSPN` 主类，leaf 类型 `Categorical` / `IdentityNumericLeaf`
- `ensemble_compilation/` — 查询解析 + SPN 推断（`spn_ensemble.py`, `probabilistic_query.py`）
- `ensemble_creation/` — 多 SPN ensemble 策略（`RDC-based`, `naive`）
- `data_preparation/` — CSV → HDF 转换 + sampling
- `evaluation/` — query parsing utilities

**入口** in `deepdb.py`:
- `train_deepdb(seed, dataset, version, workload, params, sizelimit)`：
  - 转 HDF 为快速加载 → `AQPSPN.learn()` 建 SPN
  - Params: `hdf_sample_size`（SPN 训练样本数，1M）, `rdc_threshold`（RDC 系数阈值，0.3，控制列切分），`ratio_min_instance_slice`（叶子 slice 最小比例，0.01）
  - **训练无验证 loss**（DeepDB 是 data-driven，不依赖 query）；论文里专门挑 100 个 validation query 做超参 grid search
- `class DeepDB(Estimator)`：`query()` → 拆 query 为 conjuncts → SPN ensemble 算条件概率 × row_num
- `test_deepdb(...)`：Params: `model`（pkl 文件名）

**Paper §2.4 摘录**: "DeepDB 用 SPN：递归按 row cluster（sum node）或 column cluster（product node）切分。Row clustering 用 KMeans，column 独立性检测用 Randomized Dependency Coefficients (RDC)。叶子是单列分布（categorical → histogram，numerical → piecewise linear）。"

---

## 7. 如何加新 Estimator（扩展指南）

加一个名为 `myest` 的新方法，需要做：

1. **创建文件** `lecarb/estimator/myest.py`（单文件简单方法）或 `lecarb/estimator/myest/` 目录（复杂方法）
2. **继承基类**:
   ```python
   from .estimator import Estimator
   class MyEst(Estimator):
       def __init__(self, table, **kwargs):
           super().__init__(table)
           # 加载/构建你的模型
       def query(self, query):
           # 必须返回 (est_card: int, dur_ms: float)
           ...
   ```
3. **暴露 train/test 函数**（如果是 learned，否则只暴露 test）:
   ```python
   def train_myest(seed, dataset, version, workload, params, sizelimit):
       table = load_table(dataset, version)
       ...
   def test_myest(seed, dataset, version, workload, params, overwrite):
       from .utils import run_test
       table = load_table(dataset, version)
       estimator = MyEst(table, **params)
       run_test(dataset, version, workload, estimator, overwrite)
   ```
4. **注册到 `__main__.py`**：在 `if args["train"]:` 和 `if args["test"]:` 分支加 elif
5. **加 Justfile target**（参考已有 `test-mhist`、`train-naru`）:
   ```
   test-myest dataset='census13' version='original' workload='base' your_param='default' seed='123':
       uv run python -m lecarb test -s{{seed}} -d{{dataset}} -v{{version}} -w{{workload}} -emyest --params \
           "{'your_param': {{your_param}}}"
   ```
6. **如有依赖** → 在 `pyproject.toml` 加 `==` 精确锁版本，跑 `uv lock` 重新生成 lockfile
7. **跑通后** 在 `LOG.md` 加 changelog entry（按 [CLAUDE.md §2 模板](../CLAUDE.md)）

---

## 8. 关键数据 / 文件命名约定

| 路径模板 | 内容 |
|---------|------|
| `data/{ds}/{ver}.csv` | 原始数据 |
| `data/{ds}/{ver}.pkl` | `csv2pkl` 后的 DataFrame |
| `data/{ds}/{ver}-table.pkl` | `pkl2table` 后的 Table 对象 |
| `data/{ds}/{ver}_num.csv` | `table2num` 后（KDE/PG 用） |
| `data/{ds}/workload/{wl}.pkl` | workload 文件（queryset） |
| `data/{ds}/workload/{wl}-{ver}-label.pkl` | ground truth cardinality |
| `output/model/{ds}/{ver}-{est_name}-*.{pt|pkl}` | 训练好的模型 |
| `output/result/{ds}/test-{est_name}-{ver}-{wl}-*.csv` | 测试结果（每 query 一行） |
| `output/log/{ds}/*.log` | 训练/测试 log |

---

## 9. 参考资料速查

- Paper §1 引言：[Are We Ready For Learned CE?](./Wang%20等%20-%202021%20-%20Are%20we%20ready%20for%20learned%20cardinality%20estimation.pdf)
- Paper §3 实验设置：含数据集、workload、hyperparam 细节
- README.md：基础流程
- hyper-params.md：每个 (dataset, method) 推荐超参
- LOG.md：本 repo 修改历史
- LOG_EXEC.md：跑实验命令手册
- 本文件：代码结构导览

**外部链接**:
- [Naru 原仓库](https://github.com/naru-project/naru)（含 BayesNet baseline）
- [MSCN 原仓库](https://github.com/andreaskipf/learnedcardinalities)
- [DeepDB 原仓库](https://github.com/DataManagementLab/deepdb-public)
- [QuickSel 原仓库](https://github.com/illinoisdata/quicksel)
- [KDE-FB 原仓库](https://github.com/martinkiefer/feedback-kde)
