"""
==============================================================
教学注释 (L2 — lecarb integration, MSCN 全部训练/推理逻辑入此文件)
==============================================================

[本文件相对 L1 的改造概要]
L1 把 MSCN 实现散在 7 个文件 (train.py, mscn/{model, util, data, common, datasets,
queries}.py, ~700 行). L2 把训练 + 推理 + helper 全部 *合并到本文件 (mscn.py)
+ model.py 共 2 个文件 (~470 行)*, 重新组织成 "实现 lecarb Estimator 接口的插件"
形态. 详见 [../../../../AllModels/MSCN/LOG_STRUCTURE.md §10.2](../../../../AllModels/MSCN/LOG_STRUCTURE.md).

[主要架构变化]
1. **从 argparse 改成 params: Dict** — lecarb CLI (`just train-mscn <dataset>`)
   通过 dict 传递超参; `Args` class 给默认值 + .update(params) 应用用户覆盖.
2. **Estimator 基类适配** — 新增 `class MSCN(Estimator)`, 实现 `.query(q) →
   (cardinality, latency_ms)` 接口. lecarb 的 `run_test()` 拿任意 Estimator 就
   能跑测, 跟 Naru/DeepDB/LWNN 等其它估计器统一. L1 的 test 函数走 MSCN 私有
   流程, L2 把测试逻辑让给 `run_test`.
3. **统一接口替代私有 helper**:
   - L1 自管 LoadForest / column_min_max → L2 用 `load_table(dataset, version)`
   - L1 自管 LoadForestQueries → L2 用 `load_queryset` + `load_labels` + `query_2_triple`
   - L1 自管 q-error / metric → L2 用 `qerror` / `evaluate` (lecarb 共享)
4. **Checkpoint state 自包含** — L1 .pt 只存 (model_state_dict, min_val, max_val);
   L2 .pt 还包括 (samples ndarray, args, label_range, seed, device, threads,
   dataset, version, workload, model_size, valid_error, train_time, current_epoch)
   —— *可分发, 离机加载也能完整恢复*.
5. **lecarb 集成约束**:
   - `torch.set_num_threads(NUM_THREADS)` 统一 4 线程 (跟 BayesNet 等公平比)
   - sizelimit 检查 (model+sample 体积超过 sizelimit*table.data_size_mb 就不训)
   - `logging.getLogger(__name__)` 替代 print, 接入 lecarb 全局日志系统
6. **文件名模板更丰富** — L1: `model/{samples}_{hid}_{seed}_{size}.pt` (容易撞);
   L2: `MODEL_ROOT/{dataset}/{version}_{workload}-{model.name()}_ep{ep}_bs{bs}_{train_num}k-{seed}.pt`
   (跨 hyperparam sweep 不重名).

[函数与 L1 文件的映射]
| L2 函数 | 来自 L1 哪个文件 | 行为变化 |
|---------|------------------|----------|
| Args                                       | (新) 替代 argparse                       | -            |
| idx_to_onehot / get_set_encoding           | mscn/util.py                              | byte-similar |
| normalize_labels                           | mscn/util.py                              | 接收 Label 对象 |
| unnormalize_labels                         | mscn/util.py                              | 删 maximum 0 clip (因为 round 已稳) |
| get_sample_bitmap                          | mscn/util.py                              | 用 query_2_triple + bool mask |
| encode_data                                | mscn/util.py                              | 接 Table.columns[c].normalize |
| load_dicts                                 | mscn/data.py                              | 不再返回 min_max_dict |
| unnormalize_torch / qerror_loss            | train.py                                  | qerror_loss 用 lecarb qerror() |
| predict / make_dataset                     | train.py / mscn/data.py                   | 基本同 L1 |
| make_sample_tensor_mask / make_predicate_* | (拆分 make_dataset 出来)                 | 复用方便 MSCN.query() 单 query 推理 |
| train_mscn                                 | train.py:train                            | 大改 (lecarb 集成) |
| MSCN class + .query                        | (新) 实现 Estimator 接口                 | -            |
| load_mscn                                  | (新) checkpoint → Estimator 实例         | -            |
| test_mscn                                  | train.py:test                             | 改用 run_test 跑评测 |

[关键模块依赖 (lecarb 子模块)]
- `from ..estimator import Estimator, OPS` — Estimator 基类 + OPS 字典 (跟 L1 util.OPS 等价)
- `from ..utils import report_model, qerror, evaluate, run_test` — 通用工具
- `from ...dataset.dataset import load_table` — 统一数据集加载入口
- `from ...workload.workload import load_queryset, load_labels, query_2_triple` —
  workload 解析
- `from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS` — 全局常量

[模型 / sample 持久化策略]
sample (1000 行 numpy 矩阵) **进 .pt** 文件 (state['samples']), 不再 *单独*
存 sample/forest_<n>_<seed>.csv. 这是关键差异 — L2 .pt 文件自包含, 可以 scp 到
另一台机器直接 load_mscn 用; L1 .pt 必须配 sample csv 文件一起拷.

[术语速查 (本文件新出现的)]
- `Estimator` — lecarb 估计器基类, 见 [../estimator.py](../estimator.py).
  约定 `__init__(table, model)` 接表 + 模型名, `.query(q)` 接 query 返回 cardinality.
- `run_test` — lecarb 统一测试函数, 跑 workload 的 test split, 写 results 文件.
- `query_2_triple` — 把 lecarb 标准 query 对象转成 (columns, operators, values) 三元组.
- `Label` (label objects) — lecarb 的 label 抽象, 有 .cardinality / .selectivity 属性.
- `Args.train_num` — 训练集大小上限 (默认 100k); 比 queryset['train'] 长就截断,
  比它短就用全部.
"""
import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataset

from .model import SetConv
from ..estimator import Estimator, OPS
from ..utils import report_model, qerror, evaluate, run_test
from ...dataset.dataset import load_table
from ...workload.workload import load_queryset, load_labels, query_2_triple
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

# === lecarb logger 接入 ===
# logging.getLogger(__name__) 用模块名 (eg. "lecarb.estimator.mscn.mscn") 作为
# logger 标识. lecarb 全局有 root logger 配置, 所有 L.info / L.debug 都被路由到
# 统一日志文件 / stdout. 替代 L1 满天飞的 print.
L = logging.getLogger(__name__)

# === Args: 配置容器 (替代 L1 的 argparse) ===
# 用法: args = Args(bs=512, hid_units=128, ...) — 关键字参数覆盖默认值.
# 默认超参跟 L1 不同:
#   - epochs 200 (L1 500) — lecarb 用 valid early-stop, 200 够; 节省训练时间
#   - hid_units 256 (L1 28) — lecarb 默认按 paper, 不再压成 size-fair
#   - train_num 100k 上限 (L1 没限制, 用 workload 全量)
# `self.__dict__.update(kwargs)`: Python 把对象属性当 dict 修改的 idiom — 把 user
# 传进来的 dict 一次性 merge 进 self, 简单且灵活. *副作用*: 用户传不存在的 key
# 也会被存进去 (eg. 拼错 `bath=1024`), 没 schema 检查. 这是 ARELY benchmark 的
# tradeoff: 灵活但需要使用者保证 key 名对.
class Args:
    def __init__(self, **kwargs):
        self.bs = 1024
        self.lr = 0.001
        self.epochs = 200
        self.num_samples = 1000
        self.hid_units = 256
        self.train_num = 100000

        # overwrite parameters from user
        self.__dict__.update(kwargs)

# === idx_to_onehot / get_set_encoding: 与 L1 util.py 等价 (byte-similar) ===
# 详注请直接看 [L0 util.py](../../../../AllModels/MSCN/mscn/util.py) 的对应函数.
# 唯一区别: L2 把这些 helper 内联到 mscn.py, 不再独立模块. lecarb 设计哲学是
# "estimator 自给自足", 减少跨模块跳跃.
def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot

def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing

# === L2 PATCH (vs L1 normalize_labels): 接收 Label 对象列表, 取 .cardinality ===
# L1 是 list[int] (整数 cardinality); L2 是 list[Label] (lecarb 抽象), 每个 Label
# 有 .cardinality 和 .selectivity 属性. Label 对象由 load_labels() 返回 (见
# lecarb/workload/workload.py).
# 平滑公式 log(card + 1) 与 L1 完全一致.
def normalize_labels(labels, min_val=None, max_val=None):
    # +1 to deal with 0 scenario
    labels = np.array([np.log(float(l.cardinality+1)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        L.info(f"min log(label): {min_val}")
    if max_val is None:
        max_val = labels.max()
        L.info(f"max log(label): {max_val}")
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

# === L2 PATCH (vs L1 unnormalize_labels): 删 np.maximum(..., 0) clip ===
# L1 是 np.maximum(np.round(np.exp(labels) - 1), 0) 防浮点误差让结果 < 0.
# L2 删了 maximum. 因为 round 之后浮点误差被舍入吃掉 → 实践中 ≥ 0 (eg.
# exp(0) - 1 = 0, round 后还是 0). 如果输入 labels_norm 严格 ≥ 0 (网络 sigmoid
# 保证), 不会出现负数. 这是 lecarb 团队精简后的版本.
def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    # -1 to deal with 0 scenario, need to restrict to >= 0
    return np.array(np.round(np.exp(labels) - 1), dtype=np.int64)

# === L2 PATCH (vs L1 get_sample_bitmap): 用 query_2_triple + bool mask ===
# L1 直接拿 (columns, operators, values) 平行数组 (从 LoadForestQueries 返回的);
# L2 拿 lecarb 标准 query 对象, 调 `query_2_triple` 转成三元组. 参数:
#   - with_none=False: 忽略 None 值的 predicate (eg. column 没条件)
#   - split_range=False: '[]' 闭区间保留, 不拆 — 因为 OPS['[]'] = in_between 函数
#     已经能直接处理 (lo, hi) tuple, 不用拆
# `np.ones(len(sample), dtype=bool)`: 初始全 True 数组, `&=` 累积 boolean AND.
# 末尾 `.astype(int)` 转回 int (跟 L1 的 numpy uint8 bitmap 兼容下游).
def get_sample_bitmap(sample, query):
    # do not need to convert [] to >= and <= here
    columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
    bitmap = np.ones(len(sample), dtype=bool)
    for c, o, v in zip(columns, operators, values):
        bitmap &= OPS[o](sample[c], v)
    return [bitmap.astype(int)]

# === L2 PATCH (vs L1 encode_data): 用 lecarb Table.columns[c].normalize ===
# 关键变化:
#   - query_2_triple split_range=True (这里要拆, 因为网络 op2vec 没 '[]')
#   - `table.columns[c].normalize(v)` 替代 L1 的 `normalize_data(v, c, min_max_dict)`.
#     lecarb 的 Column 对象自带 normalize 方法 (per-column min/max 内化, 见
#     lecarb/dataset/dataset.py).
def encode_data(table, query, column2vec, op2vec):
    columns, operators, values = query_2_triple(query, with_none=False, split_range=True)
    predicates_enc = []
    for c, o, v in zip(columns, operators, values):
        norm_val = table.columns[c].normalize(v)
        pred_vec = []
        pred_vec.append(column2vec[c])
        pred_vec.append(op2vec[o])
        pred_vec.append(norm_val)
        pred_vec = np.hstack(pred_vec)
        predicates_enc.append(pred_vec)
    # for no predicate scenario
    if len(predicates_enc) == 0:
        predicates_enc.append(np.zeros((len(column2vec) + len(op2vec) + 1)))
    return predicates_enc

# === L2 NEW: encode_datas — encode_data 的 list 版本, 训练时用 ===
# 名字带 's' 复数, 一次性 encode 一批 query. 实际上 train_mscn 里走的是 list
# comprehension `[encode_data(...) for q in ...]` (line 241-242), 没直接调
# encode_datas — 是预留接口或者从某次重构残留下来.
def encode_datas(table, queries, column2vec, op2vec):
    return [encode_data(table, q, column2vec, op2vec) for q in queries]

# === L2 PATCH (vs L1 load_dicts): 不返回 min_max_dict ===
# L1 返回 (column2vec, op2vec, min_max_dict); L2 返回 (column2vec, op2vec) 两元组.
# 因为 lecarb Column 对象 (table.columns[c]) 自己有 .normalize 方法, normalize 时
# 直接调它, 不需要 min/max 字典.
# column vocab 也变了: L1 用 `range(table.data.shape[1])` (整数列索引);
# L2 用 `table.data.columns` (pandas DataFrame 的列名). lecarb 的 query_2_triple
# 返回的 column id 跟 table.data.columns 的 order 一致, 所以 vocab 改用 column 名
# 一样能对齐.
def load_dicts(table):
    # Get column name dict
    # we assume any column can have predicates
    column2vec, _ = get_set_encoding(table.data.columns)

    # Get operator name dict
    # NOTICE: [] should be converted to two operators: >= and <= later for mscn
    operators = set(['=', '>=', '<='])
    op2vec, _ = get_set_encoding(operators)

    # Get min max value for each column
    return column2vec, op2vec

# === unnormalize_torch: 跟 L1 train.py 同名同实现 ===
# torch.exp(vals) - 1: 反对应 normalize_labels 的 log(l+1).
def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals) - 1 # -1 since we +1 when normalize

# === L2 PATCH (vs L1 qerror_loss): 用 lecarb qerror() + 处理 non-tensor 边界 ===
# L1 用本地 error_metric 函数 (q-error + 1 平滑) 算 loss; L2 调 `qerror` 函数
# (从 ..utils import 来), 这是 lecarb 跨 estimator 共享的 q-error 实现, 同一公式
# 让多个 estimator (MSCN/Naru/DeepDB/LWNN) 报告的 q-error 数字可直接比较.
#
# 复杂之处: `qerror(preds[i], targets[i])` 在 preds 或 targets 是 *Python scalar*
# (eg. estimator 返回 int) 时返回非 tensor 值; 在 tensor 时返回 tensor. 这里用
# `torch.is_tensor(e)` 检查, 不是 tensor 就手工 wrap 成 tensor (requires_grad=True
# 保证后续 backward 能算梯度). 这种二态处理是因为 lecarb 的 qerror 函数本身
# polymorphic.
def qerror_loss(preds, targets, min_val, max_val):
    errors = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        e = qerror(preds[i], targets[i])
        errors.append(e if torch.is_tensor(e) else torch.tensor([e], requires_grad=True, device=torch.device(DEVICE)))
    return torch.mean(torch.cat(errors))

# === predict: 跟 L1 train.py:predict 同名同实现 (单表 5-tuple unpack) ===
# 详见 [L1 train.py](../../../../AllModels/0_AreCELearnedYet/MSCN/train.py) 同名函数注释.
def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, targets, sample_masks, predicate_masks = data_batch

        if cuda:
            samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
            sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
        samples, predicates, targets = Variable(samples), Variable(predicates), Variable(
            targets)
        sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)

        t = time.time()
        outputs = model(samples, predicates, sample_masks, predicate_masks)
        t_total += time.time() - t

        for i in range(outputs.shape[0]):
            preds.append(outputs[i].cpu().item())

    return preds, t_total

# === L2 REFACTOR: 把 L1 make_dataset 的 sample / predicate pad 步骤拆出来 ===
# 拆成两个 helper 函数, 让 MSCN.query() (单 query 推理) 也能 reuse — 单 query 推
# 理时不需要 batch, 只 pad 单条 predicate 到自身 max 即可.
#
# make_sample_tensor_mask: 单表 query 的 sample 只有 1 行 (恒等于 1, 见 L1 data.py
# 的 assert num_pad == 0), 所以不需要 pad. 直接 vstack + mask 全 1.
def make_sample_tensor_mask(sample):
    # no need to pad since only for single table
    sample_tensor = np.vstack(sample)
    sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
    return sample_tensor, sample_mask

# make_predicate_tensor_mask: predicate set 需要 pad 到 max_pred. 跟 L1 make_dataset
# 内部 predicate 段逻辑完全一致, 只是函数化了.
def make_predicate_tensor_mask(predicate, max_pred):
    predicate_tensor = np.vstack(predicate)
    num_pad = max_pred - predicate_tensor.shape[0]
    predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
    predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
    predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
    return predicate_tensor, predicate_mask

# === make_dataset: 跟 L1 data.py:make_dataset 同名 (基本同实现) ===
# 唯一区别: 内部调上面拆出来的 make_sample_tensor_mask / make_predicate_tensor_mask,
# 以及 print 改 L.debug. 元组顺序还是 (samples, predicates, targets, sample_masks,
# predicate_masks).
def make_dataset(samples, predicates, labels, max_pred):
    """Add zero-padding and wrap as tensor dataset."""
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor, sample_mask = make_sample_tensor_mask(sample)
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)
    L.debug(f'Sample tensor shape: {sample_tensors.shape}, mask shape: {sample_masks.shape}')

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate, max_pred)
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)
    L.debug(f'Predicate tensor shape: {predicate_tensors.shape}, mask shape: {predicate_masks.shape}')

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, target_tensor,
                                 sample_masks, predicate_masks)

# === train_mscn: 主训练函数 (替代 L1 train.py:train + 大量 lecarb 集成) ===
# lecarb CLI 调用方式: `just train-mscn <dataset> --params bs=512,hid_units=128 ...`
# 在 lecarb/cli.py 里 dispatch 到本函数. 入参:
#   - seed: 全局随机种子
#   - dataset: 数据集名 (eg. 'census', 'forest', 'dmv')
#   - version: 数据集版本 (lecarb 支持同一 dataset 多 schema/data 版本)
#   - workload: workload 名 (eg. 'base', 'eval')
#   - params: dict, hyperparam 覆盖 (传给 Args.__init__)
#   - sizelimit: 浮点, > 0 时启用 size limit (单位是 table.data_size_mb 的倍数)
#
# 流程跟 L1 train 类似但加了大量 lecarb 集成:
#   1. 统一线程数 + seed
#   2. load_table + load_queryset + load_labels
#   3. 截断到 train_num (lecarb 标配, 防止训练时间炸)
#   4. 算 model_size + sample_size → 跟 sizelimit 对比 (超就退出)
#   5. encode 训练 / 验证集 → make_dataset → DataLoader
#   6. 训练 + 每 epoch valid → torch.save best to MODEL_ROOT/{dataset}/...pt
def train_mscn(seed, dataset, version, workload, params, sizelimit):
    # === lecarb 集成步骤 1: 统一 torch 线程数 ===
    # lecarb 用 NUM_THREADS=4 (在 constants.py 定义) 给所有 estimator 设统一线程,
    # 让 *不* 用 GPU / 多核的方法 (BayesNet, KDE 等) 跟用并行的方法公平比较.
    # `assert == get_num_threads()`: 防御 (PyTorch 在某些 backend 下 set 不上时
    # 会静默, 这里 fail-fast).
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    # === lecarb 集成步骤 2: 显式 set seed (np + torch) ===
    # L1 在 train.py:main 里设, L2 在 train_mscn 里设 — 因为 train_mscn 直接被 CLI
    # dispatch 调用, 没有 main 函数夹在中间.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # convert parameter dict of mscn
    L.info(f"params: {params}")
    # Args(**params) 把 params dict 展开成关键字参数. 默认值 (bs=1024, lr=0.001 等)
    # 由 Args.__init__ 给, 用户提供的 key 覆盖之.
    args = Args(**params)

    # === lecarb 集成步骤 3: 通过统一 API load 数据 + workload ===
    # load_table: 见 lecarb/dataset/dataset.py, 返回 lecarb Table 对象 (跟 L1
    # common.CsvTable 接口接近, 但加了 .columns[c].normalize / .data_size_mb /
    # .row_num 等 lecarb 扩展属性).
    table = load_table(dataset, version)
    L.info(f"Start loading queryset:{workload} and labels for version {version} of dataset {dataset}...")
    # load_queryset: 返回 {'train': [...], 'valid': [...], 'test': [...]} 三段
    # 已经分好的 query dict. lecarb 不再 90/10 自己切, workload 文件自带 split.
    queryset = load_queryset(dataset, workload)
    # load_labels: 返回 {'train': [Label, ...], 'valid': [...], 'test': [...]},
    # 每个 Label 是带 .cardinality / .selectivity 的对象 (而不是 int).
    labels = load_labels(dataset, version, workload)
    # === train_num 截断 ===
    # Args.train_num=100k 是上限. 如果 workload 训练集 > 100k 就截前 100k; 否则
    # 全用. valid_num = train_num // 10 = 10k 上限.
    if args.train_num < len(queryset['train']):
        queryset['train'] = queryset['train'][:args.train_num]
        labels['train'] = labels['train'][:args.train_num]
    valid_num = args.train_num // 10
    if valid_num < len(queryset['valid']):
        queryset['valid'] = queryset['valid'][:valid_num]
        labels['valid'] = labels['valid'][:valid_num]
    L.info(f"Use {len(queryset['train'])} queries for train and {len(queryset['valid'])} queries for validation")

    # === 模型实例化 + 体积报告 ===
    # report_model: lecarb 工具, 算并 log model size, 返回 MB. 跟 L1 SetConv.size()
    # 类似但在外面跑 — model.py 不再带 size() 方法.
    # create model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    model_size = report_model(model)

    # === sample materialize: 抽 sample, 算 sample 体积 ===
    # table.data.sample(n, random_state=seed): pandas API, random_state 控制
    # 抽样的随机种子 (跟全局 np.random.seed 独立, 防止跑了别的 np.random.* 之后
    # 这里抽出不同样本).
    # materialize sample
    sample = table.data.sample(n=args.num_samples, random_state=seed)
    # sample_size 按 *比例* 算: 1000 行占全表 1000/row_num, 对应 data_size_mb 的同比例.
    # 这是估计 sample 在内存的占用, 用于 sizelimit 检查.
    sample_size = table.data_size_mb * (args.num_samples / table.row_num)

    # === L2 NEW: size limit 早退检查 ===
    # mscn_size = model 参数 + 1000 行 sample, 整体大小.
    # sizelimit = 0 表示不检查 (常用); > 0 表示 cap 在 `sizelimit * data_size_mb` 内.
    # 超 cap 就直接 return 不训练 (省时间). ARELY benchmark sweep 时用 sizelimit
    # 来快速跳过会超体积的配置.
    # check size limit
    mscn_size = model_size + sample_size
    if sizelimit > 0 and mscn_size > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mscn_size:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue training!")
        return
    L.info(f'Overall MSCN model size + sample size = {mscn_size:.2f}MB')

    # === encode all queries + 共享归一化范围 ===
    # 注意: normalize_labels 接收 labels['train'] + labels['valid'] (Python list
    # 拼接) — 拿合并后的 *全部* labels 算 min/max, 保证 train + valid 用同一套
    # 归一化, 不会因为 valid 出现极端 card 而被 clip. 然后再 split 回去.
    # Get feature encoding and proper normalization
    samples_train = [get_sample_bitmap(sample, q) for q in queryset['train']]
    samples_valid = [get_sample_bitmap(sample, q) for q in queryset['valid']]
    predicates_train = [encode_data(table, q, column2vec, op2vec) for q in queryset['train']]
    predicates_valid = [encode_data(table, q, column2vec, op2vec) for q in queryset['valid']]
    label_norm, min_val, max_val = normalize_labels(labels['train'] + labels['valid'])
    labels_train = label_norm[:len(queryset['train'])]
    labels_valid = label_norm[len(queryset['train']):]
    L.info(f"Number of training samples: {len(labels_train)}")
    L.info(f"Number of validation samples: {len(labels_valid)}")

    # === L2 NEW: state 字典 — 自包含 checkpoint ===
    # L1 .pt 只存 (model_state_dict, min_val, max_val). L2 状态全部进 state, 让
    # .pt 可单文件分发 (内含 sample / args / metrics / metadata).
    # NOTICE 注释解释 column min/max 不存: 因为它们来自 table.columns[c].normalize,
    # 是 *Table 派生属性*. test 时只要 load 同 version 的 table 就能复原, 没必要
    # 重复存进 .pt.
    # Train model
    # NOTICE: do not record min max value for each column, make sure to load the same table when test
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        'model_size': mscn_size,
        'label_range': (min_val, max_val),
        # *samples* 是 numpy 矩阵 (table.data.sample 后的 DataFrame.values 实质).
        # torch.save 能 pickle 任意 Python 对象, 所以 ndarray 直接 dump 没问题.
        'samples': sample
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_valid_loss = float('inf')
    cuda = False if DEVICE == 'cpu' else True

    if cuda:
        model.cuda()

    max_pred = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_valid]))
    train_dataset = make_dataset(samples_train, predicates_train,
                                 labels=labels_train, max_pred=max_pred)
    valid_dataset = make_dataset(samples_valid, predicates_valid,
                                 labels=labels_valid, max_pred=max_pred)
    train_data_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.bs)

    # === L2 PATCH: 文件名模板大幅扩张 ===
    # MODEL_ROOT (lecarb 全局常量) 是 model 根目录, 通常 ./output/model/.
    # *pathlib.Path.__truediv__*: `MODEL_ROOT / table.dataset` 用 `/` 运算符拼路径,
    # 比手写 os.path.join 更 pythonic.
    # mkdir(parents=True, exist_ok=True): 父目录都建好, 已存在不报错.
    # 文件名模板:
    #   {version}_{workload}-{model_name}_ep{N}_bs{B}_{K}k-{seed}.pt
    # 例: v1_base-mscn_hid256_sample1000_ep200_bs1024_100k-123.pt
    # 内含: 数据版本 / workload / 模型名 / epoch / batch / train_num 千数 / seed.
    # 跨 hyperparam sweep 各文件不互覆盖, 解决 L1 的撞名问题.
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}_{workload}-{model.name()}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.pt"

    # === 主训练循环 — 跟 L1 train 几乎一样, 但多 valid_time 跟踪 + valid_error 进 state ===
    # valid_time 累计 valid 步骤耗时, train_time = 总时间 - valid_time, 给后续
    # 报告 "纯训练" 耗时 (不含 valid 跑) 用.
    model.train()
    L.info('start train mscn...')
    start_stmp = time.time()
    valid_time = 0
    for epoch in range(args.epochs):
        loss_total = 0.
        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, targets, sample_masks, predicate_masks = data_batch
            if cuda:
                samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
                sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
            samples, predicates, targets = Variable(samples), Variable(predicates), Variable(targets)
            sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)
            optimizer.zero_grad()
            outputs = model(samples, predicates, sample_masks, predicate_masks)
            loss = qerror_loss(outputs, targets.float().reshape(-1, 1), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {loss_total/len(train_data_loader)}, time since start: {dur_min:.1f} mins")

        # === L2 PATCH: 用 lecarb evaluate() 算 q-error 分位数 dict ===
        # evaluate() 返回 (preds_array, metrics_dict). metrics_dict 包含
        # {'mean', 'median', '90th', '95th', '99th', 'max'} 等键, 跟所有
        # estimator 共享 → 横向比较时数字直接可比.
        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        preds_valid, _ = predict(model, valid_data_loader, cuda)

        # Unnormalize
        preds_valid_unnorm = unnormalize_labels(preds_valid, min_val, max_val)
        labels_valid_unnorm = unnormalize_labels(labels_valid, min_val, max_val)

        L.info("Q-Error on validation set:")
        _, metrics = evaluate(preds_valid_unnorm, labels_valid_unnorm)

        # === best-checkpoint: 用 mean q-error 判定; state 内容更丰富 ===
        # L1 best 是 qerror.mean(); L2 同标准, 但 state 还存 optimizer_state_dict
        # (能断点续训) + valid_error metrics dict + train_time + current_epoch.
        valid_loss = metrics['mean']
        if valid_loss < best_valid_loss:
            L.info(f'best valid loss for now: {valid_loss}(mean)!')
            best_valid_loss = valid_loss
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()
            state['valid_error'] = {workload: metrics}
            # train_time = 当前累计时间 - 累计 valid 时间, 单位分钟
            state['train_time'] = (valid_stmp-start_stmp-valid_time) / 60
            state['current_epoch'] = epoch
            torch.save(state, model_file)

        # 累计 valid 段时间 (用 *本次 valid 结束* 减 *本次 valid 开始*)
        valid_time += time.time() - valid_stmp

    L.info(f'Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins')
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}")

# === L2 NEW: MSCN(Estimator) — 实现 lecarb 统一估计器接口 ===
# Estimator 基类 (lecarb/estimator/estimator.py) 约定:
#   - __init__(table, model) 接表对象 + 模型名 (字符串)
#   - .query(q) → (cardinality, latency_ms) 给定一条 query 返回估计基数 + 推理耗时
#
# `super().__init__(table=table, model=model_name)`: 让父类 Estimator 把 table /
# model_name 存好, 这样 lecarb 的 run_test 能从 estimator 拿到 table 跟 model name.
#
# *self.model.to(self.device)*: 把模型搬到 DEVICE (CPU 或 GPU, 见 constants).
# *self.model.eval()*: 切推理模式 (这里 SetConv 没 dropout/BN, 但好习惯, 也让
# 后面 with torch.no_grad() 配合).
class MSCN(Estimator):
    def __init__(self, model, model_name, samples, table, column2vec, op2vec, label_range):
        super(MSCN, self).__init__(table=table, model=model_name)
        self.model = model
        self.samples = samples
        self.column2vec = column2vec
        self.op2vec = op2vec
        self.minval = label_range[0]
        self.maxval = label_range[1]
        self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self.model.eval()

    # === .query: 单 query 推理 (Estimator 协议核心方法) ===
    # 流程:
    #   1. encode query: sample_enc + predicate_enc (复用训练同名 helper)
    #   2. pad 到张量: 用拆出来的 make_sample/predicate_tensor_mask, 没 batch 维
    #   3. expand_dims(axis=0): 补 batch=1 维, 跟训练时 [B, ...] 张量 shape 对齐
    #   4. torch.FloatTensor + .to(device): 上 CPU/GPU
    #   5. `with torch.no_grad():` 关闭 autograd → 省显存 + 提速 (推理不需要梯度)
    #   6. unnormalize_labels(pred.cpu()): 反归一化回真实 cardinality, .item() 取标量
    #   7. dur_ms = 秒 × 1000 = 毫秒延迟
    #
    # 返回: (cardinality: int, dur_ms: float) — 跟所有 Estimator 一致.
    def query(self, query):
        sample_enc = get_sample_bitmap(self.samples, query)
        predicate_enc = encode_data(self.table, query, self.column2vec, self.op2vec)

        sample_tensor, sample_mask = make_sample_tensor_mask(sample_enc)
        # max_pred = len(predicate_enc) 因为单 query 自身就是 batch 内 max
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate_enc, len(predicate_enc))

        # np.expand_dims(arr, axis=0) 在最前插 batch=1 维 → 跟 SetConv.forward 期望
        # 的 [batch_size, set_size, feat] 三维 shape 对齐.
        sample_tensor = torch.FloatTensor(np.expand_dims(sample_tensor, axis=0)).to(self.device)
        sample_mask = torch.FloatTensor(np.expand_dims(sample_mask, axis=0)).to(self.device)
        predicate_tensor = torch.FloatTensor(np.expand_dims(predicate_tensor, axis=0)).to(self.device)
        predicate_mask = torch.FloatTensor(np.expand_dims(predicate_mask, axis=0)).to(self.device)

        start_stmp = time.time()
        # `with torch.no_grad():` 是上下文管理器 — 块内所有 tensor 都不建梯度计算
        # 图, 推理快 30% + 省显存. 不写 no_grad 也能跑, 但浪费.
        with torch.no_grad():
            pred = self.model(sample_tensor, predicate_tensor, sample_mask, predicate_mask)
        # 毫秒 = 秒 × 1000; 1e3 = 1000 浮点写法
        dur_ms = (time.time() - start_stmp) * 1e3

        # pred 形状 [1, 1] (batch=1, 输出=1). .cpu() 搬回 CPU (unnormalize 是 numpy).
        # unnormalize_labels 返回 ndarray, .item() 转成 Python int 标量.
        return unnormalize_labels(pred.cpu(), self.minval, self.maxval).item(), dur_ms

# === L2 NEW: load_mscn — checkpoint → 实例化的 MSCN estimator ===
# 给外部 (eg. lecarb CLI dispatch / 其它脚本) 用的入口. 流程:
#   1. torch.load .pt 文件 (map_location=DEVICE 让旧 GPU saved 模型能 load 到 CPU)
#   2. 从 state['args'] 拿超参 (Args 对象, pickled in)
#   3. load_table 用 state 里记录的 version 把 *精确同一份* 数据集复原 — 这是为
#      什么 train 时不存 column min/max: 同 dataset+version 算出来一致.
#   4. 重 build vocab + 重 build model 架构 (按 args.num_samples / args.hid_units 还原)
#   5. load_state_dict 把权重灌回去
#   6. 拿 state['samples'] (numpy 矩阵) + label_range 实例化 MSCN(Estimator)
#
# 返回 Tuple[Estimator, Dict] — Estimator 给 run_test, state 给调用方读其它字段
# (eg. valid_error 报告).
# Type hint: `Tuple[Estimator, Dict[str, Any]]` 是 typing 模块的泛型, 跟 Java 的
# `Tuple<Estimator, Map<String, Object>>` 类似 — 仅给 IDE 提示, runtime 不强检查.
def load_mscn(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    table = load_table(dataset, state['version'])
    # load model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    report_model(model)
    L.info(f"Overall MSCN model size + sample size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])

    estimator = MSCN(model,
                     model_name,
                     state['samples'],
                     table,
                     column2vec,
                     op2vec,
                     state['label_range'])

    return estimator, state


# === L2 NEW: test_mscn — lecarb CLI test 入口 (替代 L1 train.py:test) ===
# 跟 load_mscn 大部分逻辑重叠 (二者都做 .pt → estimator), 但 test_mscn 多做了
# 跑 run_test (走 workload 的 test split 报告 q-error) + 写 results 文件.
# 没复用 load_mscn 是因为 model_name 在两边来源不同: load_mscn 是入参; test_mscn
# 从 params['model'] 拿.
#
# 入参:
#   dataset / version / workload: lecarb 工作流标识 (跟 train_mscn 对称)
#   params: 必须包含 'model' key (model 文件 stem); 其它 key 备用
#   overwrite: True 则覆盖已存在的 results 文件, False 则跳过
def test_mscn(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
    """
    # uniform thread number
    # 跟 train_mscn 一样设 NUM_THREADS, 让推理时 latency 跟训练时可比 (公平 benchmark)
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    # load model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    report_model(model)
    L.info(f"Overall MSCN model size + sample size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])

    estimator = MSCN(model,
                     params['model'],
                     state['samples'],
                     table,
                     column2vec,
                     op2vec,
                     state['label_range'])

    # === lecarb 集成: run_test 跑 workload test split + 写 results ===
    # run_test (lecarb/estimator/utils.py) 内部:
    #   1. load_queryset(dataset, workload)['test'] 拿测试 queries
    #   2. for each q: estimator.query(q) → (card, latency)
    #   3. 跟 ground-truth labels 算 q-error
    #   4. 写 results/{dataset}/{version}_{workload}-{model}.csv (含 qerror, pred, true, latency)
    #   5. log 分位数统计
    # **跟所有 estimator 共用同一个 run_test → 报告数字直接可比**.
    L.info(f"load and built mscn estimator: {estimator}")
    run_test(dataset, version, workload, estimator, overwrite)
