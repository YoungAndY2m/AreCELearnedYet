"""Model training."""
# ================================================================
# 教学注释 (annotation pass, L2 综合) — naru.py 总览
# ================================================================
# L2 = ARELY 团队的 lecarb 整合版。把 L0 的 7 个文件
# (train_model.py / eval_model.py / estimators.py:ProgressiveSampling /
#  common.py:TableDataset / datasets.py / queries.py / query_formatter.py)
# 合并重写进这 1 个文件 (~926 行), 接入 lecarb 框架的统一 CLI / 数据
# 抽象 / workload 抽象 / 模型存储目录 / 评估工具。算法核心 (MADE +
# progressive sampling) 没变, 见 [L0 estimators.py:134](../../../../AllModels/Naru/estimators.py#L134) 的详细注释。
#
# 跟 L0 / L1 的对比表
# ----------------------------------------------------------------
#   L0 文件                           → L2 去向 (本文件)
#   train_model.py 的 argparse        → 这里的 Args class (dict→对象)
#   train_model.py 的 TrainTask()     → train_naru()
#   train_model.py 的 RunEpoch        → RunEpoch (除了 args 改成参数传入, 几乎相同)
#   eval_model.py 的 Main()           → test_naru()
#   eval_model.py 的 ckpt 加载逻辑    → load_naru()
#   estimators.py:ProgressiveSampling → Naru (class), 重写成 lecarb Estimator
#   common.py:TableDataset            → NaruTableDataset (用 lecarb Column API)
#   common.py:Column/Table/CsvTable   → 删除, 用 lecarb 自己的 Table/Column
#   datasets.py:LoadDmv / LoadForest  → 删除, 用 load_table(dataset, version)
#   eval_model.py:GenerateQuery       → 删除, query 由 workload module 生成
#   queries.py / query_formatter.py   → 删除, 用 query_2_triple / load_queryset
#
# 新增 (L2-only) 功能
# ----------------------------------------------------------------
#   - update_naru()  : incremental fine-tune (在新版本数据上接着训), L0 没有
#   - 'auto' input/output encoding 模式 (made.py:AutoEncode 实现)
#   - state dict 多存了 optimizer_state / args / valid_error / threads /
#     dataset / version / model_bits 等元数据 (L0 只存 model_state_dict)
#   - NUM_THREADS 强制 (公平 benchmark)
#   - VALID_NUM_DATA_DRIVEN 在线 valid q-error 追踪
#
# 三个 CLI 入口 (lecarb 调用)
# ----------------------------------------------------------------
#   - train_naru(seed, dataset, version, workload, params, sizelimit)
#       由 `lecarb train --estimator naru` 触发
#   - test_naru(seed, dataset, version, workload, params, overwrite)
#       由 `lecarb test --estimator naru` 触发
#   - update_naru(...)  增量微调入口 (新版数据上继续训)
#
# 跟 lecarb 框架的接口
# ----------------------------------------------------------------
#   - Estimator (base class): 要求子类实现 .query(query) → (card, dur_ms)
#   - load_table(dataset, version): 返回 lecarb Table 对象 (替代 CsvTable)
#   - load_queryset / load_labels: 标准 query / ground truth 加载
#   - query_2_triple: 把 Query 对象拆成 (columns, operators, vals) 三元组
#   - report_model / run_test / evaluate: 通用模型/测试工具
#   - MODEL_ROOT / dataset / *.pt: 标准模型存储目录布局
# ================================================================
import time
import copy
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import made
from . import transformer
# lecarb 框架的核心抽象:
#   - Estimator: 所有 estimator 的基类, 强制 .query() 接口签名
#   - OPS: 字符串 '=' / '<' / '>=' 等到 numpy 比较算子的映射
from ..estimator import Estimator, OPS
# lecarb 共享工具:
#   - report_model: 打印模型参数数 + MB (L0 版叫 ReportModel, 拷贝出来作通用工具)
#   - run_test: 标准评估循环 (跑 workload 的所有 query, 算 q-error, 存结果 CSV)
#   - evaluate: 算 q-error 分位数 (median / 95th / 99th / max)
from ..utils import report_model, run_test, evaluate
# lecarb 常量:
#   - DEVICE: 'cuda' / 'cpu', 跟 L0 一致
#   - MODEL_ROOT: 模型存储根目录, checkpoint 写在 MODEL_ROOT/{dataset}/*.pt
#   - NUM_THREADS: torch 线程数上限 (= benchmark 公平性, 不同 estimator 在同样 thread 下比)
#   - VALID_NUM_DATA_DRIVEN: 训练时 online valid 集 query 数 (默认几百, 用来监控过拟合)
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS, VALID_NUM_DATA_DRIVEN
# load_table: lecarb 的统一 dataset loader (替代 L0 的 datasets.LoadDmv())。
# 输入 (dataset_name, version), 自动找 CSV/parquet, 构造 Table 对象。
from ...dataset.dataset import load_table
# - load_queryset: 加载 workload 下的 train/valid/test query
# - load_labels:   加载对应 query 的 ground truth cardinality
# - query_2_triple: Query 对象 → (cols, ops, vals) 三元组, 兼容 L0 的接口
from ...workload.workload import load_queryset, load_labels, query_2_triple

# lecarb 用标准 logging 而不是 print (L0 用 print, L2 改 logger 是工程化的体现)。
L = logging.getLogger(__name__)

# ================================================================
# Args: 配置容器 (替代 L0 的 argparse)
# ================================================================
# lecarb CLI 把超参装在 params dict 里传过来 (而不是 argparse 解析 CLI),
# 这个类用 **kwargs 解包+ 默认值, 让代码里 args.xxx 写法跟 L0 一样, 不用大改。
# 默认值的来源:
#   - 大部分 = L0 train_model.py argparse 的 default
#   - residual / direct_io / column_masking 默认 True (L0 是 False, L2 把它们当推荐配置默认开)
#   - bs = 2048 (L0 默认 1024, L2 调大了)
#   - embed_size = 64 (L0 默认 32, L2 调大)
#   - embed_threshold = 128 (L2 新加, 给 AutoEncode 用, L0 没这选项)
class Args:
    def __init__(self, **kwargs):
        # general parameters
        self.order = None
        self.num_orderings = 1
        self.bs = 2048
        self.epochs = 20
        self.column_masking = True
        self.constant_lr = None
        self.warmups = 0

        # Transformer TODO
        self.heads = 0
        self.blocks = 2
        self.dmodel = 32
        self.dff = 128
        self.transformer_act = 'gelu'

        # MADE & ResMADE
        self.fc_hiddens = 128
        self.layers = 4
        self.residual = True
        self.direct_io = True
        self.inv_order = False
        self.input_encoding = 'binary'
        self.output_encoding = 'one_hot'
        self.embed_size = 64
        self.embed_threshold = 128

        # overwrite parameters from user
        # 用户从 lecarb CLI 传来的 params 覆盖默认值, e.g.
        #   just train-naru ... --params "{'input_encoding': 'auto', 'epochs': 100}"
        self.__dict__.update(kwargs)

# ================================================================
# NaruTableDataset: 整张表 discretize 成 tensor (替代 L0 的 common.TableDataset)
# ================================================================
# 跟 L0 TableDataset 功能一样, 区别只是用 lecarb Column 的 API:
#   - L0: col.data (raw values) + col.all_distinct_values (vocab)
#     + Discretize(col) 全局函数
#   - L2: table.data 是 DataFrame, col.discretize(series) 是 method
#         col.vocab_size 拿 domain 大小 (= L0 col.DistributionSize())
#         col.vocab 拿 vocab array      (= L0 col.all_distinct_values)
# 接口对外不变, 继续返回 [N, ncols] 的 float32 tensor 给 DataLoader。
# 见 [L0 common.py:TableDataset](../../../../AllModels/Naru/common.py) 的详细 ML 解释。
class NaruTableDataset(Dataset):
    def __init__(self, table):
        super(NaruTableDataset, self).__init__()
        # 深拷贝 lecarb Table, 隔离训练数据。
        table = copy.deepcopy(table)
        # 对每列分别 discretize → [N], 然后 stack 成 [N, ncols]。
        # col.discretize(series) 内部就是 L0 Discretize(col) 那套逻辑:
        # vocab 编码 + NaN 永远 bin_id=0 的约定。
        self.tuples_np = np.stack([col.discretize(table.data[cname]) for cname, col in table.columns.items()], axis=1)
        # float32 tensor (虽然内容是整数 bin_id, float 方便 GPU; EncodeInput 里再 .long())
        self.tuples = torch.as_tensor(self.tuples_np.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

# ================================================================
# Entropy: 算数据熵, 给 train_naru 报 "table_bits" 用
# ================================================================
# 跟 L0 完全一样 (打印改 L.info 而已)。table_bits = data NLL 的下界,
# 模型 NLL - table_bits = "entropy gap" 衡量欠拟合程度。
# 见 [L0 train_model.py:Entropy](../../../../AllModels/Naru/train_model.py) 的详细解释。
def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    L.info(s)
    return ret

# ================================================================
# RunEpoch: 跑一个 epoch (train 或 test)
# ================================================================
# 跟 L0 RunEpoch 几乎逐行一致, 唯一区别是 args 从函数参数传入 (L0 用 module-level
# global)。包含: LR warmup / forward 多 ordering / NLL 三分支 / backward。
# 完整 ML 解释见 [L0 train_model.py:RunEpoch](../../../../AllModels/Naru/train_model.py):
#   - Noam warmup schedule
#   - 多 ordering ensemble 的 logsumexp 平均
#   - NLL = Σᵢ cross_entropy(p(xᵢ|x<ᵢ), data[:,i])
def RunEpoch(args,
             split,
             model,
             opt,
             train_data,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == 'train':
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                    .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if (step+1) % log_every == 0:
            if split == 'train':
                L.info(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num+1, step+1, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))
            else:
                L.info('{} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(split, step+1, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            L.info('%s epoch average loss: %f' % (split, np.mean(losses)))

    if return_losses:
        return losses
    return np.mean(losses)

# ================================================================
# InvertOrder: 求 permutation 的逆 (同 L0)
# ================================================================
# MADE 内部用 "position → natural_idx", 用户传的 --order 是
# "natural_idx → position", 互为逆置换。
# 详见 [L0 train_model.py:InvertOrder](../../../../AllModels/Naru/train_model.py).
def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


# ================================================================
# MakeMade: 构造 MADE/ResMADE 模型 (跟 L0 几乎一致)
# ================================================================
# L2 跟 L0 的具体差异:
#   - cols_to_train: 现在传 list(table.columns.values()) (lecarb Column 字典)
#   - c.vocab_size 替代 L0 的 c.DistributionSize()
#   - 新增 embed_threshold=args.embed_threshold 参数 (L2 only, 给 AutoEncode 用)
#   - 新增 epoch=args.epochs 参数 (L2 only, MADE 内部记录, 给 ResMADE 调度用)
def MakeMade(args, scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        L.info('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.vocab_size for c in cols_to_train]),
        input_bins=[c.vocab_size for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        #  embed_size=32,
        embed_size=args.embed_size,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        embed_threshold=args.embed_threshold,
        epoch=args.epochs
    ).to(DEVICE)

    return model


# ================================================================
# MakeTransformer: 同 L0, 只改 c.DistributionSize → c.vocab_size
# ================================================================
def MakeTransformer(args, cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.vocab_size for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


# ================================================================
# InitWeight: 跟 L0 一字不差
# ================================================================
# Linear / MaskedLinear → Xavier uniform; Embedding → N(0, 0.02²)。
# 见 [L0 train_model.py:InitWeight](../../../../AllModels/Naru/train_model.py).
def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)

# ================================================================
# train_naru: 训练 entry point (替代 L0 的 TrainTask)
# ================================================================
# 由 lecarb CLI 触发: `lecarb train --estimator naru --dataset DS --version V --workload W --params P --seed S`
# 跟 L0 TrainTask 的主要差别 (按顺序):
#   1. 强制 NUM_THREADS (公平 benchmark)
#   2. load_table(dataset, version) 替代 datasets.LoadDmv()
#   3. load_queryset + load_labels 拿到 valid query (训练完顺手算一遍 q-error 监控)
#   4. params dict → Args object (替代 argparse)
#   5. MakeMade 强制 seed=0 → natural_ordering=True (注释 "force natrual_ordering=True")
#      —— 这是 L2 的一个明显配置选择: 默认禁用 random ordering
#   6. sizelimit 检查: 模型大小超过 table 数据大小的 sizelimit 倍就 abort
#      (= 控制 model footprint, 避免训出比数据还大的 model 失去意义)
#   7. NaruTableDataset 替代 common.TableDataset
#   8. table_bits 用 df.groupby(list(df.columns)).size() 算 (= 同 L0 算法,
#      只是写法换了)
#   9. 训练完后跑 valid_queries 的 q-error → metrics, 存进 state 字典
#  10. state 字典存的字段比 L0 多得多 (见前面 L2 新增功能清单)
def train_naru(seed, dataset, version, workload, params, sizelimit):
    # uniform thread number
    # 强制 torch 线程数 = NUM_THREADS, 让 benchmark 公平 (所有 estimator 在同样线程预算下比)。
    torch.set_num_threads(NUM_THREADS)
    L.info(f"torch threads: {torch.get_num_threads()}")
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()

    # 用传入的 seed (不像 L0 硬编码 seed=0), 让多 seed 实验可控。
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 用 lecarb 统一 loader; version 让同一数据集可有多个版本 (e.g. 原版 / data shift 后的版本)。
    table = load_table(dataset, version)

    # load validation queries and labels
    # 训练时同时载入 valid query, 训练完用它算 q-error 监控 (在线评估)。
    # [:VALID_NUM_DATA_DRIVEN] 截前 N 个 (默认配置, 控制 valid 开销)。
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    # convert parameter dict to original naru code format
    # dict → Args object, 把 dict.key 写法换成 args.key (兼容拷过来的 L0 代码)。
    L.info(f"params: {params}")
    args = Args(**params)

    fixed_ordering = None
    if args.order is not None:
        L.info(f"Using passed-in order: {args.order}")
        fixed_ordering = args.order

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=fixed_ordering,
                                seed=seed)
    else:
        # 注意这里强制 seed=0, 不用外面传的 seed。
        # 理由: seed=0 在 MADE.__init__ 里走 natural_ordering=True 分支, L2 默认禁用
        # random ordering, 防止 mask cycle 引入额外变量 (debug 时更可控)。
        model = MakeMade(
            args,
            scale=args.fc_hiddens,
            cols_to_train=list(table.columns.values()),
            seed=0, # force natrual_ordering=True
            fixed_ordering=fixed_ordering)

    mb = report_model(model)
    # ========= sizelimit 检查 =========
    # 控制 model footprint: 模型不能比数据本身大太多 (否则相当于 "记忆" 整张表, 无意义)。
    # 例 sizelimit=0.1 → model 不超过 data 大小的 10%。
    # ARELY 的 benchmark 标准设置, 让 model 和 baseline (e.g. histogram) 大小可比。
    if sizelimit > 0 and mb > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mb:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue training!")
        return

    if not isinstance(model, transformer.Transformer):
        L.info('Applying InitWeight()')
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    L.info(f"start building naru dataset for table {table.name}...")
    train_data = NaruTableDataset(table)
    L.info("dataset build finished")

    L.info('calculate table entropy...')
    df = pd.DataFrame(data=train_data.tuples_np)
    table_bits = Entropy(
        table.name,
        df.groupby(list(df.columns)).size(), [2])[0]

    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunEpoch(
            args,
            'train',
            model,
            opt,
            train_data=train_data,
            val_data=train_data,
            batch_size=args.bs,
            epoch_num=epoch,
            log_every=200,
            table_bits=table_bits)

        dur_min = (time.time() - train_start) / 60
        L.info(f'epoch {epoch+1} train loss {mean_epoch_train_loss:.4f} nats / {mean_epoch_train_loss/np.log(2):.4f} bits, time since start: {dur_min:.1f} mins')

    dur_min = (time.time() - train_start) / 60
    L.info('Training finished! Time spent since start: {:.1f} mins'.format(dur_min))

    L.info('Evaluating likelihood on full data...')
    all_losses = RunEpoch(
        args,
        'test',
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        batch_size=args.bs,
        log_every=200,
        table_bits=table_bits,
        return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    # ========= 在线 valid: 跑 VALID_NUM_DATA_DRIVEN 个 query 拿 q-error =========
    # hardcode psample=2000 (= paper 标准设置); 评估只是看趋势, 不需要可变。
    # 注意: 这一步只是训练完打印监控数字, 训练已经结束了 (跟最终 test_naru 是两回事)。
    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = Naru(model,
                     'valid',
                     table,
                     2000, # hardcode 2000 psample for evaluation
                     device=DEVICE,
                     shortcircuit=args.column_masking)
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    # evaluate: lecarb 标准 q-error 分位数算法。
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    # ========= 保存全量 state =========
    # L0 只保存 model.state_dict(), L2 保存全状态:
    #   model_state_dict / optimizer_state_dict (给 update_naru 增量微调用)
    #   train_time / model_size / seed / args 等元数据 (paper 报表 / 复现需要)
    #   table_bits / model_bits (= 评估指标)
    #   valid_error (训练完算出来的在线 q-error metrics)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_time': dur_min,
        'model_size': mb,
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'table_bits': table_bits,
        'model_bits': model_bits,
        'valid_error': {workload: metrics}
    }

    # 标准路径: MODEL_ROOT/{dataset}/{version}-{model.name()}_warm{warmups}-{seed}.pt
    # model.name() 已经编码 fc_hiddens / layers / encoding 等所有超参 (见 made.py:name)。
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-{model.name()}_warm{args.warmups}-{seed}.pt"
    torch.save(state, model_file)
    L.info(f'model saved to:{model_file}')

# ================================================================
# Naru: lecarb-compatible progressive sampling estimator
# ================================================================
# 替代 L0 [estimators.py:ProgressiveSampling](../../../../AllModels/Naru/estimators.py#L134)。
# 算法 100% 一样 (按列截断采样 + ∏Zᵢ 是 sel(R) 无偏估计 — 完整证明
# 见 L0 那边的注释)。区别仅在框架接口:
#
# 接口层差异
# ----------------------------------------------------------------
#   L0 .Query(columns, operators, vals) -> int
#   L2 .query(query) -> (int, dur_ms)
#     query 是 lecarb Query 对象, 内部 query_2_triple 拆三元组
#     返回多带一个 dur_ms (本次 query 耗时, 给 paper 报 inference latency 用)
#
# 列访问差异
# ----------------------------------------------------------------
#   L0 table.columns[i].all_distinct_values  →  L2 self.table.columns[name].vocab
#   L0 col.DistributionSize()                →  L2 col.vocab_size
#   L0 table.cardinality                     →  L2 table.row_num
#   L0 columns 是 list 按 idx 取                →  L2 columns 是 dict 按 name 取
#
# return_probs (L2-only debug 选项)
# ----------------------------------------------------------------
# _sample_n / query 新增 return_probs=True 模式, 把每步的 softmax 后概率
# 也吐出来。给 paper 做 "Naru 输出分布合不合理" 的可解释性分析用 (L0 没有)。
class Naru(Estimator):
    """Progressive sampling from Naru."""
    def __init__(
            self,
            model,
            model_name,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
        # 调 lecarb Estimator 基类: 记录 table / model 标签 / psample 等元数据。
        super(Naru, self).__init__(table=table, model=model_name, psample=r)
        # 推理全程关 autograd (跟 L0 一致)。
        torch.set_grad_enabled(False)
        self.model = model
        self.shortcircuit = shortcircuit

        # r 的语义二分 (跟 L0 一致, 注意这里用 < 而不是 ≤): 比例 vs 固定数。
        # 见 [L0 ProgressiveSampling.__init__](../../../../AllModels/Naru/estimators.py#L137).
        if r < 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        # row_num 是 lecarb Table 的总行数字段 (= L0 table.cardinality)。
        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.row_num

        # 预算 init_logits (= 全零 input forward 一次, 给采样起点)。同 L0。
        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        # 各列 domain 大小; lecarb Column 用 vocab_size, L0 用 DistributionSize()。
        self.dom_sizes = [c.vocab_size for c in self.table.columns.values()]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.
        # 以下都是推理加速 trick, 跟 L0 完全一样:
        #   - MaskedLinear.masked_weight 预乘缓存
        #   - 冻结所有参数 (detach_ + requires_grad = False)
        #   - 预分配 kZeros + inp buffer (避免每次 query 重新 alloc)
        # 详细解释见 [L0 ProgressiveSampling.__init__](../../../../AllModels/Naru/estimators.py#L208) 的注释。

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        L.info('Setting masked_weight in MADE, do not retrain!')
        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)

    # ============================================================
    # _sample_n: progressive sampling 主循环 (= L0 _sample_n + return_probs)
    # ============================================================
    # 跟 L0 [estimators.py:206](../../../../AllModels/Naru/estimators.py#L206) 一字不差, 除了:
    #   - columns[natural_idx] 现在是 *列名字符串*, 不是 Column 对象
    #     所以 self.table.columns[<name>] 二次查 → 真正的 lecarb Column
    #   - col.vocab 替代 L0 col.all_distinct_values
    #   - col.has_nan 触发 SQL NULL 语义的 assert (vocab[0] 必须是 NaN, valid_i[0]=False)
    #     —— 这是 L2 多一道防御性检查 (lecarb 强约束 NaN 一定在 vocab 第 0 位)
    #   - return_probs=True 时把每步 softmax 输出收集进 all_probs (debug 用)
    # 完整的算法解释 (Z_i 是无偏估计的证明、4 个 Step 等) 见 L0 那边的注释。
    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None,
                  return_probs=False):
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []
        all_probs = [] # for analysis of naru's output

        # Use the query to filter each column's domain.
        # 跟 L0 同样的 Step 0: 每列预算 valid_i mask = 𝟙[v ∈ Rᵢ]。
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]

            # Column i.
            # L2 改动: columns[natural_idx] 是列名字符串 (lecarb 抽象), 要再去 dict 查 Column 对象。
            col = self.table.columns[columns[natural_idx]]
            op = operators[natural_idx]
            if op is not None:
                # There exists a filter.
                valid_i = OPS[op](col.vocab,
                                  vals[natural_idx]).astype(np.float32,
                                                            copy=False)
                # L2 新增防御性 assert: vocab 长度对得上 (lecarb Column 不变性检查)。
                assert len(valid_i) == len(col.vocab), valid_i
                # Comparing with NaN will always be False
                # L2 关键 invariant 检查: 如果列含 NaN, NaN 一定在 vocab[0] 位置 + 所以
                # 任何 op 比较 NaN 都返回 False (跟 SQL NULL 语义对齐, 见 common.py 那边的注释)。
                assert not col.has_nan or not valid_i[0], col
            else:
                continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]
                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]])
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r])

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        # 主循环跟 L0 一字不差。完整 4-step (softmax / mask / sum=Zᵢ / multinomial)
        # 解释 + ∏Zᵢ 无偏估计的展开证明见 [L0 estimators.py:257](../../../../AllModels/Naru/estimators.py#L257).
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            # L2 新增 return_probs 分支: 每步收集 *未截断的* softmax 概率,
            # 给可解释性分析用 (= 看 Naru 模型本身觉得每列分布啥样, 跟数据真实分布比对)。
            if return_probs:
                all_probs.append(torch.softmax(self.model.logits_for_col(natural_idx, logits), 1))

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or operators[natural_idx] is not None:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)
                #  L.debug(f"{i},{probs_i.shape}, {probs_i}")

                valid_i = valid_i_list[i]
                if valid_i is not None:
                    probs_i *= valid_i

                probs_i_summed = probs_i.sum(1)

                masked_probs.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                #  L.debug(f"{i}, {probs_i}")

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.shortcircuit and operators[natural_idx] is None:
                    data_to_encode = None
                else:
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]
                    data_to_encode = samples_i.view(-1, 1)

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    if not isinstance(self.model, transformer.Transformer):
                        if natural_idx == 0:
                            self.model.EncodeInput(
                                data_to_encode,
                                natural_col=0,
                                out=inp[:, :self.model.
                                        input_bins_encoded_cumsum[0]])
                        else:
                            l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                     - 1]
                            r = self.model.input_bins_encoded_cumsum[
                                natural_idx]
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                    else:
                        # Transformer.  Need special treatment due to
                        # right-shift.
                        l = (natural_idx + 1) * self.model.d_model
                        r = l + self.model.d_model
                        if i == 0:
                            # Let's also add E_pos=0 to SOS (if enabled).
                            # This is a no-op if disabled pos embs.
                            self.model.EncodeInput(
                                data_to_encode,  # Will ignore.
                                natural_col=-1,  # Signals SOS.
                                out=inp[:, :self.model.d_model])

                        if transformer.MASK_SCHEME == 1:
                            # Should encode natural_col \in [0, ncols).
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                        elif natural_idx < self.model.nin - 1:
                            # If scheme is 0, should not encode the last
                            # variable.
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])

                # Actual forward pass.
                # L2 跟 L0 微调: 多了 "and return_probs is False" 条件。
                # 原因: return_probs 模式想看下一列的 *真实* probs, 即使下一列是
                # wildcard 也得做这次 forward; 不开 return_probs 时仍然 skip 省 forward。
                next_natural_idx = i + 1 if ordering is None else ordering[i +
                                                                           1]
                if self.shortcircuit and operators[next_natural_idx] is None and return_probs is False:
                    # If next variable in line is wildcard, then don't do
                    # this forward pass.  Var 'logits' won't be accessed.
                    # But if we want to see the true probability predication for next column
                    # we have to run forward
                    continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    logits = self.model.do_forward(inp, ordering)
                else:
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp)
                    else:
                        logits = self.model.forward_with_encoded_input(inp)

        # deal with no predicates or one predicate
        # L2 新增两个边界情况处理 (L0 直接 return masked_probs[1] 起步, 0/1 列时崩溃):
        #   - 0 个 predicate 列 (全 wildcard): sel = 1, 返回整张表
        #   - 1 个 predicate 列: 只有 masked_probs[0], 直接平均返回
        #  print(masked_probs)
        if len(masked_probs) == 0:
            return 1, all_probs
        elif len(masked_probs) == 1:
            return masked_probs[0].mean().item(), all_probs
        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_probs[1]
        for ls in masked_probs[2:]:
            p *= ls
        p *= masked_probs[0]

        # L2 跟 L0 多返回 all_probs (L0 是单返回 scalar)。
        return p.mean().item(), all_probs

    # ============================================================
    # query: 对外接口, 替代 L0 ProgressiveSampling.Query()
    # ============================================================
    # lecarb Estimator 基类强制的接口签名: .query(query) → (card, dur_ms)。
    # query 是一个 lecarb Query 对象, 内部用 query_2_triple 拆出 (cols, ops, vals)。
    # cols 现在是列名字符串列表 (跟 L0 的 Column 对象列表不同)。
    # 完整流程 (ordering 选择 + 多 ordering averaging + 取 ceil) 见 L0 [estimators.py:Query](../../../../AllModels/Naru/estimators.py#L365).
    def query(self, query, return_probs=False):
        # Massages queries into natural order.
        # query_2_triple: lecarb 标准 query 拆解工具。
        # with_none=True 让没 predicate 的列补 None (= L0 FillInUnqueriedColumns)。
        columns, operators, vals = query_2_triple(query, with_none=True)

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            L.info('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            # 单 ordering 快路径 (跟 L0 一致, 多了 dur_ms 计时和可选 all_probs 返回)。
            # 注意 L0 用 np.ceil 取上整, L2 改成 np.round 四舍五入 —— 微妙差异:
            # ceil 保证估计 ≥1 (避免 q-error 分母 0), round 跟概率几何意义更对齐。
            if num_orderings == 1:
                ordering = orderings[0]
                start_stmp = time.time()
                p, all_probs = self._sample_n(
                    self.num_samples,
                    ordering if isinstance(
                        self.model, transformer.Transformer) else inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf,
                    return_probs=return_probs)
                dur_ms = (time.time() - start_stmp) * 1e3
                if return_probs:
                    return np.round(p * self.cardinality).astype(dtype=np.int32,
                                                                copy=False), dur_ms, all_probs
                return np.round(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False), dur_ms

            # Num orderings > 1.
            ps = []
            start_stmp = time.time()
            for ordering in orderings:
                p_scalar, all_probs = self._sample_n(self.num_samples // num_orderings,
                                          ordering, columns, operators, vals, return_probs=return_probs)
                ps.append(p_scalar)
            dur_ms = (time.time() - start_stmp) * 1e3
            if return_probs:
                return np.round(np.mean(ps) * self.cardinality).astype(
                    dtype=np.int32, copy=False), dur_ms, all_probs
            return np.round(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False), dur_ms

# ================================================================
# load_naru: 从 checkpoint 重建 estimator (供外部模块导入用)
# ================================================================
# 流程: state 字典加载 → load_table(同 version) → 重建同结构 model →
# load_state_dict → 包成 Naru estimator。
# 比 L0 [eval_model.py:Main](../../../../AllModels/Naru/eval_model.py) 干净:
# L0 用正则解析文件名拿超参, L2 直接从 state['args'] 拿 (序列化 Args 对象)。
def load_naru(dataset: str, model_name: str, psample: int) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load model from {model_file} ...")
    # map_location=DEVICE: 在 GPU 上训, CPU 上加载也能用 (PyTorch 默认会强制对齐 device)。
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    # 向后兼容: 旧 checkpoint (L2 早期版本) 没有 embed_threshold 字段, 补默认 128。
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    estimator = Naru(model,
                     model_name,
                     table,
                     psample,
                     device=DEVICE,
                     shortcircuit=args.column_masking)

    L.info(f"load and built naru estimator: {estimator}")
    return estimator, state


# ================================================================
# test_naru: 评估 entry point (替代 L0 eval_model.py:Main)
# ================================================================
# 由 lecarb CLI 触发: `lecarb test --estimator naru --dataset DS --workload W --params "{'model': ..., 'psample': ...}"`
# 比 L0 简单很多, 因为:
#   - 不用 glob 找 checkpoint (lecarb 直接传文件名)
#   - 不用正则解析 checkpoint 文件名拿超参 (从 state['args'] 直接读)
#   - 不用自己写 Query loop (用 run_test 通用工具)
#   - 不用自己构造 baseline (lecarb 每个 estimator 单独一个文件)
#
# params 字典格式 (由 lecarb CLI 传):
#   - model:   checkpoint 文件名 (不带 .pt 不带路径, 只是 stem)
#   - psample: 推理时的 progressive sample 数
def test_naru(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
        psample: number of progressive sample during each inference
    """
    # uniform thread number
    # 同 train_naru, 强制 NUM_THREADS 公平 benchmark。
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    estimator = Naru(model,
                     params['model'],
                     table,
                     params['psample'],
                     device=DEVICE,
                     shortcircuit=args.column_masking)

    L.info(f"load and built naru estimator: {estimator}")

    # init random seed before progressive sampling
    # 推理也固定 seed, 让 multinomial 采样可复现 (同一 query 多次跑结果一样)。
    torch.manual_seed(seed)
    np.random.seed(seed)

    # run_test: lecarb 通用测试 loop。内部:
    #   1. load_queryset(...) 拿 test query
    #   2. load_labels(...) 拿真值
    #   3. for q in queries: estimator.query(q) → 算 q-error
    #   4. 写 result CSV 到 RESULT_ROOT/{dataset}/{workload}/...
    # overwrite=True 覆盖已有结果文件。
    run_test(dataset, version, workload, estimator, overwrite)

# ================================================================
# update_naru: L2-only 新功能 — 在新版本数据上增量微调
# ================================================================
# 场景: data shift / online learning。表数据更新了 (version 变), 不想从头重训
# 一遍 (慢), 而是从 *现有 checkpoint* 继续训几个 epoch (= "fine-tune on
# new data")。L0 完全没有这功能, 这是 L2 加的新能力。
#
# 流程
# ----------------------------------------------------------------
# 1. load 旧 checkpoint (含 model_state_dict + optimizer_state_dict)
# 2. load_table(new_version) 拿新数据
# 3. 用旧 args 重建同结构 model + Adam optimizer
# 4. model.load_state_dict + opt.load_state_dict (= 恢复完整训练状态, 包括 Adam 的二阶 momentum)
# 5. 跑 args.epochs 个 epoch (一般 1-5 个, 比从头训快 10-100x)
# 6. 同 train_naru 一样: 算 valid q-error → 保存新 checkpoint
#
# 跟 train_naru 的区别 (除了 load 旧 state):
#   - 默认 epochs=1 (微调通常不需要多 epoch)
#   - 多存 update_time 元数据
#   - 不做 sizelimit 检查 (model 大小没变)
def update_naru(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    # args: 旧 checkpoint 里的 args (= 训练时配置), 微调要保持模型结构一致, 所以用旧的。
    args = state['args']
    # new_args: 用户这次 update 传的新参数 (主要用来覆盖 epochs)。
    new_args = Args(**params)
    # 默认微调 1 epoch (fine-tune 不需要多), 用户可通过 params['epochs'] 覆盖。
    epochs = 1
    if new_args.epochs:
        epochs = new_args.epochs
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128
    
    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    # load new version of table
    # 关键: version 是 *新* 版本 (data shift 后的), 但 args 是 *旧* 配置 → model 结构按旧的来。
    table = load_table(dataset, version)

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    L.info(f"start building naru dataset for table {table.name}...")
    train_data = NaruTableDataset(table)
    L.info("dataset build finished")

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)
    # 关键: 恢复 optimizer state (包含 Adam 的一阶/二阶 momentum buffers)。
    # 不恢复会让微调 "冷启动" optimizer, 前几步不稳定。这就是为什么 train_naru
    # 要把 optimizer_state_dict 也存进 checkpoint。
    opt.load_state_dict(state['optimizer_state_dict'])

    L.info('calculate table entropy...')
    df = pd.DataFrame(data=train_data.tuples_np)
    table_bits = Entropy(
        table.name,
        df.groupby(list(df.columns)).size(), [2])[0]
    
    update_start = time.time()
    for epoch in range(epochs):
        mean_epoch_train_loss = RunEpoch(
                args,
                'train',
                model,
                opt,
                train_data=train_data,
                val_data=train_data,
                batch_size=args.bs,
                epoch_num=epoch,
                log_every=200,
                table_bits=table_bits)
    dur_min = (time.time() - update_start) / 60
    L.info(f'Update train loss {mean_epoch_train_loss:.4f} nats / {mean_epoch_train_loss/np.log(2):.4f} bits, time since start: {dur_min:.4f} mins')
        
    L.info('Evaluating likelihood on full data...')
    all_losses = RunEpoch(
        args,
        'test',
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        batch_size=args.bs,
        log_every=200,
        table_bits=table_bits,
        return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = Naru(model,
                     'valid',
                     table,
                     2000, # hardcode 2000 psample for evaluation
                     device=DEVICE,
                     shortcircuit=args.column_masking)
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    new_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_time': state['train_time'],
        'model_size': state['model_size'],
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'table_bits': table_bits,
        'model_bits': model_bits,
        'valid_error': {workload: metrics},
        'update_time': dur_min
    }
    model.epoch = epochs
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-{model.name()}_warm{args.warmups}-{seed}.pt"
    torch.save(new_state, model_file)
    L.info(f'model saved to:{model_file}')