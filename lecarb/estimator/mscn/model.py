"""
==============================================================
教学注释 (L2 — lecarb integration, MSCN model.py)
==============================================================

[本文件相对 L1 的改造概要 (vs Desktop/AllModels/0_AreCELearnedYet/MSCN/mscn/model.py)]
SetConv 网络结构 *与 L1 完全一致* (2-set, 删 join 分支). 改动只有 2 处:

  1. **删** `import numpy as np` (L1 size() 用 np.prod 算参数数; L2 改用字符串
     `name()` 不需要 numpy)
  2. **保存** sample_feats / hid_units 进 self (L1 没存, 因为 size() 不需要;
     L2 name() 把它们编进字符串)
  3. **`size()` → `name()`**: L1 是 `def size(self, blacklist=None) -> mb_float`,
     L2 是 `def name(self) -> str` 返回 "mscn_hid{H}_sample{S}". 用途也变了:
     - L1 size(): 训练时跟 sizelimit 比, 决定是否早退; .pt 文件名里也用到 size
     - L2 name(): 给 lecarb checkpoint 模板用 (eg. MODEL_ROOT/{dataset}/...-{name}-...)
     L2 通过 lecarb.estimator.utils 间接做 size 限制 (统一框架接管)

[设计动因]
lecarb 框架要求每个 estimator 实现一个 `Estimator` 基类 (见 lecarb/estimator/estimator.py),
该基类约定每个模型必须能产生 *唯一字符串名*, 这个名字嵌入 checkpoint 文件名 +
results 文件名. 所以 L2 用 name() 替代 size() — 后者是 ARELY 私有需求, 不通用.

[size 检查搬到哪去了]
L2 mscn.py:232-235 加了 size 限制的代码 (用 lecarb 的 NUM_THREADS / sizelimit /
table.data_size_mb), 模型本身不再操心. 这是 lecarb 集成 "把通用逻辑抽到框架,
模型只管自己" 的设计哲学.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model architecture
# removed all join related components since we only do cardinality estimation on single table

class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, hid_units):
        super(SetConv, self).__init__()
        # === L2 PATCH (vs L1): 把这两个超参存 self, name() 要用 ===
        # L1 没存 sample_feats / hid_units (因为 L1 size() 通过 named_parameters
        # 直接算参数数, 不用知道这两个数字). L2 name() 把它们编进字符串当模型 id.
        self.sample_feats = sample_feats
        self.hid_units = hid_units

        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 2, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, sample_mask, predicate_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid = torch.cat((hid_sample, hid_predicate), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

    # === L2 PATCH (vs L1 size()): name() 返回字符串 id ===
    # 返回格式 "mscn_hid{H}_sample{S}", eg. "mscn_hid256_sample1000".
    # 用途: lecarb checkpoint 文件名 + results 文件名都嵌入这个字符串, 让一个
    # dataset 下多 hyperparam sweep 不互相覆盖. 完整模板见 mscn.py:294 行附近.
    # f-string ('f' 前缀): Python 3.6+ 字符串插值, {var} 直接展开变量值.
    def name(self):
        return f"mscn_hid{self.hid_units}_sample{self.sample_feats}"
