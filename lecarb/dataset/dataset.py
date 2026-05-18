# ================================================================
# 教学注释 (annotation pass) — dataset.py 总览
# ================================================================
# lecarb 的数据抽象层。所有 estimator (Naru / DeepDB / MSCN / Bayes net / MHist / ...)
# 都通过这套统一接口拿数据, 不直接操作 CSV / numpy。
#
# 跟 Naru L0 [common.py](../../../AllModels/Naru/common.py) 的对应
# ----------------------------------------------------------------
#   L0 Column.all_distinct_values          → L2 Column.vocab
#   L0 Column.distribution_size            → L2 Column.vocab_size
#   L0 Column.SetDistribution() + Fill()   → L2 Column.__init__ 一次完成
#   L0 Column.ValToBin(v)                  → L2 Column.discretize([v])[0]
#   L0 Discretize(col) (module-level)      → L2 col.discretize(data) (method)
#   L0 Table.cardinality                   → L2 Table.row_num
#   L0 CsvTable(name, file, cols, ...)     → L2 Table(dataset, version) (用 DATA_ROOT)
# 接口对外: estimator 仍按 column.vocab / column.vocab_size / table.row_num 操作。
#
# 数据流
# ----------------------------------------------------------------
# 1. CSV → gen_dataset.py 处理后存 → DATA_ROOT/{dataset}/{version}.pkl (DataFrame)
# 2. load_table(dataset, version): 第一次调时构造 Table + 推每列 vocab + dump 到
#    {version}.table.pkl (cache); 第二次直接 unpickle, 省去 vocab 解析
# 3. estimator 拿到 table 后用: table.columns[name].vocab / table.data / table.row_num
#
# 文件结构
# ----------------------------------------------------------------
#   - Column                : 单列抽象 (vocab + discretize + normalize)
#   - Table                 : DataFrame 包装 + 列字典 + mutual info 工具
#   - dump_table / load_table : 序列化 / 反序列化 Table (带 cache)
#   - dump_table_to_num     : 把 Table 离散化后导出 CSV (给 QuickSel / 外部工具用)
# ================================================================
import os
import copy
import logging
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
# sklearn.mutual_info_score: 两列离散变量的互信息 (= 列相关性度量)
# scipy.stats.entropy: 单列熵 (= 信息量)
# 都用在 get_max_muteinfo_order: 给 autoregressive estimator 找好的列顺序。
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

from ..constants import DATA_ROOT, PKL_PROTO
from ..dtypes import is_categorical

L = logging.getLogger(__name__)

# ================================================================
# Column: 单列抽象 (= L0 common.Column 的对应)
# ================================================================
# Write-once 语义: __init__ 解析 vocab 一次, 之后只读。
# 关键字段:
#   - name      : 列名
#   - dtype     : numpy dtype (categorical / numerical, 由 dtypes.py 判断)
#   - vocab     : sorted unique 值 array (NaN 永远放第 0 位, 跟 L0 同约定)
#   - vocab_size: len(vocab) = domain 大小
#   - minval    : 最小值 (跳过 NaN); 给 normalize 用
#   - maxval    : 最大值
#   - has_nan   : 数据里是否有 NaN/NaT (= vocab[0] 是否是 NaN)
class Column(object):
    def __init__(self, name, data):
        self.name = name
        self.dtype = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        self.minval = self.vocab[1] if self.has_nan else self.vocab[0]
        self.maxval = self.vocab[-1]

    def __repr__(self):
        return f'Column({self.name}, type={self.dtype}, vocab size={self.vocab_size}, min={self.minval}, max={self.maxval}, has NaN={self.has_nan})'

    # ============================================================
    # __parse_vocab: 解析数据得到 sorted unique vocab + NaN 放第 0 位
    # ============================================================
    # 跟 L0 common.SetDistribution 同算法 (NaN 摘出 → sort+unique → 回填 NaN 第 0 位)。
    # 为什么 NaN 放第 0 位: 跟 SQL NULL 语义对齐 (NaN <op> any → False, 进 estimator
    # 的 mask 时自动被排除, 详见 L0 common.py 教学注释)。
    def __parse_vocab(self, data):
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(data)
        contains_nan = np.any(is_nan)
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    # ============================================================
    # discretize: 把 raw value 转成 bin_id (= L0 Discretize)
    # ============================================================
    # pd.Categorical(data, categories=vocab).codes:
    #   - 输入数据 data, 用 vocab 作为映射表
    #   - 返回每个值对应 vocab 中的 index (= bin_id), 不在 vocab 内的得 -1
    # NaN trick: pd.Categorical 不接受 categories 含 NaN → 摘除 vocab[0] (= NaN),
    #            data 中 NaN 得 -1, 整体 +1 把 -1 修复成 0 (= 预期 NaN bin_id)。
    def discretize(self, data):
        """Transforms data values into integers using a Column's vocabulary"""

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values.
        if self.has_nan:
            bin_ids = pd.Categorical(data, categories=self.vocab[1:]).codes
            # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
            # add 1 to everybody
            bin_ids = bin_ids + 1
        else:
            # This column has no nan or nat values.
            bin_ids = pd.Categorical(data, categories=self.vocab).codes

        # int32 节省内存 (bin_id 一般 << 2^31)。
        bin_ids = bin_ids.astype(np.int32, copy=False)
        # 防御: 不应有 -1 残留 (NaN 路径已 +1 修复, 没 NaN 的列也不会出 -1)。
        assert (bin_ids >= 0).all(), (self, data, bin_ids)
        return bin_ids

    # ============================================================
    # normalize: 把 raw value 映射到 [0, 1]
    # ============================================================
    # 给 mhist (用 normalize 后的数据当 partition 边界) / 部分 NN-based estimator 用。
    # categorical 列先 discretize 成 bin_id, 再用 [0, vocab_size-1] 作 minmax 归一化。
    # numerical 列直接用 minval / maxval。
    def normalize(self, data):
        """Normalize data to range [0, 1]"""
        minval = self.minval
        maxval = self.maxval
        # if column is not numerical, use descretized value
        if is_categorical(self.dtype):
            data = self.discretize(data)
            minval = 0
            maxval = self.vocab_size - 1
        data = np.array(data, dtype=np.float32)
        if minval >= maxval:
            L.warning(f"column {self.name} has min value {minval} >= max value{maxval}")
            return np.zeros(len(data)).astype(np.float32)
        val_norm = (data - minval) / (maxval - minval)
        return val_norm.astype(np.float32)

# ================================================================
# Table: 一组 Column + 原始 DataFrame
# ================================================================
# 跟 L0 common.Table 的对应: row_num = cardinality, columns = OrderedDict 替代 list。
# (dataset, version) 二元组定位: 同数据集多版本支持 (e.g. census13 + original/v1/v2 给 data shift 实验用)。
# 持久化: pickle 整个 Table 对象 (含解析好的 Column vocab) 到 DATA_ROOT, 后续 load_table 快速反序列化。
class Table(object):
    def __init__(self, dataset, version):
        self.dataset = dataset
        self.version = version
        self.name = f"{self.dataset}_{self.version}"
        L.info(f"start building data {self.name}...")

        # load data
        self.data = pd.read_pickle(DATA_ROOT / self.dataset / f"{self.version}.pkl")
        self.data_size_mb = self.data.values.nbytes / 1024 / 1024
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        self.parse_columns()
        L.info(f"build finished: {self}")
    
    def parse_columns(self):
        self.columns = OrderedDict([(col, Column(col, self.data[col])) for col in self.data.columns])

    def __repr__(self):
        return f"Table {self.name} ({self.row_num} rows, {self.data_size_mb:.2f}MB, columns:\n{os.linesep.join([repr(c) for c in self.columns.values()])})"

    def get_minmax_dict(self):
        minmax_dict = {}
        for i, col in enumerate(self.columns.values()):
            minmax_dict[i] = (col.minval, col.maxval)
        return minmax_dict

    def normalize(self, scale=1):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            data[cname] = col.normalize(data[cname].values) * scale
        return data

    def digitalize(self):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            if is_categorical(col.dtype):
                data[cname] = col.discretize(data[cname])
            elif col.has_nan:
                data[cname].fillna(0, inplace=True)
        return data

    # ============================================================
    # get_max_muteinfo_order: 离线工具 — 找一个潜在有利的 ordering (未自动调用!)
    # ============================================================
    # 重要: **L2 Naru 训练流程不自动调它**。grep 全 lecarb 仅 __main__ 注释掉的
    # 调用引用了它 (line 293/297/302), 注释里 "7 1 8 6 5 9 0 4 3 2" 是 forest
    # 数据集上的输出结果 — 作者应该是早期手工跑过几次, 把结果手动复制到 args.order。
    # Naru.train_naru 实际用 args.order (用户传) 或 seed=0 自然顺序, 不调这函数。
    #
    # 启发式 (论文里提过, 但实测改进有限, 估计是没 wire 进去的原因):
    #   第一列选熵最大的 (= 最 "不可预测" 的, 优先建模);
    #   后续列依次选 "跟已选列互信息最大" 的 (= 信息增益大, 帮助预测下一列)。
    # 直觉: AR 模型 p(x1,...,xn) = ∏ p(xi | x<i), 把高熵列放前面、相关列连在一起
    # 让每步条件分布更确定。如果想试这个 ordering, 自己 import 调一下输出列名,
    # 再传给 `lecarb train --params "{'order': [4, 3, 2, ...]}"`。
    def get_max_muteinfo_order(self):
        order = []

        # find the first column with maximum entropy
        max_entropy = float('-inf')
        first_col = None
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            if e > max_entropy:
                first_col = c
                max_entropy = e
        assert first_col is not None, (first_col, max_entropy)
        order.append(first_col)
        sep = '|'
        chosen_data = self.data[first_col].astype(str) + sep

        # add the rest columns one by one by choosing the max mutual information with existing columns
        while len(order) < self.col_num:
            max_muinfo = float('-inf')
            next_col = None
            for c in self.columns.keys():
                if c in order: continue
                m = mutual_info_score(chosen_data, self.data[c])
                if m > max_muinfo:
                    next_col = c
                    max_muinfo = m
            assert next_col is not None, (next_col, max_entropy)
            order.append(next_col)
            # concate new chosen columns
            chosen_data = chosen_data + sep + self.data[next_col].astype(str)

        return order, [self.data.columns.get_loc(c) for c in order]

    def get_muteinfo(self, digital_data=None):
        data = digital_data if digital_data is not None else self.digitalize()
        muteinfo_dict = {}
        for c1 in self.columns.keys():
            muteinfo_dict[c1] = {}
            for c2 in self.columns.keys():
                if c1 != c2 and c2 in muteinfo_dict:
                    assert c1 in muteinfo_dict[c2], muteinfo_dict.keys()
                    muteinfo_dict[c1][c2] = muteinfo_dict[c2][c1]
                else:
                    muteinfo_dict[c1][c2] = mutual_info_score(data[c1], data[c2])
        return pd.DataFrame().from_dict(muteinfo_dict)

# ================================================================
# dump_table: 把 Table 对象 pickle 到磁盘 (= cache 解析好的 vocab)
# ================================================================
# 写入 {DATA_ROOT}/{dataset}/{version}.table.pkl, PKL_PROTO=4 支持大文件。
def dump_table(table: Table) -> None:
    with open(DATA_ROOT / table.dataset / f"{table.version}.table.pkl", 'wb') as f:
        pickle.dump(table, f, protocol=PKL_PROTO)

# ================================================================
# load_table: 加载 (或重新构造) Table
# ================================================================
# 优先从 .table.pkl cache 加载 (快); cache 不存在或 overwrite=True 时
# 从 {version}.pkl (DataFrame) 重新解析 vocab + dump 到 .table.pkl。
# 所有 estimator (e.g. test_naru / test_sample) 都通过这函数拿 table。
def load_table(dataset: str, version: str, overwrite: bool=False) -> Table:
    table_path = DATA_ROOT / dataset / f"{version}.table.pkl"

    if not overwrite and table_path.is_file():
        L.info("table exists, load...")
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        L.info(f"load finished: {table}")
        return table

    table = Table(dataset, version)
    L.info("dump table to disk...")
    dump_table(table)
    return table

# ================================================================
# dump_table_to_num: 把表 discretize 后导出 CSV (给外部工具用)
# ================================================================
# digitalize: categorical 列 → bin_id, numerical 列保持原值 (NaN fill 0)。
# 输出 {version}_num.csv 给 QuickSel / R 脚本 / 其它非 Python estimator 用。
def dump_table_to_num(dataset: str, version: str) -> None:
    table = load_table(dataset, version)
    num_data = table.digitalize()
    csv_path = DATA_ROOT / dataset / f"{version}_num.csv"
    L.info(f"dump csv file to {csv_path}")
    num_data.to_csv(csv_path, index=False)


if __name__ == '__main__':
    #  table = load_table('forest')
    #  print(table.get_max_muteinfo_order())
    # 7 1 8 6 5 9 0 4 3 2

    #  table = load_table('census')
    #  print(table.get_max_muteinfo_order())
    # 4 3 2 0 6 12 7 5 1 13 9 10 8 11

    table = Table('census', 'original')
    print(table)
    #  print(table.get_max_muteinfo_order())
    # 4 0 1 2 3 5 8 7 6
