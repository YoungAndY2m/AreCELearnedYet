# ================================================================
# 教学注释 (annotation pass) — mhist.py 总览
# ================================================================
# Multi-dimensional MaxDiff Histogram (MHIST-2 variant) estimator。
# 跟 Naru L0 [estimators.py:MaxDiffHistogram](../../../AllModels/Naru/estimators.py#L1249) 同算法
# (按 maxdiff 准则递归切桶, 每桶用 uniform spread + density 概述), 但代码
# **完全重写**, 不是搬运。
#
# 算法直觉 (跟 L0 同样, 完整解释见 L0 教学注释)
# ----------------------------------------------------------------
# 1. 初始: 1 个 partition 覆盖全表
# 2. 重复直到 partition 数到 num_bins:
#    - 找全局 maxdiff(V, A) = max(spread_i · count_i) 最大的 partition × col
#    - 在那个跳变点二分切开
# 3. 每个 partition 内部假设数据均匀, 用 uniform_spreads + density 表示
# 4. Query 时: 找跟 query 相交的 partitions, 各算 ∏overlap · density, 加起来
#
# 跟 L0 实现的代码结构差异 (= 完全 rewrite)
# ----------------------------------------------------------------
#   - L0 的 boundaries / uniform_spreads / col_value_list / density 4 个字段
#     → L2 合并成 1 个 meta list[5] per column + density
#   - L0 用整数位置索引 boundaries[cid][0], [1], [2]
#     → L2 用 IntEnum (M.LEFT / M.RIGHT / M.LEFT_IN / M.SPREAD / M.DISTINCT) 命名索引
#   - L0 的 _build_histogram 在 class 里
#     → L2 抽成 module-level 函数 construct_maxdiff (跟 lecarb load/save 风格一致)
#   - L0 的 maxdiff 用 dict + partition_to_maxdiff 反向索引 (复杂)
#     → L2 每个 partition 直接存 self.maxdiff 三元组 (area, cid, split_value) (简单)
#   - L2 新加 Estimation class (并行 worker 用, 但代码里被注释掉)
#   - L2 新加 load_mhist / 持久化 (用 pickle 存 partitions list, L0 没有 save/load)
#   - L2 用 lecarb table.normalize() 替代 L0 自己 discretize
#
# 文件结构
# ----------------------------------------------------------------
#   - M (IntEnum)         : meta tuple 字段命名 (LEFT/RIGHT/LEFT_IN/SPREAD/DISTINCT)
#   - Partition           : 一个桶, 含 meta + density + 算 maxdiff / split / query 方法
#   - Estimation          : 并行 query 计数器 (实际代码里没启用)
#   - MHist(Estimator)    : lecarb wrapper, 含 column_bound_map index 加速 query
#   - get_partition_num   : 由 size_limit 反算最大 bin 数
#   - get_hist_size       : 由 bin 数算 histogram 占多少 MB
#   - construct_maxdiff   : 主 build 函数 (= L0 _build_histogram)
#   - load_mhist          : 从 .pkl 加载已建好的 histogram
#   - test_mhist          : lecarb CLI entry point
# ================================================================
import enum
import copy
import logging
import pickle
import time
import threading
import bisect
from typing import Any, Dict, Tuple
import numpy as np

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple

L = logging.getLogger(__name__)

# ================================================================
# M (IntEnum): meta list 的字段名 (替代魔法数字 0/1/2/3/4)
# ================================================================
# Partition.meta[cid] 是 list of 5 元素, 用这些 enum 索引让代码可读:
#   meta[cid][M.LEFT]     - 该列在该 partition 的下界
#   meta[cid][M.RIGHT]    - 上界
#   meta[cid][M.LEFT_IN]  - 下界是否闭区间 (True = 包含 left)
#   meta[cid][M.SPREAD]   - "uniform spread" 步长 (假装均匀分布的代表点间距)
#   meta[cid][M.DISTINCT] - 该 partition 内该列的 distinct 值数
# L0 是直接用 boundaries[cid][0/1/2] + 单独 uniform_spreads 列表 + density 字段。
class M(enum.IntEnum):
    LEFT = 0
    RIGHT = 1
    LEFT_IN = 2
    SPREAD = 3
    DISTINCT = 4

# ================================================================
# Partition: 一个桶 (= 高维矩形区域)
# ================================================================
# 状态分两阶段:
#   build 阶段 (调 split_partition 时): 持有 self.data (真实行子集), 算 maxdiff
#   inference 阶段 (clean() 之后): self.data=None, 只保留 meta + density
# 这样推理时内存只装 metadata, 不装数据本身 (L2 比 L0 干净, L0 还存了 col_value_list)。
class Partition(object):
    def __init__(self, pid=0):
        self.pid = pid

        # for inference
        self.meta = [] # (left, right, include_left, spread_length, distinct)
        self.density = None

        # for construct
        self.data = None
        self.maxdiff = None # (area size, column, value)

    def __str__(self):
        if self.density is None:
            return f'{self.pid}: # : {len(self.data)}\nMaxDiff: {self.maxdiff}\nMetadata: {self.meta}'
        else:
            return f'{self.pid}: Density: {self.density}\nMetadata: {self.meta}'

    def clean(self):
        self.data = None
        self.maxdiff = None

    # ============================================================
    # construct_from_table: 初始化为覆盖全表的 root partition
    # ============================================================
    # table.normalize() 让数据归一化到 [0, 1] (跟 digitalize 不同, 保留连续值精度;
    # MHIST 算 spread = (right - left) / distinct 需要数值, 不能只是 bin_id)。
    # 每列的 meta 初始化为 [min, max, True (闭左界), None (spread 还没算), None (distinct)]。
    def construct_from_table(self, table):
        # normalize is better than only digitalize data
        self.data = table.normalize()
        #  self.data = table.digitalize()
        for c in self.data.columns:
            self.meta.append([self.data[c].min(), self.data[c].max(), True, None, None])

    # ============================================================
    # get_maxdiff: 算这个 partition 在所有列上的 maxdiff, 返回最大值
    # ============================================================
    # maxdiff(V, A) 公式 (跟 L0 同):
    #   对列 A 排序 distinct values v_1 < v_2 < ...
    #   spread_i = v_{i+1} - v_i (gap)
    #   count_i  = freq(v_i)
    #   maxdiff_A = max_i (spread_i · count_i)
    # 然后取所有列里最大的 maxdiff (= 最该切的 column 上最该切的位置)。
    # self.maxdiff 缓存 (cid, split_value, max_area), 避免重复计算。
    def get_maxdiff(self):
        # only need to calculate split point once if data dose not change
        if self.maxdiff is not None:
            return self.maxdiff[0]

        for cid, c in enumerate(self.data.columns):
            # value_counts 数频次, sort_index 按值排序 → 跟 L0 算法一致。
            counter = self.data[c].value_counts().sort_index()
            # areas[i] = spread_i · count_i (NumPy 向量化, 比 L0 的 .iloc[:-1] 写法更简洁)
            areas = counter.iloc[:-1] * (counter.index[1:] - counter.index[:-1])
            if len(areas) > 0:
                c_max = areas.max()
                # 全局最大 maxdiff: 在所有列里取最大。
                # 存 (max_area, cid, 切点 value) — split_partition 直接用这个 cid 和 value。
                if self.maxdiff is None or c_max > self.maxdiff[0]:
                    self.maxdiff = (c_max, cid, areas.idxmax())

        if self.maxdiff is None:
            self.maxdiff = (0, None, None)
        return self.maxdiff[0]

    # ============================================================
    # split_partition: 按 maxdiff 切点把 self 切成 p1 / p2
    # ============================================================
    # 切完后:
    #   p1: 列 cid 范围 [old_left, split], 闭右界 (= 包含 split 的行)
    #   p2: 列 cid 范围 (split, old_right], 开左界 (= 不包含 split, 已分给 p1)
    # 其它列的 boundary 不变 (深 copy 父 partition 的 meta)。
    def split_partition(self):
        if self.maxdiff is None:
            self.get_maxdiff()
        assert self.maxdiff is not None and self.maxdiff[0] > 0

        _, cid, split = self.maxdiff
        c = self.data.columns[cid]

        # p1: 左半 (<= split)
        p1 = Partition()
        p1.data = self.data[self.data[c] <= split]
        # deepcopy meta: 不能共享 list 引用, 否则改 p1 影响 p2。
        p1.meta = copy.deepcopy(self.meta)
        # 更新 split 列的边界: [old_left, split], left_in 继承父。
        p1.meta[cid] = [p1.meta[cid][0], split, p1.meta[cid][2], None, None]

        # p2: 右半 (> split)
        p2 = Partition()
        p2.data = self.data[self.data[c] > split]
        p2.meta = copy.deepcopy(self.meta)
        # 更新 split 列的边界: (split, old_right], left_in=False (开左界, 避免跟 p1 重叠)。
        p2.meta[cid] = [split, p2.meta[cid][1], False, None, None]

        return p1, p2

    # ============================================================
    # calculate_spread_density: 算 partition 内每列的 SPREAD + 全 partition 的 density
    # ============================================================
    # 算完用于推理阶段 (build 完之后调一次, 之后 self.data 就可以丢了)。
    # 详细公式见 L0 mhist 教学注释。
    def calculate_spread_density(self):
        total_distinct = 1 # product of # distinct of each column
        #  L.error(f'total data: {len(self.data)}')
        for cid in range(len(self.meta)):
            c = self.data.columns[cid]
            unique = self.data[c].unique()
            distinct = len(unique)
            self.meta[cid][M.DISTINCT] = distinct
            #  L.error(f'column: {cid}, distinct: {distinct}')
            # 边界情形: 该 partition 在该列只有 1 个值 → SPREAD 用 (val - LEFT) 表示
            # (= 单点宽度, 没多余空间分布)。
            if distinct == 1:
                self.meta[cid][M.SPREAD] = float(unique.item() - self.meta[cid][M.LEFT])
                continue
            # 正常: SPREAD = (RIGHT - LEFT) / (distinct - 1 or distinct)。
            # 闭左界: 用 distinct - 1 (代表点正好覆盖 LEFT 到 RIGHT 两端)
            # 开左界: 用 distinct (代表点从 LEFT + spread 起步, 不到 LEFT 本身)
            if self.meta[cid][M.LEFT_IN] is True:
                self.meta[cid][M.SPREAD] = float((self.meta[cid][M.RIGHT] - self.meta[cid][M.LEFT]) / (distinct - 1))
            else:
                self.meta[cid][M.SPREAD] = float((self.meta[cid][M.RIGHT] - self.meta[cid][M.LEFT]) / distinct)
            total_distinct *= distinct
        # density = 该 partition 总行数 / 所有列 distinct 的乘积。
        # 物理含义: "假设均匀分布的话, 每个 (列值组合) 期望出现多少行"。
        self.density = len(self.data) / total_distinct
        #  L.error(self)

    # ============================================================
    # Partition.query: 该 partition 对一个 query 的贡献估计
    # ============================================================
    # 流程:
    #   1. 对每个 predicate 列, 算 "该列 spread 代表点中有多少个落在 R 内" (c_covered)
    #   2. 没 predicate 的列: 贡献 = 该列总 distinct (= 全选)
    #   3. 总贡献 = ∏ c_covered · density
    # density 把 distinct 代表点数转换成预期行数 (因为 density = rows/∏distinct)。
    def query(self, columns, operators, values):
        # get_points_on_left: 算 "spread 代表点中 ≤ v (closed) 或 < v (open) 的个数"。
        # 代表点的位置: LEFT, LEFT+spread, LEFT+2·spread, ..., RIGHT。
        # closed: True 表示 v 处也算 (用于 <= / >=); False 用于 < / >。
        def get_points_on_left(v, closed=False):
            # v 在 [left, right] 外的边界情形:
            if v < left or (v == left and (not closed)):
                return 0
            if v > right or (v == right and closed):
                return distinct
            # 一般情形: 用 divmod 算 v 落在第几个 spread 后面。
            covered, remains = divmod((v - left), spread)
            # 浮点误差兜底: 如果 v 恰好落在某个代表点上且 open, 不算它。
            if not closed and remains < 1e-10:
                covered -= 1
            # +1 把 0-indexed 转 count (= "有几个代表点 ≤ v")。
            covered = int(covered) + 1
            return covered

        #  L.error('')
        #  L.error(self)
        #  L.error('')

        # 累乘 ∏ c_covered, 初值 1。
        total_covered = 1
        for cid, op, val in zip(columns, operators, values):
            # 解包 meta 五元组: 边界 + 闭开 + spread + distinct。
            left, right, left_in, spread, distinct = self.meta[cid]

            # if only has one point, get the true value
            # 边界情形 1: 该列只有 1 个值 → 把 left 调到那个点 (single-point partition)。
            if distinct == 1:
                left = right = left + spread
            # 边界情形 2: 开左界 → 实际第一个代表点是 left + spread, 不是 left 本身。
            elif not left_in:
                left += spread

            assert left <= right, f'{self.pid}-{cid}: {self.meta[cid]}'

            # 按 operator 分类计数 (= 用 get_points_on_left 的差值算 R 内的代表点数)。
            c_covered = None
            if op == '<':
                c_covered = get_points_on_left(val, closed=False)
            elif op == '<=':
                c_covered = get_points_on_left(val, closed=True)
            elif op == '>':
                # > val: 总 distinct 减掉 ≤ val 的数。
                c_covered = distinct - get_points_on_left(val, closed=True)
            elif op == '>=':
                # >= val: 总 distinct 减掉 < val 的数。
                c_covered = distinct - get_points_on_left(val, closed=False)
            elif op == '[]':
                # (>= val[0] and <= val[1]) -> (<= val[1]) - (< val[0])
                # range query: 上界包含计数 - 下界不包含计数。
                c_covered = get_points_on_left(val[1], closed=True)
                if c_covered > 0:
                    c_covered -= get_points_on_left(val[0], closed=False)
            elif op == '=':
                if val < left or val > right:
                    c_covered = 0
                else:
                    # if equal, cover 1 value
                    # MHIST 假设均匀, 所以 = 任何点都覆盖 1 个代表点 (这是已知缺陷, 见 paper 讨论)。
                    c_covered = 1
                # just like mentioned in Naru, it tends to underestimate
                #  elif (spread == 0 and val == right) or (val - left) % spread == 0:
                #      c_covered = 1
                #  else:
                #      c_covered = 0

            assert type(c_covered) == int and c_covered >= 0, f'{self.pid}-{cid}-{op}-{val}:{self.meta[cid]}, c_cover: {c_covered}'
            total_covered *= c_covered
            if total_covered == 0:
                break

        if total_covered == 0:
            return 0

        # ========= 没 predicate 的列贡献 = 该列 distinct 全选 =========
        # 例: 3 列 partition, query 只过滤 col 0 → total_covered *= distinct_1 * distinct_2。
        # 然后乘 density 转成预期行数。
        for cid in range(len(self.meta)):
            if not cid in columns:
                total_covered *= self.meta[cid][M.DISTINCT]
        return total_covered * self.density

# ================================================================
# Estimation: 多线程并行 query 时的累加器 (实际代码里没启用, 见 MHist.query 注释)
# ================================================================
# 把 candidate partitions 平分给 num_threads 个 thread, 各自累加进 card[tid]。
# 注: MHist.query() 里的多线程代码被注释掉了, 单线程跑 (Python GIL 让多线程
# 对 CPU bound numpy 操作没好处, 这设计是预留接口)。
class Estimation(object):
    def __init__(self, num_part, num_threads):
        self.card = np.zeros((num_threads))
        self.parts = np.array(range(num_threads)) * int(num_part / num_threads)
        self.parts = np.append(self.parts, num_part)

# ================================================================
# MHist: lecarb Estimator wrapper, 加 column_bound_map 加速 query
# ================================================================
# 跟 L0 类似, 用一组 bound map 做 query 时的候选 partition 范围裁剪 (避免
# 线性扫所有 partition)。详见 _get_valid_pids 和 _build_index。
class MHist(Estimator):
    def __init__(self, partitions, table):
        super(MHist, self).__init__(table=table, bins=len(partitions))
        self.partitions = partitions

        # index for faster inference (refer from Naru)
        # map<cid, map<bound_type, map<bound_value, list(partition id)>>>
        self.column_bound_map = {}
        for cid in range(self.table.col_num):
            self.column_bound_map[cid] = {}
            self.column_bound_map[cid]['l'] = {}
            self.column_bound_map[cid]['u'] = {}
        # map<cid, map<bound_type, sorted_list(bound_value)>>
        self.column_bound_index = {}
        for cid in range(self.table.col_num):
            self.column_bound_index[cid] = {}
            self.column_bound_index[cid]['l'] = []
            self.column_bound_index[cid]['u'] = []
        self._build_index()

    # ============================================================
    # _build_index: 构造 column_bound_map / column_bound_index 索引
    # ============================================================
    # 用途: query 时按 column predicate 快速找候选 partition 集合, 避免线性扫。
    # 结构:
    #   column_bound_map[cid]['l'][bound_value] = list of partition ids with that lower bound
    #   column_bound_map[cid]['u'][bound_value] = ditto for upper bound
    #   column_bound_index[cid]['l'] = sorted unique list of all lower bounds (二分查找用)
    #   column_bound_index[cid]['u'] = sorted unique list of all upper bounds
    # _get_valid_pids 用 bisect 在 column_bound_index 上二分定位, 拿到 partition id 集合。
    def _build_index(self):
        for cid in range(self.table.col_num):
            for pid, p in enumerate(self.partitions):
                if p.meta[cid][M.LEFT] not in self.column_bound_map[cid]['l']:
                    self.column_bound_map[cid]['l'][p.meta[cid][M.LEFT]] = [pid]
                else:
                    self.column_bound_map[cid]['l'][p.meta[cid][M.LEFT]].append(pid)

                if p.meta[cid][M.RIGHT] not in self.column_bound_map[cid]['u']:
                    self.column_bound_map[cid]['u'][p.meta[cid][M.RIGHT]] = [pid]
                else:
                    self.column_bound_map[cid]['u'][p.meta[cid][M.RIGHT]].append(pid)

                self.column_bound_index[cid]['l'].append(p.meta[cid][M.LEFT])
                self.column_bound_index[cid]['u'].append(p.meta[cid][M.RIGHT])
            self.column_bound_index[cid]['l'] = sorted(set(self.column_bound_index[cid]['l']))
            self.column_bound_index[cid]['u'] = sorted(set(self.column_bound_index[cid]['u']))

    # ============================================================
    # _get_valid_pids: 给一个 column predicate, 用 bound index 找候选 partition ids
    # ============================================================
    # 对 column cid 的每种 operator 都用二分:
    #   < / <=  : 找所有 "lower bound < val" 的 partition
    #   > / >=  : 找所有 "upper bound > val" 的 partition
    #   = / []  : 上下界互相 intersect
    # 返回 set, 之后跟其它列的 set 求 ∩ → 真正命中的 partitions。
    def _get_valid_pids(self, cid, op, val):
        if op in ['<', '<=']:
            valid_set = set()
            if op == '<':
                insert_index = bisect.bisect_left(self.column_bound_index[cid]['l'], val)
                for i in range(insert_index):
                    valid_set = valid_set.union(self.column_bound_map[cid]['l'][self.column_bound_index[cid]['l'][i]])
            else:
                insert_index = bisect.bisect(self.column_bound_index[cid]['l'], val)
                for i in range(insert_index):
                    if self.column_bound_index[cid]['l'][i] == val:
                        for pid in self.column_bound_map[cid]['l'][val]:
                            if self.partitions[pid].meta[cid][M.LEFT_IN]:
                                # add only when the lower bound is inclusive
                                valid_set.add(pid)
                    else:
                        valid_set = valid_set.union(self.column_bound_map[cid]['l'][self.column_bound_index[cid]['l'][i]])
            return valid_set

        if op in ['>', '>=']:
            valid_set = set()
            insert_index = None
            if op == '>':
                insert_index = bisect.bisect(self.column_bound_index[cid]['u'], val)
            else:
                insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'], val)
            for i in range(insert_index, len(self.column_bound_index[cid]['u'])):
                valid_set = valid_set.union(self.column_bound_map[cid]['u'][self.column_bound_index[cid]['u'][i]])
            return valid_set

        assert op in ['=', '[]'], op
        lower_v, upper_v = val if type(val) is tuple else (val, val)
        lower_bound_set = set()
        insert_index = bisect.bisect(self.column_bound_index[cid]['l'], upper_v)
        for i in range(insert_index):
            if self.column_bound_index[cid]['l'][i] == upper_v:
                for pid in self.column_bound_map[cid]['l'][upper_v]:
                    if self.partitions[pid].meta[cid][M.LEFT_IN]:
                        # add only when the lower bound is inclusive
                        lower_bound_set.add(pid)
            else:
                lower_bound_set = lower_bound_set.union(
                    self.column_bound_map[cid]['l'][
                        self.column_bound_index[cid]['l'][i]])

        upper_bound_set = set()
        insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'], lower_v)
        for i in range(insert_index, len(self.column_bound_index[cid]['u'])):
            upper_bound_set = upper_bound_set.union(
                self.column_bound_map[cid]['u'][self.column_bound_index[cid]['u'][i]])
        return lower_bound_set.intersection(upper_bound_set)

    # ============================================================
    # query_worker: 多线程并行 query 的 worker (实际没启用)
    # ============================================================
    # 给 tid 号 thread 分配 est.parts[tid:tid+1] 区间的 partition 跑 query, 累加到 est.card[tid]。
    # 因 Python GIL + 单 partition.query 已经是纯 numpy, 多线程没好处, 代码留着但被注释掉。
    def query_worker(self, tid, est, columns, operators, values, candidate_pids):
        for i in range(est.parts[tid], est.parts[tid+1]):
            est.card[tid] += self.partitions[candidate_pids[i]].query(columns, operators, values)

    # ============================================================
    # query: 主推理入口
    # ============================================================
    # 步骤:
    #   1. query_2_triple 拆 query, 取出 columns / ops / vals
    #   2. col.normalize(val) 把 raw value 转 normalized scale (对齐 partition meta 的边界)
    #   3. column 名字 → index (因为 Partition.query 用 cid)
    #   4. 用 _get_valid_pids 各列求 candidate, ∩ 得最终命中 partitions
    #   5. 对每个命中 partition 调 Partition.query, 累加 cardinality
    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        # descritize predicate parameters for non-numerical columns
        for i, predicate in enumerate(zip(columns, operators, values)):
            cname, op, val = predicate
            col = self.table.columns[cname]
            val = col.normalize(list(val) if type(val) is tuple else [val])
            values[i] = tuple(val) if len(val) > 1 else val.item()
            #  if is_categorical(col.dtype):
            #      val = col.discretize(list(val) if type(val) is tuple else [val])
            #      values[i] = tuple(val) if len(val) > 1 else val.item()
        # convert column names to indices
        columns = [self.table.data.columns.get_loc(c) for c in columns]
        start_stmp = time.time()

        # use index to find related partition ids
        candidate_pids = set(range(len(self.partitions)))
        for cid, op, val in zip(columns, operators, values):
            candidate_pids = candidate_pids.intersection(self._get_valid_pids(cid, op, val))

        #  query on each partition
        #  candidate_pids = list(candidate_pids)
        #  num_threads = NUM_THREADS if len(candidate_pids) > (NUM_THREADS * 10) else 1
        #  est = Estimation(len(candidate_pids), num_threads)
        #  for i in range(num_threads):
        #      t = threading.Thread(target=self.query_worker, args=(i, est, columns, operators, values, candidate_pids))
        #      t.start()

        #  main_thread = threading.currentThread()
        #  for t in threading.enumerate():
        #      if t is not main_thread:
        #          t.join()

        est_card = []
        for pid in candidate_pids:
            est_card.append(self.partitions[pid].query(columns, operators, values))

        dur_ms = (time.time() - start_stmp) * 1e3

        #  return np.round(est.card.sum()), dur_ms
        return np.round(np.sum(est_card)), dur_ms

# ================================================================
# get_partition_num: 给定 size_limit (MB), 反算最多能用多少 bin
# ================================================================
# 每个 partition 的存储成本估算:
#   - density: 1 个 float (4 bytes)
#   - 每列: LEFT + RIGHT + SPREAD (3 个 float) + LEFT_IN (1 byte) = 13 bytes
#   - DISTINCT 算进推理 metadata 不算 (注释里 17 那行包含 DISTINCT, 是另一版)
# 总: 4 + col_num · 13 bytes per partition。
# 跟其它 estimator (Naru / Sampling) 用同样 model size 预算对比 q-error。
def get_partition_num(col_num, size_limit_mb):
    # for each partition, we need record the follow information for each column:
    # density, left, right, spread_length: 4 bytes for each
    # include_left: 1 byte
    # do not count total of row number here since all method need to record this
    # 13 = 3 * 4 + 1
    # 17 = 4 * 4 + 1
    return int((size_limit_mb * 1024 * 1024) // (4 + col_num * 13))
    #  return int((size_limit_mb * 1024 * 1024) // (4 + col_num * 17))

# ================================================================
# get_hist_size: 反过来, 给定 num_bins 算 histogram 实际占多少 MB
# ================================================================
# 给 paper 报 model footprint 用。
def get_hist_size(num_bins, col_num):
    # for each partition, we need record the follow information for each column:
    # density, left, right, spread_length: 4 bytes for each
    # include_left: 1 byte
    # do not count total of row number here since all method need to record this
    return (num_bins * (col_num * 13 + 4)) / 1024 / 1024
    #  return (num_bins * (col_num * 17 + 4)) / 1024 / 1024

# ================================================================
# print_partitions: debug 用, 打印所有 partition (调试时用)
# ================================================================
def print_partitions(partitions):
    L.info('')
    for p in partitions:
        L.info(f'\n{p}')
    L.info('======================')

# ================================================================
# construct_maxdiff: 主构造函数 — 把 1 个 partition 递归切到 num_bins 个
# ================================================================
# = L0 MaxDiffHistogram._build_histogram 的 functional 重写。
# 输出 state 字典 (含 partitions list + metadata), 后续 pickle 存盘。
def construct_maxdiff(table, num_bins):
    partitions = []

    start_stmp = time.time()
    # ========= 主切桶 loop =========
    # 每次迭代:
    #   1. 第一次: 建 root partition 覆盖全表
    #   2. 后续: 找全局 maxdiff 最大的 partition → split 成两个新的
    # 如果 maxdiff = 0 (= 所有 partition 内每列都只有 1 个值, 完美均匀) 提前结束。
    for i in range(num_bins):
        if len(partitions) == 0:
            partitions.append(Partition())
            partitions[0].construct_from_table(table)
            continue

        # find the partition has maxdiff to split
        maxdiff = 0
        pid = None
        for i, p in enumerate(partitions):
            p_md = p.get_maxdiff()
            if p_md > maxdiff:
                maxdiff = p_md
                pid = i

        #  print_partitions(partitions)
        if maxdiff == 0:
            L.info('Maxdiff is 0 before reach partition limit!')
            break

        # pop 旧 partition, 切成 p1/p2, extend 回 list。
        # 注意 i 变量在内层循环被重用了 (= Python 闭包陷阱, 但功能上不影响)。
        p = partitions.pop(pid)
        p1, p2 = p.split_partition()
        partitions.extend([p1, p2])

        if (i+1) % 100 == 0:
            L.info(f'Constructed {i+1} partitions!')

    # ========= 切完后, 每个 partition 算 spread + density, 再 clean 掉 raw data =========
    # clean() 把 self.data / self.maxdiff 置 None, 推理时只保留 meta + density, 省内存。
    for p in partitions:
        p.calculate_spread_density()
        p.clean()
    hist_size = get_hist_size(len(partitions), len(table.columns))
    #  print_partitions(partitions)

    dur_min = (time.time() - start_stmp) / 60
    L.info(f'Construct MaxDiff Hist (MHIST-2) finished, use {len(partitions)} partitions ({hist_size:.2f}MB)! Time spent since start: {dur_min:.2f} mins')

    state = {
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': table.dataset,
        'version': table.version,
        'partitions': partitions,
        'train_time': dur_min,
        'model_size': hist_size,
    }
    return state

# ================================================================
# load_mhist: 从 .pkl 读已构造好的 histogram
# ================================================================
# MHIST 构造时间贵 (大表 + bin 多时几十分钟), 一次性 build 完 pickle 存,
# 之后 test 直接 load — 这是 lecarb 的标准 cache 模式 (Naru 用 .pt 同理)。
def load_mhist(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    table = load_table(dataset, state['version'])
    partitions = state['partitions']
    #  print_partitions(partitions)
    estimator = MHist(partitions, table)
    return estimator, state

# ================================================================
# test_mhist: lecarb CLI entry point
# ================================================================
# `lecarb test --estimator mhist --params "{'num_bins': 1000}"`
# 跟 train 不分开 (MHIST 没"训练"概念, 只有 construct), 但有 cache:
#   - 已有 .pkl 直接 load
#   - 没有就调 construct_maxdiff 现建, 然后 pickle 存
def test_mhist(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-mhist_bin{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct MHist with at most {params['num_bins']} bins...")
        state = construct_maxdiff(table, params['num_bins'])
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"MHist saved to {model_file}")

    partitions = state['partitions']
    #  print_partitions(partitions)
    estimator = MHist(partitions, table)
    L.info(f"Built MHist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)
