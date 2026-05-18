# ================================================================
# 教学注释 (annotation pass) — utils.py 总览
# ================================================================
# lecarb estimator 通用工具集:
#   - report_model       : 打印模型参数数 + MB (Naru / MSCN 等 NN-based 用)
#   - qerror             : q-error 公式 (= max(est,true)/min(est,true), 同 L0)
#   - rmserror           : RMS error 算法 (用 selectivity 而不是 cardinality, 给 paper 报指标用)
#   - evaluate           : 跑一遍 q-error 算 max/95th/99th/median/mean/gmean 分位数
#   - evaluate_errors    : 从已有 errors list 算分位数 (跟上面同但不重新算 q-error)
#   - report_errors      : 从 result CSV 读 errors 然后报分位数
#   - report_dynamic_errors: 模拟 "model 更新延迟" 的实验工具
#   - lazy_derive        : data shift 实验下, 复用旧 result + 缩放成新 version 的 result
#   - run_test           : 主测试 loop (所有 estimator 通过这跑 workload)
#
# 跟 Naru L0 的对应
# ----------------------------------------------------------------
#   L0 eval_model.py:RunN + Query + ReportEsts 三函数  → L2 run_test 一函数
#   L0 ReportModel                                     → L2 report_model
#   L0 ErrorMetric                                     → L2 qerror
# ================================================================
import csv
import ray
import logging
import numpy as np
import pandas as pd
import torch
# scipy.stats.mstats.gmean: 几何平均数。q-error 的几何平均比算术平均更稳定
# (geom mean 不被大 outlier 主导, 更代表 "典型" 表现)。
from scipy.stats.mstats import gmean

#  from .lw.lw_nn import LWNN
#  from .lw.lw_tree import LWTree
from .estimator import Estimator
from ..constants import NUM_THREADS, RESULT_ROOT
from ..workload.workload import load_queryset, load_labels
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

# ================================================================
# report_model: 打印模型参数数 + MB (= L0 Naru ReportModel)
# ================================================================
# 默认 float32 (4 bytes/param)。blacklist 用来排除某类参数 (例不算 embedding)。
def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    L.info(f'Number of model parameters: {num_params} (~= {mb:.2f}MB)')
    L.info(model)
    return mb

# ================================================================
# qerror: 标准 q-error 公式 (= L0 ErrorMetric)
# ================================================================
# q-error = max(est, true) / min(est, true), 总是 ≥ 1, 越接近 1 越准。
# 边界情形 (跟 L0 一致): 0/0 = 1 (完美); est=0 而 true>0 → true (= 跟把 0 当 1 比);
#                       est>0 而 true=0 → est (对称处理)。
def qerror(est_card, card):
    if est_card == 0 and card == 0:
        return 1.0
    if est_card == 0:
        return card
    if card == 0:
        return est_card
    if est_card > card:
        return est_card / card
    else:
        return card / est_card

# ================================================================
# rmserror: RMS error in selectivity space
# ================================================================
# 用 selectivity (= cardinality / total_rows) 算 RMS, 让不同大小的表可比。
# RMSE 跟 q-error 是互补指标: q-error 更看 ratio (= 倍数误差), RMSE 看绝对值。
def rmserror(preds, labels, total_rows):
    return np.sqrt(np.mean(np.square(preds/total_rows-labels/total_rows)))

# ================================================================
# evaluate: 算 q-error + 6+ 个分位数 + 可选 RMS
# ================================================================
# 报 7 个指标: max / 99th / 95th / 90th / median / mean / gmean。
# Naru paper 主要关心 99th / median (= 极端 vs 典型 query 表现)。
# 返回 (errors np.array, metrics dict)。
def evaluate(preds, labels, total_rows=-1):
    errors = []
    for i in range(len(preds)):
        errors.append(qerror(float(preds[i]), float(labels[i])))

    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'median': np.median(errors),
        'mean': np.mean(errors),
        'gmean': gmean(errors)
    }

    if total_rows > 0:
        metrics['rms'] = rmserror(preds, labels, total_rows)
    L.info(f"{metrics}")
    return np.array(errors), metrics

# ================================================================
# evaluate_errors: 从已有 errors list 直接算分位数 (跳过 qerror 重算)
# ================================================================
# run_test / report_errors 内部调; 给已经算好 q-error 的场景复用代码。
def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'median': np.median(errors),
        'mean': np.mean(errors),
        'gmean': gmean(errors)
    }
    L.info(f"{metrics}")
    return metrics

# ================================================================
# report_errors: 从 result CSV 文件读 error 列, 算分位数报出来
# ================================================================
# CLI 用 `lecarb report --result-file ...` 触发。
# 不重新跑 estimator, 只读历史 CSV 重报指标。
def report_errors(dataset, result_file):
    df = pd.read_csv(RESULT_ROOT / dataset / result_file)
    evaluate_errors(df['error'])

# ================================================================
# report_dynamic_errors: 模拟 "model 更新延迟" 实验 (= data shift 评估)
# ================================================================
# 场景: 数据更新了, 但 model 重训需要时间 (current_t < max_t)。
# 在 [0, current_t] 时段用旧 model 跑新数据 (= old_new), [current_t, max_t]
# 用更新后 model (= new_new)。两份 result CSV 按时间权重混合后算 q-error。
# = 模拟 "在线 system 在 model 还没更新完成时的真实表现"。
def report_dynamic_errors(dataset, old_new_file, new_new_file, max_t, current_t):
    '''
    max_t: Time limit for update
    current_t: Model's update time.
    old_new_path: Result file of applying stale model on new workload
    new_new_path: Result file of applying updated model on new workload
    '''
    old_new_path = RESULT_ROOT / dataset / old_new_file
    new_new_path = RESULT_ROOT / dataset / new_new_file
    if max_t > current_t:
        try:
            o_n = pd.read_csv(old_new_path)
            n_n = pd.read_csv(new_new_path)
            assert len(o_n) == len(n_n), "In current version, the workload test size should be same."
            o_n_s = o_n.sample(frac = current_t / max_t)
            n_n_s = n_n.sample(frac = 1 - current_t / max_t)
            mixed_df = pd.concat([o_n_s, n_n_s], ignore_index=True, sort=False)
            return evaluate_errors(mixed_df['error'])
        except OSError:
            print('Cannot open file.')
    return -1

# ================================================================
# lazy_derive: data shift 实验下复用旧 result, 按行数比例缩放
# ================================================================
# 场景: 在 version A 上跑过 result, 现在想测 version B (数据多了几行) 的 result。
# 如果 estimator 没重训, 估计值大概按行数比例缩放就行 (= 经验近似, 不是精确)。
# r = test_row / origin_row, 把 prediction 乘 r, 重新算 q-error 写新 CSV。
# 节省一次完整跑 estimator 的时间。
def lazy_derive(origin_result_file, result_file, r, labels):
    L.info("Already have the original result, directly derive the new prediction!")
    df = pd.read_csv(origin_result_file)
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
        for index, row in df.iterrows():
            p = np.round(row['predict'] * r)
            l = labels[index].cardinality
            writer.writerow([int(row['id']), qerror(p, l), p, l, row['dur_ms']])
    L.info("Done infering all predictions from previous result")

# ================================================================
# run_test: 主测试 loop, 替代 L0 [eval_model.py:RunN](../../../AllModels/Naru/eval_model.py)
# ================================================================
# 任何 estimator 跑 workload 测试都通过这个函数, 输入 estimator 实例, 输出 result CSV
# (含每条 query 的 error / predict / label / 耗时)。
#
# 三条路径
# ----------------------------------------------------------------
#   1. 已有 result + overwrite=False  → 直接 exit (避免重复跑)
#   2. data shift (version 跟 estimator.table.version 不同) + lazy=True + 有旧 result
#      → 调 lazy_derive 按行数比例缩放, 不真跑 estimator
#   3. query_async=True (BN/MSCN 慢 estimator 走 Ray 分布式)
#      → estimator.query_async + ray.get 收 stats → 算 q-error
#   4. 默认: 串行 for-loop 跑每条 query
def run_test(dataset: str, version: str, workload: str, estimator: Estimator, overwrite: bool, lazy: bool=True, lw_vec=None, query_async=False) -> None:
    # for inference speed.
    # cuDNN 优化 (跟 L0 eval_model.py 顶部一致): non-deterministic + benchmark mode 推理快。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # uniform thread number
    # 强制 NUM_THREADS 让 benchmark 公平 (所有 estimator 同样 CPU 预算)。
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    L.info(f"Start loading queryset:{workload} and labels for version {version} of dataset {dataset}...")
    # only keep test queries
    queries = load_queryset(dataset, workload)['test']
    labels = load_labels(dataset, version, workload)['test']

    # ========= LW (lightweight) estimator 的特殊 hack =========
    # lw_nn / lw_tree 内部已经把 query 预处理成 (X, gt) tensor, 测试时直接用 X
    # 而不是 raw Query 对象。这里检查 X 长度 + gt 跟 labels 对齐, 然后替换 queries。
    if lw_vec is not None:
        X, gt = lw_vec
        #  assert isinstance(estimator, LWNN) or isinstance(estimator, LWTree), estimator
        assert len(X) == len(queries), len(X)
        assert np.array_equal(np.array([l.cardinality for l in labels]), gt)
        L.info("Hack for LW's method, use processed vector instead of raw query")
        queries = X

    # prepare file path, do not proceed if result already exists
    result_path = RESULT_ROOT / f"{dataset}"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / f"{version}-{workload}-{estimator}.csv"
    if not overwrite and result_file.is_file():
        L.info(f"Already have the result {result_file}, do not run again!")
        exit(0)

    # ========= data shift 检测 =========
    # 如果 test version 跟 estimator 训练用的 version 不同 (= 数据变了, model 没重训):
    #   - 算 row 数比例 r = test_row / train_row, 后面把 prediction 乘 r 缩放
    #   - 如果旧 version 上已有 result + lazy 模式 → 调 lazy_derive 直接复用 + 缩放
    # 默认 lazy=True 是为了加速 data shift 系列实验。
    r = 1.0
    if version != estimator.table.version:
        test_row = load_table(dataset, version).row_num
        r = test_row / estimator.table.row_num
        L.info(f"Testing on a different data version, need to adjust the prediction according to the row number ratio {r} = {test_row} / {estimator.table.row_num}!")

        origin_result_file = RESULT_ROOT / dataset / f"{estimator.table.version}-{workload}-{estimator}.csv"
        if lazy and origin_result_file.is_file():
            return lazy_derive(origin_result_file, result_file, r, labels)

    # ========= Async 路径 (BN / 其它慢 estimator 走这) =========
    # estimator 持有一组 ray workers, 每条 query 异步派发, 全派完后
    # ray.get(...) 阻塞收集所有 worker 的 stats。
    if query_async:
        L.info("Start test estimator asynchronously...")
        for i, query in enumerate(queries):
            estimator.query_async(query, i)

        L.info('Waiting for queries to finish...')
        stats = ray.get([w.get_stats.remote() for w in estimator.workers])

        errors = []
        latencys = []
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
            # 解码 worker_id 与 query_index 的映射:
            # query i 被发给了 worker (i % num_workers), 在该 worker 的 stats 里位置是 (i // num_workers)。
            # 这是 bayesnet.py:Bayes.query_async 的 round-robin 调度的逆映射。
            for i, label in enumerate(labels):
                r = stats[i%estimator.num_workers][i//estimator.num_workers]
                assert i == r.i, r
                error = qerror(r.est_card, label.cardinality)
                errors.append(error)
                latencys.append(r.dur_ms)
                writer.writerow([i, error, r.est_card, label.cardinality, r.dur_ms])

        L.info(f"Test finished, {np.mean(latencys)} ms/query in average")
        evaluate_errors(errors)
        return

    # ========= 默认串行路径 (Naru / Sampling / PG / MHist 走这) =========
    # 串行 for-loop 跑每条 query, 用 r 缩放 (data shift), 算 q-error, 写 CSV。
    # CSV 每行: id / error / predict / label / dur_ms (= 后续分析的基础数据)。
    L.info("Start test estimator on test queries...")
    errors = []
    latencys = []
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
        for i, data in enumerate(zip(queries, labels)):
            query, label = data
            est_card, dur_ms = estimator.query(query)
            est_card = np.round(r * est_card)
            error = qerror(est_card, label.cardinality)
            errors.append(error)
            latencys.append(dur_ms)
            writer.writerow([i, error, est_card, label.cardinality, dur_ms])
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} queries finished")
    L.info(f"Test finished, {np.mean(latencys)} ms/query in average")
    evaluate_errors(errors)


