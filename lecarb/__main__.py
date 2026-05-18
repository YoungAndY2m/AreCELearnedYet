"""Le Carb - LEarned CARdinality estimator Benchmark

Usage:
  lecarb workload gen [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [--params <params>] [--no-label] [-o <old_version>] [-r <ratio>]
  lecarb workload label [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload update-label [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [--sample-ratio <sample_size>]
  lecarb workload merge [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload dump [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload quicksel [-d <dataset>] [-v <version>] [-w <workload>] [--params <params>] [--overwrite]
  lecarb dataset table [-d <dataset>] [-v <version>] [--overwrite]
  lecarb dataset gen [-s <seed>] [-d <dataset>] [-v <version>] [--params <params>] [--overwrite]
  lecarb dataset update [-s <seed>] [-d <dataset>] [-v <version>] [--params <params>] [--overwrite]
  lecarb dataset dump [-s <seed>] [-d <dataset>] [-v <version>]
  lecarb train [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--sizelimit <sizelimit>]
  lecarb test [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--overwrite]
  lecarb report [-d <dataset>] [--params <params>]
  lecarb report-dynamic [-d <dataset>] [--params <params>]
  lecarb update-train [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--overwrite]

Options:
  -s, --seed <seed>                Random seed.
  -d, --dataset <dataset>          The input dataset [default: census13].
  -v, --dataset-version <version>  Dataset version [default: original].
  -w, --workload <workload>        Name of the workload [default: base].
  -e, --estimator <estimator>      Name of the estimator [default: naru].
  --params <params>                Parameters that are needed.
  --sizelimit <sizelimit>          Size budget of method, percentage to data size [default: 0.015].
  --no-label                       Do not generate ground truth label when generate workload.
  --overwrite                      Overwrite the result.
  -o, --old-version <old_version>  When data updates, query should focus more on the new data. The <old version> is what QueryGenerator refers to.
  -r, --win-ratio <ratio>          QueryGen only touch last <win_ratio> * size_of(<old version>).
  --sample-ratio <sample-ratio>    Update query set with sample of dataset
  -h, --help                       Show this screen.
"""
# ================================================================
# 教学注释 (annotation pass) — __main__.py 总览
# ================================================================
# lecarb 的 CLI 入口。`python -m lecarb ...` 或 `lecarb ...` 都跑这。
# 用 docopt: 顶部 docstring 既是文档又是 grammar (= argparse 的替代方案,
# 不用手写 add_argument, 维护成本低)。
#
# CLI 顶层 verb (= 第一个位置参数, 见 docstring Usage 段)
# ----------------------------------------------------------------
#   workload {gen|label|update-label|merge|dump|quicksel}
#       生成 / 标注 / 合并 workload, 输出 quicksel 格式
#   dataset  {table|gen|update|dump}
#       建 / 生成 / 扰动 / dump dataset
#   train  -e <estimator>     训 estimator (Naru/MSCN/DeepDB/LW/CoLSE)
#   test   -e <estimator>     测 estimator (= 上述 + Sampling/PG/MySQL/MHist/BN/KDE)
#   report                    从 result CSV 报 q-error 分位数
#   report-dynamic            data shift 时间混合 q-error
#   update-train              增量微调 (= 用 update_naru / update_deepdb)
#
# 通用 flag
# ----------------------------------------------------------------
#   -s seed / -d dataset / -v version / -w workload / -e estimator
#   --params: 字符串 dict (literal_eval 解析, 例 "{'epochs': 100}")
#   --sizelimit: model size / data size 上限比例, 0 关闭
#   --overwrite: 已有 result 也覆盖
#
# Module 加载副作用警告
# ----------------------------------------------------------------
# 顶部所有 import 在 lecarb 启动时一次性 import (= 慢, 因为要加载 torch /
# pomegranate / mysql.connector / ray 等重依赖)。即使你只想跑 sample
# 也要等所有 import 完。这是 lecarb 工程上的小痛点。
# ================================================================
from ast import literal_eval
from time import time

from docopt import docopt

from .workload.gen_workload import generate_workload
from .workload.gen_label import generate_labels, update_labels
from .workload.merge_workload import merge_workload
from .workload.dump_quicksel import dump_quicksel_query_files, generate_quicksel_permanent_assertions
from .dataset.dataset import load_table, dump_table_to_num
from .dataset.gen_dataset import generate_dataset
from .dataset.manipulate_dataset import gen_appended_dataset
from .estimator.sample import test_sample
from .estimator.postgres import test_postgres
from .estimator.mysql import test_mysql
from .estimator.mhist import test_mhist
from .estimator.bayesnet import test_bayesnet
from .estimator.feedback_kde import test_kde
from .estimator.utils import report_errors, report_dynamic_errors
from .estimator.naru.naru import train_naru, test_naru, update_naru
from .estimator.mscn.mscn import train_mscn, test_mscn
from .estimator.lw.lw_nn import train_lw_nn, test_lw_nn
from .estimator.lw.lw_tree import train_lw_tree, test_lw_tree
from .estimator.deepdb.deepdb import train_deepdb, test_deepdb, update_deepdb
from .estimator.colse import train_colse, test_colse
from .workload.workload import dump_sqls

if __name__ == "__main__":
    # docopt 把 docstring 解析成 args dict, key 是带 -- 的 flag 名或位置 verb。
    args = docopt(__doc__, version="Le Carb 0.1")

    # seed: 没指定就用当前时间戳 (= 不可复现, 但实验脚本通常显式传)。
    seed = args["--seed"]
    if seed is None:
        seed = int(time())
    else:
        seed = int(seed)

    # ========= verb dispatch =========
    # 每个 verb 一个 if-block, 解析对应参数后调子模块。`exit(0)` 防止 fall-through。
    if args["workload"]:
        if args["gen"]:
            generate_workload(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                name=args["--workload"],
                no_label = args["--no-label"],
                old_version=args["--old-version"],
                win_ratio=args["--win-ratio"],
                params = literal_eval(args["--params"])
            )
        elif args["label"]:
            generate_labels(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"]
            )
        elif args["update-label"]:
            update_labels(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"],
                sampling_ratio=literal_eval(args["--sample-ratio"])
            )
        elif args["merge"]:
            merge_workload(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"]
            )
        elif args["quicksel"]:
            dump_quicksel_query_files(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"],
                overwrite=args["--overwrite"]
            )
            generate_quicksel_permanent_assertions(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["dump"]:
            dump_sqls(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"])
        else:
            raise NotImplementedError
        exit(0)

    if args["dataset"]:
        if args["table"]:
            load_table(args["--dataset"], args["--dataset-version"], overwrite=args["--overwrite"])
        elif args["gen"]:
            generate_dataset(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["update"]:
            gen_appended_dataset(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["dump"]:
            dump_table_to_num(args["--dataset"], args["--dataset-version"])
        else:
            raise NotImplementedError
        exit(0)

    # ========= train: 训练 NN-based estimator =========
    # 6 个可训练 estimator (naru / mscn / deepdb / lw_nn / lw_tree / colse)。
    # Sampling / PG / MHist 不需要训练, 只在 test 路径出现。
    if args["train"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        sizelimit = float(args["--sizelimit"])

        if args["--estimator"] == "naru":
            train_naru(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "mscn":
            train_mscn(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "deepdb":
            train_deepdb(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_nn":
            train_lw_nn(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_tree":
            train_lw_tree(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "colse":
            train_colse(seed, dataset, version, workload, params, sizelimit)
        else:
            raise NotImplementedError
        exit(0)

    # ========= test: 全部 12 个 estimator 都支持 =========
    # 包含传统 baseline (sample/postgres/mysql/mhist/bayesnet/kde) +
    # NN-based (naru/mscn/deepdb/lw_nn/lw_tree/colse)。
    # 注意 seed 参数: sample/postgres/mysql/mhist/bayesnet/kde/naru 接收 seed,
    # 但 mscn/deepdb/lw_*/colse 不传 (因为它们的随机性已经 fix 在 checkpoint 里)。
    if args["test"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        overwrite = args["--overwrite"]

        if args["--estimator"] == "sample":
            test_sample(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "postgres":
            test_postgres(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mysql":
            test_mysql(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mhist":
            test_mhist(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "bayesnet":
            test_bayesnet(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "kde":
            test_kde(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "naru":
            test_naru(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mscn":
            test_mscn(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "deepdb":
            test_deepdb(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "lw_nn":
            test_lw_nn(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "lw_tree":
            test_lw_tree(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "colse":
            test_colse(dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)

    if args["report"]:
        dataset = args["--dataset"]
        params = literal_eval(args["--params"])
        report_errors(dataset, params['file'])
        exit(0)
    
    if args["report-dynamic"]:
        dataset = args["--dataset"]
        params = literal_eval(args["--params"])
        report_dynamic_errors(dataset, params['old_new_file'], params['new_new_file'], params['T'], params['update_time'])
        exit(0)

    # ========= update-train: 增量微调 (= 不重新训, 从旧 checkpoint 继续) =========
    # 只有 Naru / DeepDB 实现了 update 入口 (= 它们的 estimator file 里有
    # update_xxx() 函数加载旧 state_dict + optimizer_state 接着训)。
    # MSCN / LW / CoLSE 都没实装 — 实测 update naru / deepdb 已经够 paper 用了。
    if args["update-train"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        overwrite = args["--overwrite"]

        if args["--estimator"] == "naru":
            update_naru(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "deepdb":
            update_deepdb(seed, dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)
