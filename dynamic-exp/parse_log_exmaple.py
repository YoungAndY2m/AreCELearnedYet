# ================================================================
# 教学注释 (annotation pass) — parse_log_exmaple.py 总览
# ================================================================
# (注: 文件名拼错了 "exmaple" → "example", 保持原拼写不动)
# 日志解析工具, 给 dynamic-exp/dynamic_exp.sh 跑出的实验日志算耗时。
# 4 个解析器, 各自匹配一种 log 行模式拿时间戳:
#   - get_gen_query_time   : workload 生成 + label 生成各花多久
#   - get_lw_nn_training_time: LW-NN estimator 训练时长 (分钟)
#   - get_postgres_time    : PG 建统计的耗时
#   - get_mysql_time       : MySQL 建直方图的耗时
#
# 实现: 用 `parse` 库 (= 字符串模板反向匹配, 比 regex 易读)。
# `parse("[{time} INFO] lecarb...: msg with {n:d}", line)` 提取 time + n。
#
# 用途 (dynamic_exp.sh 调)
# ----------------------------------------------------------------
# data shift 实验: 数据更新一次, 跑各 estimator 的 update + test, 比较哪种
# update 策略最快 + 最准。这个脚本把各种 log 里的时间戳解析出来, 喂给
# report_dynamic_errors (utils.py) 算时间混合 q-error。
# ================================================================
import sys
import os
from parse import *
from datetime import datetime as dt

# path hack
# 让脚本能 import 上一级目录 (= AreCELearnedYet 根) 的 lecarb 包。
sys.path.append(os.getcwd())
sys.path.append('..')

# 跟 lecarb logger format 对齐 (= __init__.py 设的)
TIME_FMT = '%Y-%m-%d %H:%M:%S,%f'

# ================================================================
# get_gen_query_time: 找 "生成 train workload + label" 这两步各花多久
# ================================================================
# 解析两对 (start, end) 时间戳:
#   train query: "Start generate workload with N queries for train..." 起,
#                "Start generate workload with M queries for valid..." 止
#                (= 用下一阶段开始时间当上一阶段结束时间)
#   train label: "Updating ground truth labels..." 起, "Dump labels to disk..." 止
# 都用 t[0]-t[1] 算 elapsed seconds, 取第一个匹配 (后续会被忽略)。
# 用于 dynamic-exp 评估 "新数据来了后, 生成新 query+label 要多久" — 这部分
# 时间会算进 model update 总时间, 影响 report_dynamic_errors 的混合权重。
def get_gen_query_time(logfile, training_size):
    '''return time_gen_train_query, time_gen_train_label'''
    t_tr_query = [[],[]]
    t_tr_label = [[],[]]
    # time_update_model = [[],[]]
    with open(logfile, 'r') as log_f:
        lines = log_f.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            s_tr_query=parse("[{time} INFO] lecarb.workload.gen_workload: Start generate workload with {train_num:d} queries for train...", line)
            e_tr_query=parse("[{time} INFO] lecarb.workload.gen_workload: Start generate workload with {test_num:d} queries for valid...", line)
            if s_tr_query and s_tr_query['train_num'] == training_size:
                t_tr_query[0].append(dt.strptime(s_tr_query['time'], TIME_FMT))
            if e_tr_query:
                t_tr_query[1].append(dt.strptime(e_tr_query['time'], TIME_FMT))
            
            # parse time for training label update
            s_tr_label=parse("[{time} INFO] lecarb.workload.gen_label: Updating ground truth labels for the workload, with sample size {}...", line)
            e_tr_label=parse("[{time} INFO] lecarb.workload.gen_label: Dump labels to disk...", line)
            if s_tr_label:
                t_tr_label[0].append(dt.strptime(s_tr_label['time'], TIME_FMT))
            if e_tr_label:
                t_tr_label[1].append(dt.strptime(e_tr_label['time'], TIME_FMT))
        # print(t_tr_query, t_tr_label)

        time_gen_tr_query = 0
        time_gen_tr_label = 0
        if len(t_tr_query[0]) >= 1:
            time_gen_tr_query = (t_tr_query[1][0] - t_tr_query[0][0]).total_seconds()
        if len(t_tr_label[0]) >= 1:
            time_gen_tr_label = (t_tr_label[1][0] - t_tr_label[0][0]).total_seconds()
    return time_gen_tr_query, time_gen_tr_label

# ================================================================
# get_lw_nn_training_time: 找 LW-NN 训练耗时 (分钟)
# ================================================================
# 匹配 "Training finished! Time spent since start: X.XX mins" 这行,
# 拿到的 train_time 已经是分钟, 直接返回。
def get_lw_nn_training_time(logfile):
    with open(logfile, 'r') as log_f:
            lines = log_f.readlines()
            for line in lines:
                line = line.strip()
                update_time=parse("[{} INFO] lecarb.estimator.lw.lw_nn: Training finished! Time spent since start: {train_time:f} mins", line)
                if update_time:
                    return update_time['train_time'] 
    return 0

# ================================================================
# get_postgres_time: 找 PG 建直方图统计的耗时
# ================================================================
# 匹配 postgres.py 里的 "construct statistics finished, using X.XX minutes" log。
# 给 data shift 实验测 "PG 重新 ANALYZE 一次需要多久"。
def get_postgres_time(logfile):
    with open(logfile, 'r') as logf:
        lines = logf.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            update_time=parse("[{} INFO] lecarb.estimator.postgres: construct statistics finished, using {update_time:f} minutes, All statistics consumes {} MBs", line)
            if update_time:
                return update_time['update_time']
    return 0
        
# ================================================================
# get_mysql_time: 找 MySQL 建直方图的耗时
# ================================================================
# 跟 get_postgres_time 同思路, log 行格式略不同 (没 "MBs" 后缀)。
def get_mysql_time(logfile):
    with open(logfile, 'r') as logf:
        lines = logf.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            update_time=parse("[{} INFO] lecarb.estimator.mysql: construct statistics finished, using {update_time:f} minutes", line)
            if update_time:
                return update_time['update_time']
    return 0
