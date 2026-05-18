#!/bin/bash
# ================================================================
# 教学注释 (annotation pass) — dynamic_exp.sh 总览
# ================================================================
# Data shift / dynamic update 实验的主驱动脚本。对每个 dataset × 每种扰动
# 类型 (cor/skew) × 每个 estimator 跑一遍 update_x + test, 把 log 写到
# log/{dataset}/{est}_{up}-exp{i}.out。
#
# 实验设计 (paper "online update" 部分)
# ----------------------------------------------------------------
# 1. 原始数据 + base workload + 训练好的初始 model (= setup, 不在本脚本)
# 2. 数据扰动: append 一批扰动数据 (cor=极强相关 / skew=极偏态), batch_ratio=0.2
# 3. 各 estimator update: 不同 estimator 用不同策略
#    - MSCN/lw-tree/lw-nn   : 真重训 (retrain), 慢但准
#    - Postgres/MySQL       : 重新 ANALYZE (= 重建 histogram), 中等
#    - Naru/DeepDB          : 增量微调 (= update_naru/update_deepdb, load 旧 state 继续训), 快
#    - QuickSel             : 重训 (走 Java 外部工具)
# 4. test 同一个 workload, 比 q-error
# 5. parse_log_exmaple.py 解析各 log 拿 update 耗时
# 6. report_dynamic_errors (utils.py) 用耗时算时间混合 q-error
#
# 两轮实验
# ----------------------------------------------------------------
# 第一轮 (line 7-37): 默认 epoch/参数, 跑全部 estimators
# 第二轮 (line 41-59): 只扫 lw-nn / naru 在不同 epoch 下的 accuracy 曲线
#                     (= 看多训 epoch 能否在 update 上比 retrain 更准)
#
# just: Justfile-based command runner (= make 的现代替代)
# 真实命令在 AreCELearnedYet 根的 Justfile 里, 例 `just dynamic-naru-census13 ...`
# 展开成完整的 `python -m lecarb update-train --estimator naru --params ...`。
# 详见 [Justfile](../Justfile)。
# ================================================================
#################### Dynamic exp
### Do not use 0 as random seed, becasue our Postgres sets random seed as 1/seed
# ↑ 强约束: postgres.py 用 setseed(1/seed), seed=0 会除零, 所以 exp index 从 1 起。
log_path='log'
exp_num=1
# 三重 for: exp number × dataset (4个) × update type (默认只 cor)。
# 每次实验输出 log 到 log/{dataset}/{est}_{up}-exp{i}.out。
for (( i=1; i < 1+$exp_num; ++i ))
do
    for dataset in 'census13' 'forest10' 'power7' 'dmv11'
    do
        for up in 'cor' #'skew' 
        do
            ## MSCN
            just dynamic-mscn-${dataset} ${dataset} 'original' 'base' ${up} '0.2' '10000' "$i" >${log_path}/${dataset}/mscn_${up}-exp${i}.out 2>&1

            ## lw retrain
            just dynamic-lw-tree-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '8000' "$i" >>${log_path}/${dataset}/lwtree_${up}-exp${i}.out 2>&1
            just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" '500' >>${log_path}/${dataset}/lwnn_eq500_${up}-exp${i}.out 2>&1
            just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" '100' >${log_path}/${dataset}/lwnn_eq100_${up}-exp${i}.out 2>&1

            ## Postgres
            just dynamic-postgres-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/postgres_${up}-exp${i}.out 2>&1

            ## MySQL
            just dynamic-mysql-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/mysql_${up}-exp${i}.out 2>&1

            ## Naru
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '1' >>${log_path}/${dataset}/naru_eq1_${up}-exp${i}.out 2>&1
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '7' >${log_path}/${dataset}/naru_eq7_${up}-exp${i}.out 2>&1
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '15' >${log_path}/${dataset}/naru_eq15_${up}-exp${i}.out 2>&1
            ## QuickSel
            just dynamic-quicksel ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/quicksel_${up}-exp${i}.out 2>&1

            ## DeepDB
            just dynamic-deepdb-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/deepdb_${up}-exp${i}.out 2>&1
        done
    done
done


# epoch vs accuracy
# ================================================================
# 第二轮: epoch scaling 实验
# ================================================================
# 只扫 lw-nn / naru, 看不同 epoch 下 update model 的 q-error 怎么变。
# 比较问题: "在 update time budget 内, 多训 epoch 是否比 retrain 更值得?"
# lw-nn 扫 100/200/300/400/500 epoch, naru 扫 1/5/10/15/20 epoch
# (= 都覆盖 1-2 个数量级, 能画出 accuracy-vs-epoch 曲线)。
for (( i=1; i < 1+$exp_num; ++i ))
do
    for dataset in 'census13' 'forest10' 'power7' 'dmv11'
    do
        for up in 'cor' 'skew' 
        do
            ## lwNN
            for ep in '100' '200' '300' '400' '500'
            do
                just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" $ep >${log_path}/${dataset}/lwnn_eq${ep}_${up}-exp${i}.out 2>&1
            done
            ## Naru
            for ep in '1' '5' '10' '15' '20'
            do
                just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" $ep >${log_path}/${dataset}/naru_eq${ep}_${up}-exp${i}.out 2>&1
            done
        done
    done
done


echo `date` "All Finished!"
