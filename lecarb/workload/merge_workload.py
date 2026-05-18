# ================================================================
# 教学注释 (annotation pass) — merge_workload.py 总览
# ================================================================
# 把分批生成的 workload 拼成一个。
# 场景: 大 workload (几万 query) 一次生成 + 算 label 慢且不好并行,
# 实际操作分成 N 份 (workload_0 / workload_1 / ... / workload_{N-1}) 并行跑,
# 然后用这个脚本合并成最终 workload。
#
# 用法: `lecarb merge-workload --params "{'count': 10}"`
# 完成后删临时文件: rm data/{dataset}/workload/{workload}_[0-9]*
# ================================================================
import logging
from .workload import load_queryset, load_labels, dump_queryset, dump_labels

L = logging.getLogger(__name__)

# ================================================================
# merge_workload: 串接 count 份分块 workload 成一个
# ================================================================
# 三个 split (train/valid/test) 分别串接, label 同步。
# 直接 list += list 不去重 (假设各分块 disjoint), 加起来 = 最终 workload。
def merge_workload(dataset: str, version: str, workload: str, count: int=10) -> None:
    queryset = {'train': [], 'valid': [], 'test': []}
    labels = {'train': [], 'valid': [], 'test': []}

    for i in range(count):
        L.info(f"Merge querset {workload}_{i}...")
        qs = load_queryset(dataset, f"{workload}_{i}")
        ls = load_labels(dataset, version, f"{workload}_{i}")
        for k in queryset.keys():
            #  print(f"{k}: {ls[k][0]}")
            queryset[k] += qs[k]
            labels[k] += ls[k]

    for k in queryset.keys():
        L.info(f"Final queryset has {len(queryset[k])} queries with {len(labels[k])} labels")

    L.info("Dump queryset and labels...")
    dump_queryset(dataset, workload, queryset)
    dump_labels(dataset, version, workload, labels)
    L.info(f"Done, run: rm data/{dataset}/workload/{workload}_[0-9]* to remove temporary files")
