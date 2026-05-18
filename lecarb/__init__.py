# ================================================================
# 教学注释 (annotation pass) — lecarb 包初始化
# ================================================================
# 整个 lecarb package 共享的 root logger 配置。
# - 所有 module 用 `import logging; L = logging.getLogger(__name__)` 拿 logger
# - 这些子 logger 自动继承根的 DEBUG level 和 StreamHandler (= 输出到 stderr)
# - format: [2026-05-18 12:34:56,789 INFO] lecarb.estimator.naru: 训练完成
# 在不同 module 加 print 已经被弃用 (跟 L0 比), 全用 L.info / L.error / L.debug。
# ================================================================
import logging
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[{asctime} {levelname}] {name}: {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)
