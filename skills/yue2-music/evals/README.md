# 验证范围

`test_contract.py`：真实 protocol / CLI 的五模式、profile、覆盖优先级、预算、
ABC 字节保留、拒绝 ACE-Step 参数、dry-run 不写盘、不覆盖已有输出。
运行：项目 Python `-m unittest discover -s skills/yue2-music/evals -p 'test_*.py' -v`。

`trigger_cases.json` + `semantic_config.json` 供 Yao trigger_eval.py 使用。
这是固定词汇/意图规则检查，不是模型路由准确率或盲测；不宣称泛化准确率。
真实音频 smoke 与 same-input step comparison 见 reports/validation.md。
