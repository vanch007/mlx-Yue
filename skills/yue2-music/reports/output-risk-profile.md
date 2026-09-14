# 输出风险
- 假完整：9000 默认预算，检查 ABC/semantic truncation，preview 明确诊断范围。
- 假最优：模型听感不可由 preset 保证，有限预算固定种子、单变量、试听选择。
- 假速度：CLI 实际墙钟与复用 AR 的 NAR+VAE 时间分开。
- 假兼容：协议/CLI 参数测试，ACE-Step 输入拒绝，源文件优先于旧 README。
- 破坏既有结果：新目录、哈希验证、编辑 ABC 另存、失败保留。
