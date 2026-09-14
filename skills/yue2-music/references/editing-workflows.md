# 保留约束的编辑流程

先明确用户允许改变的范围：旋律音高、节奏、和声、速度、歌词、人声、编制、段落。
复用已有明确要求；只在冲突时问核心问题。默认自己完成编辑和复核，生成永远串行。

1. 冻结原音频、request、ABC、配置及来源。给新尝试独立路径。
2. 用 abc inspect 看小节、声部、key/meter/tempo。读 abc-editing.md 的原生方言限制。
3. 和声改编改 quoted chord symbols；保持音符/节奏及其他指定 invariant。melody 模式
   不接受和弦符号；如果需要控制新和声用 full supplied。风格不决定精确和弦。
4. 改速度使用 Q；改变调性时需要同时正确移调音符，不能只改 K 导致 unintended 音高。
   改节拍和段落时检查各声部小节时值、rests、section 与歌词顺序。
5. 用 `mlx-yue abc compare before.abc after.abc --voices Vocal` 核对需要固定的声部。
   意图允许 tempo 改变时才加 --allow-tempo-change。预期的其他差异记录到 edit manifest；
   检查不通过不能改名为通过，修复或解释该变化在允许范围内。
6. 新歌词先配音节/重音/换气，alignment sidecar 是创作计划，不是模型硬约束。
7. 将新 score 通过 abc_path 连接新请求并重新生成。换词不支持 waveform repaint；
   decode 旧 latents 只换解码结果，不能让旧歌唱新词。
8. 比较完整基线及编辑点前后过渡。记录可辨歌词、旋律遵循、编曲变化及不满意之处。

保留 edit-brief.md、before/after ABC、invariant 报告、请求和声音。
需要局部音频重绘、锁定伴奏、换声音身份、逐采样替换时说明此项目无原生能力；
不要声称 YuE2 可以硬保留人声/伴奏或直接移植 ACE-Step cover/repaint。
