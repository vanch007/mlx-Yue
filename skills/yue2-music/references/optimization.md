# 从需求到合适设置

## 用户 brief 与创作蓝图

提取：用途/受众、主题、主语言、1–3 个主要流派、情绪变化、2–5 个核心乐器与
演唱/制作描述、要保留的旋律/和声、歌词、目标时长、速度/质量优先级。
用户给了歌词就保留；允许代写时先写一个易唱 hook，再用主歌铺场景、副歌复现。
缺少源音频/ABC 且要求基于原歌时才阻塞询问。无需逐项询问流派、种子和参数。

`style` 模板：语言 + 主流派 + 演唱音色 + 核心乐器 + 节奏/情绪曲线 + 制作空间。
可在 style 写 `88 BPM, G major, 4/4` 作为软提示；需要谱面控制时使用 ABC 的
Q/K/M 和实际音符。这里没有 ACE-Step 独立的 bpm/keyscale/duration/instrumental
字段、NO 硬排除机制、DCW、thinking、Suno tags 接口。

例：`Mandarin acoustic folk, clear warm female vocal, fingerpicked acoustic guitar,
soft bamboo flute, restrained verse rising into a memorable chorus, intimate dry
recording, slow 76 BPM, sparse acoustic arrangement`。
民族音乐描述具体乐器及演奏法，避免互相冲突的编制词；提示抑制某乐器只属软约束。

歌词使用已有官方示例的 `[Verse]`、`[Chorus]`、`[Bridge]` 等简洁段落标记。
中文每行约 7–12 字、英文约 6–10 音节只是写作起点，按旋律和语言修正；不把产品
参数长串或技术说明唱进歌词。多语言按完整段落安排。翻译歌词按重音、音节和呼吸
改编，不承诺逐音素对齐。Instrumental 可尝试空歌词、明确 instrumental style 和
Ins 谱面，但模型没有保证零人声的硬开关。

时长来自歌词结构、乐谱小节数/速度和模型 EOS。2–4 分钟可设计 verse/chorus
复现及 bridge；短广告先写独立短结构。默认保留 semantic.max_tokens=9000
（约 360 秒的语义帧上限，不是目标时长），ABC.max_tokens=4096。
9000 也不是无限整曲保证：特别长歌词/谱仍可能耗尽预算或上下文。需要更长时先
检查 24576 上下文约束，不能随意放大。要求精确秒数时说明其为编曲目标；必要时
把后期裁剪/淡出另存并标记，不能当作模型自然结束。`--preview-seconds` 仅用于用户
要的诊断截短，绝不能拿这种输出代替完整歌曲验收。

## 配置决策

| 需求 | 起点 | 理由 / 证据边界 |
|---|---|---|
| 普通创作、最终整曲 | standard: AR 8bit, 32 midpoint | 本项目完整歌曲标准路径 |
| 速度优先、快速候选、明确 8 步 | fast: AR 8bit, 8 midpoint | 已有同语义声学重合成对比；整曲墙钟另外量测 |
| 排查量化影响、精度基线 | reference: BF16, 32 midpoint | 与 8bit 对比；不预言听感更好 |
| 需要编曲和可编辑和声 | full | 自动完整谱或 supplied ABC |
| 旋律不变而重编伴奏 | melody + supplied chord-free ABC | 保留符号旋律条件；音频遵循度仍需试听 |
| 完全自由或无需可编辑谱 | off | 直接生成，仍可有人声；不保证更快/更好 |
| 只给录音想翻唱 | SheetSage2 melody-full → inspect → melody | 含器乐主题；melody-vocal 只取主唱需求 |
| 和声也保留 | full 转谱 → full supplied | 检查原转谱和和弦后再生成 |

显式请求优先：例如 JSON 已指定 ode_steps=16，则 fast 不覆写 16，展示实际值。
不会用 profile 暗改 mode、歌词、种子、cfg 或 sampling。默认 seed=831001；
探索变体需代理显式设置不同种子，串行 2–3 条（按用户时间预算）。用户有单曲/耗时
限制时先生成一条。不要承诺自动参数优化能找到全局最佳音乐。

## 调整顺序与停止条件

1. 保留需求和判定标准：风格、歌词可辨、hook、旋律遵循、自然结尾、瑕疵、耗时。
2. 第一次使用项目默认 sampling（见 generation-and-covers），先解决词/谱矛盾。
3. 需要探索且有预算：用 8 步生成不同 seed 的完整候选；试听后选定结构。
4. 改风格/歌词/ABC 必须重新生成 semantic；纯提高声学步数可复用 semantic/noise。
5. 旋律/结构漂移先选 supplied ABC。减少随机性可试 ABC temperature 0.6 对 0.7，
   semantic 0.9 对 1.0；想更多变化可试 1.05。**这些是待任务验证的启发式**。
   出现重复先检查文本，后在固定 seed 下小幅调整 repetition_penalty；不要同时改
   temperature/top_p/top_k/penalty。少量探索 top_k=80 vs 100，不宣称适用于所有风格。
6. cfg 默认 full/melody=1.0、off=1.01。需要试约束可比较 1.0/1.2；CFG 引入额外
   分支会增加成本，也不能锁定音高/人声。不要抄 ACE-Step CFG 值。
7. 在被选候选上用 compare_steps.py 对比 8/32。相同 seed 的跨精度重采样并非
   同一歌曲；严格 NAR 对比必须同 IDs/noise/weights。只复用 latents 的 decode
   无法改变歌曲内容，也不会改变 NAR 步数。
8. 满足用户标准后停止；有限预算内最多两轮针对性改进。没有听评工具时提供可播
   候选让用户选，不虚构听感评分。需要更多迭代必须有具体问题和判据。

交付写“针对本次目标选用的设置”，附实际依据。不要写“参数已达绝对最佳”。
