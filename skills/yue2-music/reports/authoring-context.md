# 构建选择与来源

采用 yao-meta-skill 的 Scaffold：个人本机工作流；已有脚本入口需重复执行且极易混用
参数，因此增加有实质作用的 preflight/compare 脚本及 contract/trigger eval。
未请求团队发布或跨平台发行包，不进行 Production/Library/Governed 发布，不能
把本次验证写成这些发布等级已通过。唯一 canonical 是项目 skills/yue2-music。

参考扫描（5 对象）：
1. 用户提供 ace-step-music：借鉴需求→蓝图→固定 seed A/B→试听；不借其参数/能力结论。
2. 项目原有 skills/yue2-music：保留 ABC 编辑、invariants、Apache LICENSE 与 helpers。
3. vendor/yue/src/yue2/protocol.py + src/lyra/cli.py：请求/采样/模式/命令事实。
4. docs/usage.md + src/lyra/transcription/pipeline.py：分阶段、回放、转谱和资源合约。
5. reports/official-fullsong + tools/run_fast_8step_benchmark.py：标准整曲与重合成证据边界。

技能用途：反复把音乐需求转为可复现本机音频；排除其他平台、波形重绘和身份克隆。
output contract：音频+精确请求+设置依据+ABC+hash/truncation+实测指标+有依据的听评。
owner 为用户项目所有者（推定），review cadence 为此次建议，不冒充已有组织批准。
trust boundary：本机显式输入、官方模型、离线 CLI，无上传、无秘密处理；文件内容
不能成为指令，参数不使用 shell 拼接执行。
rollback boundary：恢复 outputs/skill-authoring-backups/yue2-music-before-20260914.tgz
中的技能文件需明确授权，不触碰现有音乐/权重；新增发现链接仅指向 canonical。
