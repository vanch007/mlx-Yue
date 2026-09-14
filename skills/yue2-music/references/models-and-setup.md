# 本机执行环境与模型

主项目：https://github.com/vanch007/mlx-Yue
模型：https://huggingface.co/vanch007/mlx-Yue2-3B
试听：https://vanch007.github.io/mlx-Yue/

当前 canonical 技能目录位于项目 `skills/yue2-music`。本机项目为
`/Users/vanch/yue2-mlx`；`.codex/skills/yue2-music` 通过符号链接发现同一技能。
移到其他机器时设 `YUE2_PROJECT` 或传 `--project`，不要复制硬编码个人路径到请求。

```bash
cd /Users/vanch/yue2-mlx
PY="$PWD/.runtime/bin/python"
SKILL="$PWD/skills/yue2-music"
"$PY" -m lyra.cli --help
"$PY" -m lyra.cli doctor --model models/converted
```

本机优先 `.runtime/bin/python`（独立 Torch-free）；否则使用项目已安装环境
`.venv/bin/python`。先检查 Python 3.12、MLX/Metal、包 `mlx-yue` 及本地
`models/paths.json` 中的 `model`/`vae`。新安装遵循根 README 的冻结环境配置；
不要仅因找不到可执行文件就运行任意远程安装脚本。

`generate_music.py` 从 `models/paths.json` 读取真实路径，也支持 `--model/--vae`。
AR: `ar-8bit.safetensors` 或 `ar-bf16.safetensors`；NAR/conditioning 保持 BF16；
默认 VAE 是原版 FP32 MLX，48 kHz stereo。`reference` 是精度基线，不保证更好听。
4bit 在协议和转换器中可用，但不是本机完整试听过的默认模型，缺文件必须报错。
仅需要量化准备时调用 `mlx-yue prepare --precision 4bit`，不得静默降低精度。

转谱需要 `ffmpeg`、transcription extras、本地 SheetSage2 和 MERT-v2-FullSong
缓存。默认离线；只有任务确需且授权范围允许时下载官方模型。无需读取 HF token
或浏览器认证。离线缓存缺失要报告具体缺项。

性能：同一 GPU 串行执行；默认 16 GiB sampled whole-process budget、256 VAE
core frames、256 NAR query tile、完整 attention、FP32 accumulation、TF32 关闭。
不可为了速度关闭资源 guard、改局部 attention 或跳过歌词。内存不足先释放其他
GPU 作业、降低 VAE tile，必要时检查 Python API 的 query_chunk_size；这两项均要
实际复测，不能保证任意更小 tile 的速度更好。保留压力量测和失败文件。

生成助手默认要求接电；用户选择电池运行时用 `--allow-battery` 并记录，不能把它
和接电基准混为一组。不要自动修改系统电源模式。无论任务多长，都等待真实结果。

兼容入口 `run_yue2.py` / `mlx-yue music` 保留，但它不接受 generation_config；
8/32 步与所有 sampling 参数使用新的 generate_music.py 或原生 generate CLI。
