# M3 Max 原生 MLX 提速验证

本机：Apple M3 Max，128 GiB，macOS 27.0，MLX 0.32.2。

## 同输入整曲对照

复用同一首 169.959 秒歌曲的 4,249 个语义 token 和完整噪声；32 个 midpoint 步骤、256 query 分块、完整注意力、16 GiB 进程预算均保持一致。每条路径单独进程运行一次，期间不运行其他本项目 GPU 任务。

| 指标 | 原路径 | 优化后 |
|---|---:|---:|
| 声学合成（含条件预填充） | 363.09 秒 | 186.94 秒 |
| 采样进程峰值内存 | 9.02 GiB | 8.80 GiB |
| 新增 swap-out | 0 | 0 |
| 相对原歌曲的声学结果最大差值 | 0 | 0 |

声学阶段实测加速 **1.94 倍**，耗时减少 **48.5%**。条件缓存 SHA256 完全一致，完整最终 latent 数组逐元素一致。此结论针对本样本，不代表任意时长、曲风或芯片的固定收益。

## 改动

移除旧款 Apple GPU 上不必要的 FP32 注意力输入转换。MLX 的 Steel 内核仍使用 FP32 累积，保留 BF16 权重/状态和 FP32 VAE。M5 NAX、未知 GPU 架构或未核对的 MLX 版本继续使用原来的 FP32 转换路径。没有减少 solver 步数或注意力范围。

缓存大小也做过探索，但只有顺序测试，因此没有将这些数值认定为稳定收益，也没有更改默认缓存大小。

## 重现

```sh
.runtime/bin/python tools/benchmark_nar.py outputs/precision-comparison/full-song-bf16 --attention baseline --output outputs/my-baseline
.runtime/bin/python tools/benchmark_nar.py outputs/precision-comparison/full-song-bf16 --attention auto --output outputs/my-optimized
```

输出目录必须为新目录。工具验证源歌曲身份和文件哈希，保存计时、缓存哈希、latent 差值和资源记录。

机器可读证据：[performance-summary.json](performance-summary.json)。8bit 对照：[本地测速与试听报告](../outputs/precision-comparison/REPORT.zh-CN.md)。源码与功能范围：[IMPLEMENTATION.md](../docs/IMPLEMENTATION.md)。

## 短片端到端复核

另以 400 token 的 16 秒样本，每种精度重复三次。原路径 BF16/8bit 中位数为 19.08/17.37 秒；优化后为 21.26/19.80 秒。两轮测试时间不同，本机同时运行其他桌面应用，因此目前不能据此分离短序列内核效率和系统负载的影响，也不能宣称短片提速。所有六个样本的 token、噪声、latent 与解码后的音频逐元素一致，证据见 `optimized-output-equivalence.json`。

随后使用同一进程、同一 400 帧状态、每轮预热并交替原路径/优化路径复核，单次声学计算中位数为 0.2003 → 0.1781 秒（1.12 倍），输出完全相同。这支持内核本身的收益，但仍不替代短片端到端速度结论。原始数据：`outputs/nar-short-alternating/profile.json`，摘要：`short-performance-summary.json`。

## 最终音频验证

优化后的完整 latent 经过原生 FP32 VAE 再次解码、保存为 48 kHz 双声道 PCM24 FLAC，与原歌曲解码后的 PCM 逐样本完全相同，最大差值为 0。新文件：`outputs/nar-optimized-16/audio.flac`；证据：`optimized-full-audio.json`。
