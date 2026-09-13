"""Render measured precision comparisons and listening files as a local report."""
import argparse
import html
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models/converted"))
    parser.add_argument("--crop-check", type=Path, help="Optional independent ASR retry on labelled song excerpts")
    args = parser.parse_args()
    rows = json.loads(args.comparison.read_text())
    controlled = json.loads(args.controlled.read_text())
    quality = json.loads((args.comparison.parent / "lyric-quality.json").read_text())
    quality = {(r["case"], r["precision"]): r for r in quality}
    crop_check = json.loads(args.crop_check.read_text()) if args.crop_check else []
    if crop_check:
        (args.comparison.parent / "asr-crop-check.json").write_text(json.dumps(crop_check, indent=2, ensure_ascii=False) + "\n")
    metrics = {}
    for precision in ("bf16", "8bit"):
        data = [r for r in controlled if r["precision"] == precision]
        assert len(data) >= 3 and len({r["audio_seconds"] for r in data}) == 1
        metrics[precision] = {
            "repetitions": len(data), "audio_seconds": data[0]["audio_seconds"],
            "generation_seconds_median": statistics.median(r["timing"]["e2e_seconds"] for r in data),
            "wall_seconds_median": statistics.median(r["wall_seconds"] for r in data),
            "ar_tps_median": statistics.median(r["timing"]["semantic"]["output_tps"] for r in data),
            "peak_footprint_gib_max": max(r["peak_footprint_gib"] for r in data),
            "ar_file_bytes": (args.models / f"ar-{precision}.safetensors").stat().st_size,
        }
    summary = {"hardware": "Apple M3 Max / 128 GiB / macOS 27.0", "controlled": metrics,
               "quality_scope": "Two requests, one seed each; ASR proxy plus waveform checks; human listening pending",
               "speedup_ar": metrics["8bit"]["ar_tps_median"] / metrics["bf16"]["ar_tps_median"],
               "speedup_generation": metrics["bf16"]["generation_seconds_median"] / metrics["8bit"]["generation_seconds_median"]}
    (args.comparison.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# BF16 / 8bit 本机对照", "", "Apple M3 Max · 128 GiB · macOS 27.0", "",
             "仅 AR Linear 权重量化为 8bit；Embedding、声学 conditioning 与 NAR 保持 BF16，VAE 保持 FP32。", "",
             "## 同长度测速", "", "相同乐谱、seed、采样参数；强制生成 400 个 token（约 16 秒），每种精度运行三次并交替顺序。此组有意触发截断，仅用于测速。每次使用独立进程，模型文件已缓存。", "",
             "| 指标 | BF16 | 8bit |", "|---|---:|---:|"]
    for label, key, suffix in (("AR 文件大小", "ar_file_bytes", " GB"),
                               ("AR 速度中位数", "ar_tps_median", " token/s"),
                               ("生成耗时中位数", "generation_seconds_median", " s"),
                               ("含校验、加载、保存的总耗时中位数", "wall_seconds_median", " s"),
                               ("最大采样进程内存", "peak_footprint_gib_max", " GiB")):
        values = [metrics[p][key] / (1e9 if key == "ar_file_bytes" else 1) for p in ("bf16", "8bit")]
        lines.append(f"| {label} | {values[0]:.2f}{suffix} | {values[1]:.2f}{suffix} |")
    lines += ["", "## 自然结束与歌词识别", "", "此组不强制相同长度。量化改变采样轨迹，歌曲内容和时长可能不同，因此不能直接用总耗时计算提速。识别器未接收目标歌词；错误率反映生成与 ASR 两者的误差，不是官方 PER 或音乐质量分。", "",
              "| 请求 | 精度 | 音频时长 | 生成耗时 | 歌词识别错误率 | 截断 |", "|---|---|---:|---:|---:|---|"]
    cards = []
    for row in rows:
        q = quality[row["case"], row["precision"]]
        # An empty recognizer response is missing evidence, not proof that the
        # generated song contains no lyrics. Preserve the original raw capture.
        quality_label = f'{100*q["asr_error_rate"]:.1f}%' if q["transcript"].strip() else "pending（识别器返回空结果）"
        lines.append(f'| {row["case"]} | {row["precision"]} | {row["audio_seconds"]:.2f} s | {row["timing"]["e2e_seconds"]:.2f} s | {quality_label} | {any(row["truncated"].values())} |')
        directory = Path(row["directory"])
        lines += []
        cards.append(f'<article><h2>{html.escape(row["case"])} · {row["precision"]}</h2>'
                     f'<p>{row["audio_seconds"]:.1f} 秒音频 · 生成 {row["timing"]["e2e_seconds"]:.1f} 秒 · 歌词识别错误率 {quality_label}</p>'
                     f'<audio controls preload="metadata" src="{html.escape(directory.name)}/audio.flac"></audio>'
                     f'<details><summary>识别歌词</summary><p>{html.escape(q["transcript"])}</p></details></article>')
    lines += ["", "## 判读边界", "", "- 两个自然结束请求、各一个 seed，不能代表所有曲风和语言。", "- 音频为原始输出，未做响度归一化；音量更大不代表音质更好。", "- 自动检查包括有限采样值、双声道、采样率、低电平片段和接近满刻度比例。", "- 人声自然度、音色、混音、段落连贯性与音乐偏好：人工试听 pending。", "", "原始依据：comparison.json、lyric-quality.json、summary.json，以及各歌曲 result.json 和资源采样日志。", ""]
    if crop_check:
        lines += ["## 8bit 整曲分段识别复核", "", "整段识别返回空结果后，对同一 8bit 音频的三个 30 秒片段分别识别。以下片段均识别到歌词，因此不能将原来的空识别结果判为无歌词或空音频。片段不构成整曲错误率评估。", ""]
        lines += [f'- {r["start_seconds"]}–{r["start_seconds"] + r["duration_seconds"]} 秒：{r["text"]}' for r in crop_check]
    report = "\n".join(lines)
    (args.comparison.parent / "REPORT.zh-CN.md").write_text(report)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YuE2 BF16 / 8bit 对照试听</title><style>body{font:16px/1.65 system-ui;background:#f5f3ee;color:#172329;max-width:1100px;margin:36px auto;padding:0 24px}h1{font-size:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}article{background:white;padding:24px;border-radius:12px}h2{font-size:20px}audio{width:100%}pre{white-space:pre-wrap;font:14px/1.7 system-ui}p{color:#4b5960}summary{cursor:pointer}</style><h1>YuE2 · BF16 / 8bit 对照试听</h1><p>原始音频 · 相同请求与 seed · 原生 MLX · 48 kHz 双声道</p><main>' + ''.join(cards) + '</main><details><summary>查看完整测速与判读边界</summary><pre>' + html.escape(report) + '</pre></details></html>'
    (args.comparison.parent / "compare.zh.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
