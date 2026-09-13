"""Benchmark official Genre Explorer cases covering all 6 languages."""
import html
import json
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from lyra import YuE2Pipeline


def monitor_memory(stop_event, peak_container):
    process = psutil.Process(os.getpid())
    peak = 0
    while not stop_event.is_set():
        try:
            mem = process.memory_info().rss
            if mem > peak:
                peak = mem
        except Exception:
            pass
        time.sleep(0.05)
    peak_container[0] = peak


def build_html_report(results, out_dir):
    cards = []
    table_rows = []
    
    for r in results:
        # Table row
        table_rows.append(f"""
        <tr>
            <td><strong>{r['language_name']}</strong> ({r['language'].upper()})</td>
            <td><span class="genre-badge">{r['genre']}</span></td>
            <td>{r['audio_seconds']:.1f}s</td>
            <td><strong>{r['e2e_seconds']:.1f}s</strong></td>
            <td><strong class="rtf-highlight">{r['rtf']:.2f}</strong> ({r['speed_multiplier']:.2f}x 实时)</td>
            <td>{r['peak_memory_gib']:.2f} GiB</td>
            <td>{r['timing_breakdown']['semantic_tokens_per_sec']:.1f} tps</td>
        </tr>
        """)
        
        # Player card
        cards.append(f"""
        <div class="case-card">
            <div class="card-header">
                <div class="card-title-group">
                    <span class="lang-tag">{r['language_name']}</span>
                    <h3>{r['genre']} · {r['title']}</h3>
                </div>
                <div class="metrics-pill">
                    RTF: <strong>{r['rtf']:.2f}</strong> | 耗时: <strong>{r['e2e_seconds']:.1f}s</strong> | 音频: <strong>{r['audio_seconds']:.1f}s</strong>
                </div>
            </div>
            
            <p class="style-prompt"><strong>风格提示词:</strong> {html.escape(r['style'])}</p>
            
            <div class="comparison-grid">
                <div class="audio-panel official-panel">
                    <h4>官方 Demo 原版音频</h4>
                    <audio controls preload="metadata" src="{r['official_audio']}"></audio>
                    <div class="meta-note"><a href="{r['official_url']}" target="_blank">查看官方发布页面 ↗</a></div>
                </div>
                <div class="audio-panel mlx-panel">
                    <h4>本机 mlx-Yue (8-bit) 生成音频</h4>
                    <audio controls preload="metadata" src="{r['mlx_audio']}"></audio>
                    <div class="meta-note">采样率: 48kHz | 峰值电平: {r['audio_quality']['peak_amplitude']:.2f} | 32步求解</div>
                </div>
            </div>
            
            <details class="lyrics-details">
                <summary>查看歌词与详细耗时构成</summary>
                <div class="details-content">
                    <div class="timing-chips">
                        <span>ABC乐谱规划: {r['timing_breakdown']['abc_planning_s']:.2f}s</span>
                        <span>AR语义生成: {r['timing_breakdown']['semantic_tokens_s']:.2f}s ({r['timing_breakdown']['semantic_tokens_per_sec']:.1f} tok/s)</span>
                        <span>NAR声学求解: {r['timing_breakdown']['nar_synthesis_s']:.2f}s</span>
                        <span>VAE音频解码: {r['timing_breakdown']['vae_decode_s']:.2f}s</span>
                        <span>进程峰值内存: {r['peak_memory_gib']:.2f} GiB</span>
                    </div>
                    <pre class="lyrics-pre">{html.escape(r['lyrics'])}</pre>
                </div>
            </details>
        </div>
        """)
        
    avg_rtf = np.mean([r['rtf'] for r in results])
    avg_speed = np.mean([r['speed_multiplier'] for r in results])
    avg_tps = np.mean([r['timing_breakdown']['semantic_tokens_per_sec'] for r in results])
    max_mem = np.max([r['peak_memory_gib'] for r in results])
    total_audio = np.sum([r['audio_seconds'] for r in results])
    pass # total_time([r['e2e_seconds'] for r in results])

    html_content = f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>mlx-Yue · 官方案例全语种实测与试听报告</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #1c2128;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --primary: #58a6ff;
            --accent: #2ea043;
            --warning: #d29922;
            --pill-bg: #21262d;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 32px 16px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1180px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--text-heading);
            font-size: 2.2rem;
            margin-bottom: 8px;
        }}
        .lead {{
            color: #8b949e;
            font-size: 1.05rem;
            margin-bottom: 28px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-card .val {{
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-card .lbl {{
            color: #8b949e;
            font-size: 0.88rem;
            margin-top: 4px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 40px;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 0.95rem;
        }}
        .summary-table th {{
            background: #21262d;
            color: var(--text-heading);
            font-weight: 600;
        }}
        .genre-badge {{
            background: #1f6feb22;
            color: #58a6ff;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid #1f6feb44;
        }}
        .rtf-highlight {{
            color: var(--accent);
        }}
        .case-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .card-title-group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .lang-tag {{
            background: #238636;
            color: #fff;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .card-title-group h3 {{
            margin: 0;
            color: var(--text-heading);
            font-size: 1.25rem;
        }}
        .metrics-pill {{
            background: var(--pill-bg);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.88rem;
        }}
        .style-prompt {{
            font-size: 0.92rem;
            color: #8b949e;
            background: #0d1117;
            padding: 10px 14px;
            border-radius: 8px;
            border-left: 3px solid var(--primary);
            margin: 12px 0 20px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 16px;
        }}
        .audio-panel {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }}
        .audio-panel h4 {{
            margin: 0 0 10px;
            font-size: 0.98rem;
        }}
        .official-panel h4 {{ color: #e3b341; }}
        .mlx-panel h4 {{ color: var(--accent); }}
        audio {{
            width: 100%;
            margin-top: 6px;
        }}
        .meta-note {{
            font-size: 0.82rem;
            color: #8b949e;
            margin-top: 8px;
        }}
        .meta-note a {{
            color: var(--primary);
            text-decoration: none;
        }}
        .lyrics-details {{
            margin-top: 14px;
        }}
        .lyrics-details summary {{
            cursor: pointer;
            color: var(--primary);
            font-size: 0.9rem;
        }}
        .details-content {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed var(--border);
        }}
        .timing-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .timing-chips span {{
            background: var(--pill-bg);
            border: 1px solid var(--border);
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #8b949e;
        }}
        .lyrics-pre {{
            background: #090d13;
            padding: 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            white-space: pre-wrap;
            color: #adbac7;
            max-height: 220px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>mlx-Yue · 官方案例全语种实测与试听报告</h1>
    <p class="lead">测试源自官方 <a href="https://map-yue2.github.io/" target="_blank" style="color:var(--primary);">YuE2 Demo (Genre Explorer)</a>，完整覆盖官方全部 6 种支持语言（中文、英文、日文、韩文、俄文、西班牙文）及代表性音乐风格。本机运行于 Apple M3 Max，采用 8-bit 量化自回归模型与原生 MLX 运行时。</p>
    
    <div class="stat-grid">
        <div class="stat-card">
            <div class="val">{avg_rtf:.2f}</div>
            <div class="lbl">平均 RTF (生成耗时 / 音频时长)</div>
        </div>
        <div class="stat-card">
            <div class="val">{avg_speed:.2f}x</div>
            <div class="lbl">平均实时生成倍速</div>
        </div>
        <div class="stat-card">
            <div class="val">{avg_tps:.1f}</div>
            <div class="lbl">AR 解码速度 (tokens/s)</div>
        </div>
        <div class="stat-card">
            <div class="val">{max_mem:.2f} GiB</div>
            <div class="lbl">采样峰值物理内存</div>
        </div>
        <div class="stat-card">
            <div class="val">{total_audio:.1f}s</div>
            <div class="lbl">累计生成音频总时长</div>
        </div>
    </div>

    <table class="summary-table">
        <thead>
            <tr>
                <th>语言</th>
                <th>流派 (Genre)</th>
                <th>音频时长</th>
                <th>生成耗时</th>
                <th>RTF / 速度</th>
                <th>峰值内存</th>
                <th>AR 速度</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <h2 style="color:var(--text-heading); margin-bottom: 20px;">🎧 详细试听对比与段落分析</h2>
    {''.join(cards)}

    <footer style="text-align: center; color: #8b949e; margin-top: 50px; font-size: 0.88rem;">
        mlx-Yue · Native Apple Silicon MLX Port · <a href="https://github.com/vanch007/mlx-Yue" target="_blank" style="color:var(--primary);">GitHub vanch007/mlx-Yue</a>
    </footer>
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_content, encoding="utf-8")
    print(f"\nHTML report written to {out_dir / 'index.html'}")


def main():
    paths = json.loads(Path("models/paths.json").read_text())
    cases_meta = json.loads(Path("examples/genre_explorer_cases.json").read_text())
    
    out_dir = Path("outputs/genre_explorer")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Initializing mlx-Yue Pipeline (8-bit Quantized AR + BF16 NAR + FP32 VAE) ===")
    init_start = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        paths["model"],
        vae=paths["vae"],
        precision="8bit",
        local_files_only=True,
        require_ac=True
    )
    init_seconds = time.perf_counter() - init_start
    print(f"Pipeline initialized in {init_seconds:.2f}s\n")
    
    results = []
    
    for case in cases_meta:
        cid = case["id"]
        title = case["title"]
        lang = case["language"]
        lang_name = case["language_name"]
        genre = case["genre"]
        case_out = out_dir / cid
        case_out.mkdir(parents=True, exist_ok=True)
        
        print("========================================================")
        print(f"[{lang.upper()}] {genre} - {title}")
        print(f"Prompt: {case['style'][:100]}...")
        print("========================================================")
        
        stop_event = threading.Event()
        peak_mem = [0]
        monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event, peak_mem))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        start_time = time.perf_counter()
        
        # 传入 kwargs 给 pipe.__call__
        song_res = pipe(
            style=case["style"],
            lyrics=case["lyrics"],
            seed=case["seed"],
            cot=case["cot"],
            semantic_sampling={
                "max_tokens": 800,
                "min_tokens": 400,
                "top_k": 80
            }
        )
        elapsed_total = time.perf_counter() - start_time
        
        stop_event.set()
        monitor_thread.join()
        
        song_res.save_artifacts(case_out)
        
        flac_path = case_out / "audio.flac"
        with sf.SoundFile(flac_path) as audio:
            audio_seconds = audio.frames / audio.samplerate
            samples = audio.read(dtype="float32")
            rms = float(np.sqrt(np.mean(samples ** 2)))
            peak_amp = float(np.max(np.abs(samples)))
        
        rtf = elapsed_total / audio_seconds if audio_seconds > 0 else 0
        speed_factor = audio_seconds / elapsed_total if elapsed_total > 0 else 0
        peak_gib = peak_mem[0] / (1024 ** 3)
        
        timing = song_res.timing
        semantic_tps = timing.get("semantic", {}).get("output_tps", 0)
        semantic_seconds = timing.get("semantic", {}).get("seconds", 0)
        abc_seconds = timing.get("abc", {}).get("seconds", 0)
        nar_seconds = timing.get("nar_seconds", 0)
        vae_seconds = timing.get("vae_seconds", 0)
        
        row = {
            "id": cid,
            "title": title,
            "language": lang,
            "language_name": lang_name,
            "genre": genre,
            "official_audio": f"official_audios/{cid}_official.mp3",
            "mlx_audio": f"{cid}/audio.flac",
            "audio_seconds": round(audio_seconds, 2),
            "e2e_seconds": round(elapsed_total, 2),
            "rtf": round(rtf, 3),
            "speed_multiplier": round(speed_factor, 2),
            "peak_memory_gib": round(peak_gib, 2),
            "timing_breakdown": {
                "abc_planning_s": round(abc_seconds, 2),
                "semantic_tokens_s": round(semantic_seconds, 2),
                "semantic_tokens_per_sec": round(semantic_tps, 1),
                "nar_synthesis_s": round(nar_seconds, 2),
                "vae_decode_s": round(vae_seconds, 2)
            },
            "audio_quality": {
                "sample_rate": 48000,
                "rms": round(rms, 4),
                "peak_amplitude": round(peak_amp, 4)
            },
            "style": case["style"],
            "lyrics": case["lyrics"],
            "official_url": "https://map-yue2.github.io/#special-prompt-control"
        }
        results.append(row)
        
        print(f"-> Generated {audio_seconds:.1f}s audio in {elapsed_total:.1f}s")
        print(f"-> RTF: {rtf:.3f} ({speed_factor:.2f}x real-time)")
        print(f"-> Peak Memory: {peak_gib:.2f} GiB")
        print(f"-> AR Speed: {semantic_tps:.1f} tok/s | NAR: {nar_seconds:.1f}s | VAE: {vae_seconds:.1f}s\n")
        
    summary = {
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mac_version": platform.mac_ver()[0],
            "total_ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 1)
        },
        "model_config": {
            "ar_precision": "8bit",
            "nar_precision": "bf16",
            "vae_precision": "fp32",
            "solver_steps": 32
        },
        "cases": results
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    build_html_report(results, out_dir)


if __name__ == "__main__":
    main()

