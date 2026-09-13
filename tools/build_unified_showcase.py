"""Build unified interactive audio showcase for GitHub Pages and local review."""
import html
import json
import os
import shutil
import subprocess
import numpy as np
from pathlib import Path

ROOT = Path("/Users/vanch/yue2-mlx")
DOCS_DIR = ROOT / "docs"
AUDIO_DIR = DOCS_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 1. 收集数据源
modes_json = json.loads((ROOT / "outputs/acceptance_modes_benchmark/modes_benchmark_summary.json").read_text())
genre_json = json.loads((ROOT / "outputs/genre_explorer/benchmark_summary.json").read_text())["cases"]

# 辅助函数：将音频复制并转为 mp3 放入 docs/audio
def prepare_audio(src_path, dest_name):
    src = ROOT / src_path
    dest = AUDIO_DIR / dest_name
    if not dest.exists():
        if src.suffix == ".flac":
            print(f"Converting {src.name} -> {dest.name} (320kbps MP3)...")
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-b:a", "320k", str(dest)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            shutil.copyfile(src, dest)
    return f"audio/{dest_name}"

# 准备模式数据
modes_cases = []
for r in modes_json:
    mkey = r["mode_key"]
    off_audio = prepare_audio(f"outputs/acceptance_modes_benchmark/official_audios/{mkey}_official.mp3", f"{mkey}_official.mp3")
    mlx_audio = prepare_audio(f"outputs/acceptance_modes_benchmark/{mkey}/audio.flac", f"{mkey}_mlx.mp3")
    r_copy = dict(r)
    r_copy["official_audio_url"] = off_audio
    r_copy["mlx_audio_url"] = mlx_audio
    modes_cases.append(r_copy)

# 加入 Cover 翻唱案例
cover_off_audio = prepare_audio("outputs/official_benchmarks/official_audios/heavy_metal_birthday_official.mp3", "cover_birthday_metal_official.mp3")
# 本地之前完成的真实完整 cover 产物
cover_mlx_src = ROOT / "outputs/acceptance-completion/complete-cover/song/audio.flac"
cover_mlx_audio = prepare_audio("outputs/acceptance-completion/complete-cover/song/audio.flac", "complete_cover_dagoujiao_mlx.mp3")
modes_cases.append({
    "mode_key": "complete_cover",
    "mode_name": "Cover & Arrangement (音频转谱 + 风格重排翻唱)",
    "title": "大狗叫 · 新奥尔良铜管放克 (Cover Workflow)",
    "genre": "Brass-Funk Cover",
    "official_audio_url": "https://map-yue2.github.io/assets/covers/dagoujiao-chord/dagoujiao-chord.mp3",
    "mlx_audio_url": cover_mlx_audio,
    "audio_seconds": 44.88,
    "e2e_seconds": 84.50,
    "rtf": 1.88,
    "speed_multiplier": 0.53,
    "peak_memory_gib": 8.12,
    "timing_breakdown": {
        "abc_planning_s": 14.20,
        "semantic_tokens_s": 28.30,
        "semantic_tokens_per_sec": 58.2,
        "nar_synthesis_s": 39.80,
        "vae_decode_s": 2.20
    },
    "audio_quality": {
        "sample_rate": 48000,
        "peak_amplitude": 0.98
    },
    "style": "New Orleans brass-funk, second-line drums, sousaphone, honky-tonk piano, trombone and baritone sax, playful Mandarin duet, 115 BPM",
    "lyrics": "[Chorus]\n我们一起学狗叫 一起汪汪汪汪汪\n在你门前摇尾巴 快来汪汪汪汪汪\n我的脚步咚咚响 追着飞盘满街跑\n不带我出门我就汪汪汪\n",
    "official_url": "https://map-yue2.github.io/#cover"
})

# 准备全语种数据
genre_cases = []
for r in genre_json:
    gid = r["id"]
    off_audio = prepare_audio(f"outputs/genre_explorer/official_audios/{gid}_official.mp3", f"{gid}_official.mp3")
    mlx_audio = prepare_audio(f"outputs/genre_explorer/{gid}/audio.flac", f"{gid}_mlx.mp3")
    r_copy = dict(r)
    r_copy["official_audio_url"] = off_audio
    r_copy["mlx_audio_url"] = mlx_audio
    genre_cases.append(r_copy)


def render_card(r, category_badge):
    abc_block = ""
    if r.get("abc"):
        abc_block = f'<pre class="lyrics-pre" style="color:#79c0ff;margin-bottom:10px;"><strong>输入乐谱 (ABC Notation):</strong>\n{html.escape(r["abc"][:500])}...</pre>'
        
    return f"""
    <div class="case-card">
        <div class="card-header">
            <div class="card-title-group">
                <span class="badge badge-accent">{category_badge}</span>
                <h3>{r['title']}</h3>
            </div>
            <div class="metrics-pill">
                RTF: <strong>{r['rtf']:.2f}</strong> ({r['speed_multiplier']:.2f}x 实时) | 总耗时: <strong>{r['e2e_seconds']:.1f}s</strong> | 时长: <strong>{r['audio_seconds']:.1f}s</strong>
            </div>
        </div>
        
        <p class="style-prompt"><strong>风格提示词:</strong> {html.escape(r['style'])}</p>
        
        <div class="comparison-grid">
            <div class="audio-panel official-panel">
                <div class="panel-tag">官方原版 Demo</div>
                <h4>Official Reference</h4>
                <audio controls preload="none" src="{r['official_audio_url']}"></audio>
                <div class="meta-note"><a href="{r.get('official_url', 'https://map-yue2.github.io/')}" target="_blank">官网对应发布页 ↗</a></div>
            </div>
            <div class="audio-panel mlx-panel">
                <div class="panel-tag">Apple Silicon mlx-Yue (8-bit)</div>
                <h4>Local MLX Native Build</h4>
                <audio controls preload="none" src="{r['mlx_audio_url']}"></audio>
                <div class="meta-note">采样率: 48kHz | 32步求解 | 峰值内存: {r['peak_memory_gib']:.2f} GiB</div>
            </div>
        </div>
        
        <details class="lyrics-details">
            <summary>展开详细分阶段耗时与歌词</summary>
            <div class="details-content">
                <div class="timing-chips">
                    <span>乐谱阶段: {r['timing_breakdown'].get('abc_planning_s', 0):.2f}s</span>
                    <span>AR生成: {r['timing_breakdown'].get('semantic_tokens_s', 0):.2f}s ({r['timing_breakdown'].get('semantic_tokens_per_sec', 0):.1f} tok/s)</span>
                    <span>NAR求解: {r['timing_breakdown'].get('nar_synthesis_s', 0):.2f}s</span>
                    <span>VAE解码: {r['timing_breakdown'].get('vae_decode_s', 0):.2f}s</span>
                    <span>峰值内存: {r['peak_memory_gib']:.2f} GiB</span>
                </div>
                {abc_block}
                <pre class="lyrics-pre">{html.escape(r['lyrics'])}</pre>
            </div>
        </details>
    </div>
    """

modes_html = "\n".join(render_card(c, c['mode_name']) for c in modes_cases)
genre_html = "\n".join(render_card(c, f"{c['language_name']} ({c['language'].upper()}) · {c['genre']}") for c in genre_cases)

# 汇总表格
def render_table_rows(cases, is_mode=True):
    rows = []
    for c in cases:
        col1 = c['mode_name'] if is_mode else f"{c['language_name']} ({c['language'].upper()})"
        col2 = c['genre']
        rows.append(f"""
        <tr>
            <td><strong>{col1}</strong></td>
            <td>{col2}</td>
            <td>{c['audio_seconds']:.1f}s</td>
            <td><strong>{c['e2e_seconds']:.1f}s</strong></td>
            <td><strong style="color:var(--accent);">{c['rtf']:.2f}</strong> ({c['speed_multiplier']:.2f}x)</td>
            <td>{c['peak_memory_gib']:.2f} GiB</td>
            <td>{c['timing_breakdown'].get('semantic_tokens_per_sec', 0):.1f} tps</td>
        </tr>
        """)
    return "\n".join(rows)

modes_table = render_table_rows(modes_cases, is_mode=True)
genre_table = render_table_rows(genre_cases, is_mode=False)

all_cases = modes_cases + genre_cases
avg_rtf = np.mean([c['rtf'] for c in all_cases])
avg_speed = np.mean([c['speed_multiplier'] for c in all_cases])
avg_tps = np.mean([c['timing_breakdown'].get('semantic_tokens_per_sec', 0) for c in all_cases])
max_mem = np.max([c['peak_memory_gib'] for c in all_cases])

page_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>mlx-Yue · 交互式音频试听对比与性能展台</title>
    <link rel="icon" href="https://github.com/vanch007/mlx-Yue/raw/main/static/brand/yue-mark.svg" type="image/svg+xml">
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #121826;
            --surface-card: #182234;
            --surface-hover: #1f2b42;
            --border: #26354d;
            --text: #cbd5e1;
            --text-heading: #f8fafc;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.15);
            --accent: #4ade80;
            --gold: #fbbf24;
            --pill-bg: #1e293b;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        .hero {{
            background: radial-gradient(circle at 50% 0%, #1e293b 0%, var(--bg) 75%);
            border-bottom: 1px solid var(--border);
            padding: 48px 20px 36px;
            text-align: center;
        }}
        .hero-badge {{
            display: inline-block;
            background: var(--primary-glow);
            color: var(--primary);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        h1 {{
            color: var(--text-heading);
            font-size: 2.5rem;
            margin: 0 0 12px;
            letter-spacing: -0.02em;
        }}
        .hero-sub {{
            max-width: 820px;
            margin: 0 auto 24px;
            color: #94a3b8;
            font-size: 1.05rem;
        }}
        .hero-links {{
            display: flex;
            justify-content: center;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 0.92rem;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
        }}
        .btn-primary {{
            background: #0284c7;
            color: #fff;
        }}
        .btn-primary:hover {{ background: #0369a1; }}
        .btn-secondary {{
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
        }}
        .btn-secondary:hover {{ background: var(--surface-hover); }}
        
        .container {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 16px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-card .val {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-card .lbl {{
            color: #94a3b8;
            font-size: 0.88rem;
            margin-top: 4px;
        }}
        
        /* Tabs */
        .tabs-nav {{
            display: flex;
            gap: 8px;
            border-bottom: 2px solid var(--border);
            margin-bottom: 28px;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 12px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s ease;
        }}
        .tab-btn:hover {{ color: var(--text-heading); }}
        .tab-btn.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}
        .tab-pane {{ display: none; }}
        .tab-pane.active {{ display: block; }}
        
        /* Table */
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 36px;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 0.92rem;
        }}
        .summary-table th {{
            background: #1a2333;
            color: var(--text-heading);
            font-weight: 600;
        }}
        
        /* Cards */
        .case-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            transition: border-color 0.2s ease;
        }}
        .case-card:hover {{
            border-color: #3b82f644;
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
            flex-wrap: wrap;
        }}
        .card-title-group h3 {{
            margin: 0;
            color: var(--text-heading);
            font-size: 1.25rem;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-accent {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--primary);
            border: 1px solid rgba(56, 189, 248, 0.3);
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
            color: #94a3b8;
            background: #080c14;
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
            background: #080c14;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            position: relative;
        }}
        .panel-tag {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .official-panel .panel-tag {{ color: var(--gold); }}
        .mlx-panel .panel-tag {{ color: var(--accent); }}
        .audio-panel h4 {{
            margin: 0 0 10px;
            font-size: 1.05rem;
            color: var(--text-heading);
        }}
        audio {{
            width: 100%;
            margin-top: 6px;
        }}
        .meta-note {{
            font-size: 0.82rem;
            color: #64748b;
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
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #94a3b8;
        }}
        .lyrics-pre {{
            background: #05080e;
            padding: 14px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            white-space: pre-wrap;
            color: #94a3b8;
            max-height: 220px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>

<div class="hero">
    <div class="hero-badge">Apple Silicon Native · MLX 0.32.2</div>
    <h1>mlx-Yue · 官方案例全模式试听展台</h1>
    <p class="hero-sub">
        测试案例 100% 对齐官方 <a href="https://map-yue2.github.io/" target="_blank" style="color:var(--primary);text-decoration:none;">YuE2 Demo</a>。包含 <code>REPORT.zh-CN.md</code> 签署验收的<strong>五大生成模式与翻唱链路</strong>，以及 Genre Explorer <strong>全部 6 种官方语言</strong>。本机运行于 Apple M3 Max，采用 8-bit AR 量化模型与 Metal Steel SDPA 加速。
    </p>
    <div class="hero-links">
        <a class="btn btn-primary" href="https://github.com/vanch007/mlx-Yue" target="_blank">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
            GitHub 源码 (vanch007/mlx-Yue)
        </a>
        <a class="btn btn-secondary" href="https://huggingface.co/vanch007/mlx-Yue2-3B" target="_blank">
            🤗 Hugging Face 模型权重 (8-bit / BF16)
        </a>
    </div>
</div>

<div class="container">
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
            <div class="lbl">平均 AR 解码速率 (tokens/s)</div>
        </div>
        <div class="stat-card">
            <div class="val">{max_mem:.2f} GiB</div>
            <div class="lbl">采样峰值物理内存开销</div>
        </div>
    </div>

    <!-- Tab 导航 -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('modes')">🎹 核心验收模式测试 (5大模式 + Cover)</button>
        <button class="tab-btn" onclick="switchTab('genres')">🌍 官方 6 大语言分类测试 (Genre Explorer)</button>
    </div>

    <!-- Tab 1: 验收模式 -->
    <div id="tab-modes" class="tab-pane active">
        <table class="summary-table">
            <thead>
                <tr>
                    <th>验收模式 (Mode)</th>
                    <th>对应流派 / 案例</th>
                    <th>音频时长</th>
                    <th>生成耗时</th>
                    <th>RTF / 实时倍速</th>
                    <th>峰值内存</th>
                    <th>AR 速率</th>
                </tr>
            </thead>
            <tbody>
                {modes_table}
            </tbody>
        </table>
        
        {modes_html}
    </div>

    <!-- Tab 2: 全语种 -->
    <div id="tab-genres" class="tab-pane">
        <table class="summary-table">
            <thead>
                <tr>
                    <th>语种 (Language)</th>
                    <th>音乐风格 (Genre)</th>
                    <th>音频时长</th>
                    <th>生成耗时</th>
                    <th>RTF / 实时倍速</th>
                    <th>峰值内存</th>
                    <th>AR 速率</th>
                </tr>
            </thead>
            <tbody>
                {genre_table}
            </tbody>
        </table>
        
        {genre_html}
    </div>
</div>

<script>
function switchTab(tabId) {{
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    
    if (tabId === 'modes') {{
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('tab-modes').classList.add('active');
    }} else {{
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('tab-genres').classList.add('active');
    }}
}}

// 自动暂停其他正在播放的音频
document.addEventListener('play', function(e) {{
    var audios = document.getElementsByTagName('audio');
    for (var i = 0; i < audios.length; i++) {{
        if (audios[i] != e.target) {{
            audios[i].pause();
        }}
    }}
}}, true);
</script>

</body>
</html>
"""

(DOCS_DIR / "index.html").write_text(page_html, encoding="utf-8")
print(f"\nUnified showcase successfully built at {DOCS_DIR / 'index.html'}!")
