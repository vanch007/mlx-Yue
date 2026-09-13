"""Build one evidence-based full-song listening page; reject stale/mixed inputs."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

from run_fullsong_benchmark import latest_directory, measurement_for

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/official-fullsong'
OUT = ROOT / 'outputs/official_fullsong'
DOCS = ROOT / 'docs'
LANGUAGES = {'ja': '日语', 'zh': '中文', 'en': '英语', 'ko': '韩语', 'ru': '俄语', 'es': '西班牙语'}
MODES = {'supplied': '官方完整乐谱', 'generated': '相同文本 · 重新作曲',
         'none': '无乐谱直接生成', 'transcribed': '录音转谱 → 生成'}


def esc(value):
    return html.escape(str(value), quote=True)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def asset(source, encode=False):
    """Content-addressed URLs prevent old 32-second clips surviving a rebuild."""
    fingerprint = digest(source)
    destination = DOCS / 'audio/fullsong' / (fingerprint[:20] + '.mp3')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if encode:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                            '-codec:a', 'libmp3lame', '-b:a', '320k', str(destination)], check=True)
        else:
            shutil.copyfile(source, destination)
    return str(destination.relative_to(DOCS))


def status_label(measurement):
    if measurement is None:
        return 'pending · 尚未完成'
    if measurement['status'] == 'failed':
        return 'fail · 运行失败'
    if set(measurement.get('truncated', {})) != {'abc', 'semantic'}:
        return 'missing evidence · 缺少结束状态'
    if any(measurement.get('truncated', {}).values()):
        return 'fail · 触及生成上限'
    if measurement['status'] != 'complete':
        return 'pending · 运行未完成'
    return 'pass · 运行自然结束；音乐质量 pending'


def prepare_case(case):
    cid = case['id']
    destination = latest_directory(cid)
    measurement = measurement_for(cid)
    if measurement:
        expected = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        if measurement['case_sha256'] != expected:
            raise ValueError('Mismatched request: ' + cid)
    row = dict(case, measurement=measurement)
    source = ROOT / case['official_audio_path']
    if digest(source) != case['official_audio_sha256']:
        raise ValueError('Reference audio hash mismatch: ' + cid)
    row['official_player'] = asset(source)
    if case.get('source_audio_path'):
        recording = ROOT / case['source_audio_path']
        if digest(recording) != case['source_audio_sha256']:
            raise ValueError('Transcription source audio hash mismatch: ' + cid)
        row['source_player'] = asset(recording)
        transcription_path = destination / 'transcription/result.json'
        if transcription_path.is_file():
            row['transcription'] = read(transcription_path)
            if row['transcription']['source_audio_sha256'] != case['source_audio_sha256']:
                raise ValueError('Transcription result belongs to another recording')
    if measurement and measurement.get('audio_sha256'):
        if digest(destination / 'song/result.json') != measurement['result_sha256']:
            raise ValueError('Result metadata hash mismatch: ' + cid)
        result = read(destination / 'song/result.json')
        if result['truncated'] != measurement['truncated']:
            raise ValueError('Truncation metadata mismatch: ' + cid)
        audio_path = destination / 'song/audio.flac'
        if digest(audio_path) != measurement['audio_sha256']:
            raise ValueError('Local audio hash mismatch: ' + cid)
        actual = read(destination / 'song/request.json')
        for key, value in case['request'].items():
            if actual[key] != value:
                raise ValueError(f'Request field mismatch: {cid}/{key}')
        row['local_player'] = asset(audio_path, encode=True)
        score_path = destination / 'song/score.abc'
        if score_path.is_file():
            row['actual_score'] = score_path.read_text()
    return row


def card(row):
    cid, req, m = row['id'], row['request'], row['measurement']
    local = '<p class="pending">尚无有效音频，见运行记录。</p>'
    details = ''
    if m and 'local_player' in row:
        peak = m['resources']['sampled_maxima']
        swap = m['resources']['end']['system_swap_out_bytes'] - m['resources']['start']['system_swap_out_bytes']
        local = f'<audio controls preload="none" src="{row["local_player"]}"></audio><p>{m["audio_seconds"]:.2f} 秒 · 48 kHz · MP3 试听副本</p>'
        details = f'''<p>端到端 {m['e2e_seconds']:.2f} 秒 · RTF {m['rtf']:.3f} · 实时倍速 {m['speed_multiplier']:.3f}×<br>
物理 footprint 峰值 {peak['physical_footprint_bytes']/2**30:.2f} GiB · RSS 峰值 {peak['rss_bytes']/2**30:.2f} GiB · MLX 分配峰值 {peak.get('mlx_peak_bytes',0)/2**30:.2f} GiB<br>
全机新增 swap-out {swap/2**20:.1f} MiB · 原版时长比 {m['duration_ratio']:.3f} · 截断：{esc(m['truncated'])}</p>
<details><summary>实测分阶段耗时、资源与配置</summary><pre>{esc(json.dumps(m, ensure_ascii=False, indent=2))}</pre></details>'''
    elif m:
        details = f'<p class="pending">{esc(m.get("error", "运行未完成"))}</p>'
    extra = ''
    actual_score = ''
    source_player = ''
    if row.get('source_player'):
        source_player = f'<div class="audio-panel"><h3>实际转谱输入：官方乐谱录音</h3><audio controls preload="none" src="{row["source_player"]}"></audio><p><a href="{esc(row["source_audio_url"])}">转谱输入来源</a></p></div>'
        if row.get('transcription'):
            source_player += f'<details><summary>完整转谱记录与诊断（不等于乐谱准确率通过）</summary><pre>{esc(json.dumps(row["transcription"], ensure_ascii=False, indent=2))}</pre></details>'
    if row['score_mode'] in ('generated', 'transcribed') and row.get('actual_score'):
        actual_score = f'<h4>本机实际生成 / 转谱的完整乐谱</h4><pre>{esc(row["actual_score"])}</pre>'
    if row['score_mode'] == 'generated':
        extra = '只控制官方原始歌词与风格；模型重新写谱，旋律不承诺相同。Melody 模式相对于带和弦官方谱属于消融测试。'
    elif row['score_mode'] == 'transcribed':
        extra = '输入为官网的原始乐谱录音，完整转谱后重新生成；转谱误差也包含在结果中。不能称作使用完全相同的 ABC。'
    else:
        extra = '逐字使用官方歌词、风格' + ('和完整 ABC 乐谱。Genre Explorer 的官方生成谱在这里作为固定条件重合成；不等同重新自动规划。' if req.get('abc') and row.get('official_mode') == 'planned' else ('和完整 ABC 乐谱。' if req.get('abc') else '；官方和本地均无乐谱直接生成。'))
    return f'''<article class="case-card" id="{cid}" data-language="{row['language']}" data-mode="{row['score_mode']}" data-search="{esc(row['genre']+' '+row['title']+' '+cid)}">
<div class="card-header"><div><span class="badge badge-accent">{LANGUAGES[row['language']]} · {req['cot']} · {MODES[row['score_mode']]}</span><h2>{esc(row['title'])}</h2></div><span class="pending">{status_label(m)}</span></div>
<p class="hint">{esc(extra)} 官方案例 ID：{row['official_id']}</p>
<div class="comparison-grid"><div class="audio-panel"><h3>官方 Demo · {row['official_seconds']:.2f} 秒</h3><audio controls preload="none" src="{row['official_player']}"></audio><p><a href="{esc(row['official_audio_url'])}">官方音频来源</a></p></div>
<div class="audio-panel"><h3>本机 MLX · AR 8bit</h3>{local}</div></div>{source_player}{details}
<details><summary>完整风格、歌词、乐谱与来源</summary><h4>风格</h4><pre>{esc(req['style'])}</pre><h4>歌词</h4><pre>{esc(req['lyrics'])}</pre><h4>输入乐谱</h4><pre>{esc(req.get('abc', '不提供固定乐谱：自动生成 / 转谱 / off，见模式说明'))}</pre>{actual_score}<a href="evidence/{cid}.json">下载该项输入与实测记录 JSON</a></details>
<details><summary>记录人工听评（未预填）</summary><p class="hint">请评价歌词是否唱全、人声与语言、旋律/乐谱遵循、风格、音质及结尾是否完整。分数不会自动判定通过。</p><label for="note-{cid}">试听记录</label><textarea id="note-{cid}" data-note="{cid}" placeholder="填写你听到的具体差异"></textarea></details>
</article>'''


def main():
    manifest = read(REPORT / 'cases.json')
    rows = [prepare_case(case) for case in manifest['cases']]
    (DOCS / 'evidence').mkdir(exist_ok=True)
    for row in rows:
        (DOCS / 'evidence' / (row['id'] + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
    ready = [row for row in rows if row['measurement'] and row['measurement'].get('audio_seconds')]
    natural = [row for row in ready if not any(row['measurement']['truncated'].values()) and row['measurement']['status']=='complete']
    table = []
    for row in rows:
        m = row['measurement']
        if m and m.get('audio_seconds'):
            cells = f"<td>{m['audio_seconds']:.2f}</td><td>{m['e2e_seconds']:.2f}</td><td>{m['rtf']:.3f}</td><td>{m['resources']['sampled_maxima']['physical_footprint_bytes']/2**30:.2f}</td>"
        else:
            cells = '<td colspan="4">pending / fail，见详情</td>'
        table.append(f'<tr><td><a href="#{row["id"]}">{esc(row["id"])}</a></td><td>{row["official_seconds"]:.2f}</td>{cells}<td>{status_label(m)}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>mlx-Yue · 官方完整输入重测与对比试听</title><link rel="stylesheet" href="showcase.css"></head><body><main class="container">
<nav><a href="https://github.com/vanch007/mlx-Yue">GitHub 项目</a><a href="https://huggingface.co/vanch007/mlx-Yue2-3B">Hugging Face 模型</a><a href="https://map-yue2.github.io/">官方 Demo</a><a href="evidence/REPORT.zh-CN.md">对比报告</a><a href="evidence/old-cover-audit.json">旧翻唱错配证据</a></nav>
<h1>mlx-Yue · 完整歌曲对比试听</h1><p class="lead">官方完整文本与展示乐谱，可追溯音频与真实耗时。所有语言、生成模式及翻唱编辑集中在本页。</p>
<div class="notice"><strong>旧报告已撤回作为效果验收依据。</strong>原 11 个样本均被 800-token 上限截断到约 32 秒；旧翻唱卡片错配音频且使用写死的指标。下方为重新生成的独立记录。自然结束、非静音和时长接近均不等于达到官方音乐质量；人工听评与严格数值验收仍未闭环。</div>
<p>Apple M3 Max / 128 GiB · AR 8bit / NAR BF16 / VAE FP32 · 32 步 midpoint · 原生采样默认值（9,000 semantic tokens 上限）· 固定种子 2026，不筛选最佳候选；断电失败保留记录并按原种子重试。</p>
<p class="hint">官方单曲的随机种子、精确采样配置与选优预算未公开；这里可验证公开输入一致性与本机输出，不能证明与官方 checkpoint/随机轨迹逐项相同，也不是论文基准复跑。六种语言为代表性抽样，不代表覆盖全部 70 个流派。</p>
<p class="hint">RTF = 端到端秒数 / 实际音频秒数，越低越快。端到端含加载/权重校验与生成，转谱案例也含完整转谱；文件保存耗时单列。内存分别列 footprint、RSS 与 MLX 计数，不相加；swap 为全机计数。逐项新进程串行运行；未清除系统文件缓存，日常桌面后台仍在运行。</p>
'''
    page += f'<p><strong>{len(natural)} / {len(rows)} 项运行自然结束</strong> · {len(ready)} 项已产出音频 · 音乐质量验收：pending</p>'
    page += '<div class="controls"><label>语言 <select id="language"><option value="">全部语言</option>' + ''.join(f'<option value="{key}">{value}</option>' for key,value in LANGUAGES.items()) + '</select></label><label>输入方式 <select id="mode"><option value="">全部模式</option>' + ''.join(f'<option value="{key}">{value}</option>' for key,value in MODES.items()) + '</select></label><label>搜索 <input id="search" type="search" placeholder="流派、曲名或案例 ID"></label><button id="export">导出听评</button></div>'
    page += '<details><summary>全部实测指标</summary><div class="table-scroll"><table class="summary-table"><thead><tr><th>案例</th><th>官方秒数</th><th>MLX 秒数</th><th>端到端秒数</th><th>RTF</th><th>footprint GiB</th><th>状态</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table></div></details>'
    page += ''.join(card(row) for row in rows)
    page += '''</main><script>
const controls=['language','mode','search'].map(id=>document.getElementById(id));
function filter(){const [lang,mode,search]=controls.map(x=>x.value.toLowerCase());document.querySelectorAll('article').forEach(a=>a.hidden=!!((lang&&a.dataset.language!==lang)||(mode&&a.dataset.mode!==mode)||(search&&!a.dataset.search.toLowerCase().includes(search))));}
controls.forEach(el=>el.addEventListener('input',filter));
document.addEventListener('play',event=>{if(event.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==event.target)a.pause();});},true);
document.querySelectorAll('textarea').forEach(t=>{const key='mlx-yue-fullsong-v2-'+t.dataset.note;try{t.value=localStorage.getItem(key)||'';}catch{}t.addEventListener('input',()=>{try{localStorage.setItem(key,t.value);}catch{}});});
document.getElementById('export').onclick=()=>{const notes=Array.from(document.querySelectorAll('textarea')).map(t=>({id:t.dataset.note,notes:t.value,evidence:'evidence/'+t.dataset.note+'.json'}));const url=URL.createObjectURL(new Blob([JSON.stringify({date:new Date().toISOString(),notes},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='mlx-yue-fullsong-listening.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></body></html>'''
    (DOCS / 'index.html').write_text(page)
    report = ['# 官方完整输入重测', '', '**旧试听报告不能作为官方效果验收证据。** 11 个 800-token 样本全部截断；旧 Cover 卡片使用了不匹配音频及写死指标。原始失败记录保留于 outputs 和 old-benchmark-audit.json。', '',
              '新测试逐字读取官网原始歌词、风格与指定乐谱；完整乐谱 / 自动作曲 / off / 音频转谱链路明确区分。种子固定 2026；只因执行失败恢复，不因听感筛选重抽。断电中止的原始记录和同输入重试均保留；AR 8bit，NAR BF16，VAE FP32，32 步 midpoint，原生 9000 semantic token 上限，允许自然 EOS，不补静音、不拉伸、不人为设定最短整曲时长。', '',
              '官方单曲种子、采样参数和选优预算 unknown；没有相同权重身份与随机轨迹证据，因此此处是公开输入对照，不能声称逐样本严格复现官方输出。六语种为代表性抽样，不代表覆盖 70 流派。', '',
              'RTF = 端到端耗时 / 实际音频时长。端到端包含加载、校验及生成；转谱模式还包含完整音频转谱；保存另列。逐项新进程串行运行；未清除系统文件缓存，日常桌面后台仍在运行。资源在同进程每 0.1 秒采样，峰值 footprint / RSS / MLX 分列，全机 swap 增量不得自动归因于本任务。', '',
              '| 案例 | 官方秒 | MLX秒 | 端到端秒 | RTF | footprint GiB | 运行 |', '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        m=row['measurement']
        if m and m.get('audio_seconds'):
            report.append(f"| {row['id']} | {row['official_seconds']:.2f} | {m['audio_seconds']:.2f} | {m['e2e_seconds']:.2f} | {m['rtf']:.3f} | {m['resources']['sampled_maxima']['physical_footprint_bytes']/2**30:.2f} | {status_label(m)} |")
        else:
            report.append(f"| {row['id']} | {row['official_seconds']:.2f} | — | — | — | — | {status_label(m)} |")
    report += ['', '## 证据与验收边界', '',
               f'- {len(natural)}/{len(rows)} 项自然结束；尚不能以 EOS、时长或非零 RMS 代替歌曲完整度、歌词覆盖或音乐质量。',
               '- 输入与来源：cases.json（完整官方记录、公开 URL、来源与输入 SHA256）；实测：measurements.json；原始请求、配置、乐谱、token、音频和资源采样：outputs/official_fullsong/<id>/。',
               '- Genre Explorer 展示的谱是官方生成结果；本地固定此谱重合成，不等同原来从文本自动规划的流程。同一官方乐谱的六语种与 Cover 案例用于检查谱/歌词遵循。自动 Full/Melody 是相同文本的重新作曲；Melody 对带和弦官方谱属于模式消融，不能要求旋律相同。',
               '- audio_to_cover 使用官方 scoreAudio 完整录音作为转谱源，并非旧录音替代。实际转谱 ABC 可与官方 ABC 比较，不预设转谱准确。',
               '- 试听 MP3 是完整 FLAC 的 320 kbps 副本，无截取、归一化或拉伸；原始 FLAC 留在本机。',
               '- 人工听评 pending。页面可记录歌词完整性、人声语言、旋律、风格、音质及结尾；未填主观分数。严格 AR/NAR 数值超限继续参考 acceptance-completion/REPORT.zh-CN.md，不因重测而豁免。',
               '- 在线统一试听：https://vanch007.github.io/mlx-Yue/。']
    for row in rows:
        if row.get('transcription'):
            tr = row['transcription']
            report += ['', '## 完整转谱链路的内容限制', '',
                       f"转谱处理 {tr['duration_seconds']:.2f} 秒官方乐谱录音，导出 {tr['abc_measures']} 小节、{tr['melody_notes']} 个音符（人声 {tr['vocal_notes']}，器乐 {tr['instrumental_notes']}）。源是乐谱录音，不是真人人声演唱；本测试不代表人声旋律转谱准确率已通过。",
                       f"truncated={tr['truncated']}；warnings={tr['warnings']}。另有 {len(tr.get('diagnostics', []))} 项 diagnostics，不能因 warnings 为空忽略它们：",
                       *['- ' + entry for entry in tr.get('diagnostics', [])]]
    report_text='\n'.join(report)+'\n'
    (REPORT / 'REPORT.zh-CN.md').write_text(report_text)
    (DOCS / 'evidence/REPORT.zh-CN.md').write_text(report_text)
    for name in ['cases.json','measurements.json','old-benchmark-audit.json','old-cover-audit.json','audio-validation.json','validation.json','aggregate.json']:
        if (REPORT/name).exists():
            shutil.copyfile(REPORT/name,DOCS/'evidence'/name)
    print(f'Built {DOCS / "index.html"}: {len(ready)} audio results, {len(natural)} natural ends / {len(rows)} cases')


if __name__ == '__main__':
    main()
