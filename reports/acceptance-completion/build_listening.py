"""Make a local, blinded listening worksheet without inventing listener scores."""
import html
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path('/Users/vanch/yue2-mlx')
out = ROOT / 'outputs/acceptance-completion/listening'
out.mkdir(parents=True, exist_ok=True)
groups, mapping = [], {}
for name, title, sources in [
    ('english', '英文 · 钢琴流行', [
        ('原版', 'outputs/acceptance-completion/reference-fixed-score-bf16/song'),
        ('MLX BF16', 'outputs/precision-comparison/fixed-score-bf16'),
        ('MLX 8bit', 'outputs/precision-comparison/fixed-score-8bit')]),
    ('chinese', '中文 · City Pop', [
        ('原版', 'outputs/acceptance-completion/reference-full-song-staged/song'),
        ('MLX BF16', 'outputs/precision-comparison/full-song-bf16'),
        ('MLX 8bit', 'outputs/precision-comparison/full-song-8bit')]),
]:
    random.Random(831001 if name == 'english' else 12300).shuffle(sources)
    available = [(i, label, ROOT / path) for i, (label, path) in enumerate(sources) if (ROOT / path / 'audio.flac').exists()]
    cards = []
    mapping[name] = {}
    for i, label, source in available:
        sample = chr(65 + i)
        filename = f'{name}-{sample}.flac'
        if not (out / filename).exists() or (out / filename).stat().st_size != (source / 'audio.flac').stat().st_size:
            shutil.copyfile(source / 'audio.flac', out / filename)
        mapping[name][sample] = {'variant': label, 'source': str(source),
                                'audio_sha256': hashlib.sha256((out / filename).read_bytes()).hexdigest()}
        cards.append(f'<article><h3>样本 {sample}</h3><audio controls preload="metadata" src="{filename}"></audio>'
                     f'<label>整体听感 <select data-key="{name}-{sample}-overall"><option value="">未评分</option>'
                     + ''.join(f'<option>{n}</option>' for n in range(1, 6)) + '</select> / 5</label>'
                     f'<textarea data-key="{name}-{sample}-notes" placeholder="歌词、旋律、人声、衔接；请记录异常的大致时间"></textarea></article>')
    request = json.loads((available[0][2] / 'request.json').read_text())
    groups.append(f'<section><h2>{title}</h2><details><summary>查看目标歌词</summary><pre>{html.escape(request["lyrics"])}</pre></details>'
                  f'<div class="grid">{"".join(cards)}</div></section>')

(out / 'mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YuE2 整曲试听验收</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#172437;background:#f4f6fa;margin:0}main{max-width:1100px;margin:auto;padding:42px 24px}
h1{font-size:32px;margin-bottom:12px}h2{font-size:23px}p{max-width:850px;color:#536176}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
article{background:white;border:1px solid #dce3ee;border-radius:14px;padding:20px}audio{width:100%;margin:8px 0 18px}select{font:inherit;margin:0 8px}textarea{box-sizing:border-box;width:100%;min-height:120px;margin-top:14px;padding:12px;border:1px solid #ccd4e1;border-radius:8px;font:inherit}
section{margin:34px 0}button{background:#215cca;color:white;padding:12px 20px;border:0;border-radius:8px;font:inherit;cursor:pointer}details{margin:10px 0 20px}pre{white-space:pre-wrap;font:inherit}small{color:#536176}
</style><main><h1>整曲试听验收</h1><p>请先完整听完每组样本，再评价整体听感、歌词是否唱对、人声是否自然、音乐结构与衔接。1 分最差，5 分最好；请特别记录杂音、重复、突兀停止及对应时间。</p>
<p>每组使用相同创作请求，独立生成，因此长度和编曲可以不同。样本标签隐藏了版本信息；目前没有预填任何人工评价。评分只保存在当前浏览器。</p>
GROUPS
<button id="export">导出我的评价</button> <small>导出后将评价文件交回，即可记录人工验收结果。</small>
<details><summary>完成评分后查看版本对应</summary><pre id="mapping"></pre></details>
<script>
const key='yue2-acceptance-listening-v1';let values=JSON.parse(localStorage.getItem(key)||'{}');
document.querySelectorAll('[data-key]').forEach(el=>{el.value=values[el.dataset.key]||'';el.addEventListener('input',()=>{values[el.dataset.key]=el.value;localStorage.setItem(key,JSON.stringify(values))})});
document.getElementById('export').onclick=()=>{let blob=new Blob([JSON.stringify({status:'human_review',recorded_at:new Date().toISOString(),samples:MAPPING,ratings:values},null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='listening-review.json';a.click();URL.revokeObjectURL(a.href)};
document.getElementById('mapping').textContent=JSON.stringify(MAPPING,null,2);
</script></main></html>'''
page = page.replace('GROUPS', ''.join(groups)).replace('MAPPING', json.dumps(mapping, ensure_ascii=False).replace('<', '\\u003c'))
(out / 'index.html').write_text(page)
print(out / 'index.html')
