"""Build test configurations for all 5 acceptance generation modes + cover workflow."""
import json
import urllib.request
from pathlib import Path

url = "https://map-yue2.github.io/data/cases.js"
req = urllib.request.urlopen(url)
content = req.read().decode("utf-8")
start = content.find("{")
end = content.rfind("}") + 1
data = json.loads(content[start:end])
cases = data.get("cases", [])
case_dict = {c["id"]: c for c in cases}

modes_eval = [
    {
        "mode_name": "Full + Generated Score (全自动乐谱规划)",
        "mode_key": "full_generated",
        "cot": "full",
        "official_id": "2998556d4b082ca29c856652",
        "title": "Thanksgiving Comedy · 欢快流行叙事",
        "genre": "Comedy / Upbeat Pop",
        "case": case_dict["2998556d4b082ca29c856652"],
        "use_supplied_abc": False
    },
    {
        "mode_name": "Full + Supplied Score (外部指定完整多声部谱)",
        "mode_key": "full_supplied",
        "cot": "full",
        "official_id": "1b91eef30046ca2f0cf5633e",
        "title": "Easy Listening · 治愈系长笛钢琴日文歌",
        "genre": "Easy Listening",
        "case": case_dict["1b91eef30046ca2f0cf5633e"],
        "use_supplied_abc": True
    },
    {
        "mode_name": "Melody + Generated Score (自动主旋律规划)",
        "mode_key": "melody_generated",
        "cot": "melody",
        "official_id": "115abed0e655f5fa9b1409b3",
        "title": "当代都市 R&B · 律动合成器流行",
        "genre": "Modern R&B",
        "case": case_dict["115abed0e655f5fa9b1409b3"],
        "use_supplied_abc": False
    },
    {
        "mode_name": "Melody + Supplied Score (外部指定主旋律谱)",
        "mode_key": "melody_supplied",
        "cot": "melody",
        "official_id": "e3d76e66a92637ebb46e61ed",
        "title": "Jazz-Funk · 律动放克与人声旋律线",
        "genre": "Jazz-Funk",
        "case": case_dict["e3d76e66a92637ebb46e61ed"],
        "use_supplied_abc": True
    },
    {
        "mode_name": "Off / Direct Generation (直接端到端生成 / 无乐谱)",
        "mode_key": "off_direct",
        "cot": "off",
        "official_id": "bdfc142ef136070839953ffa",
        "title": "Enka · 经典复古演歌 (Direct Generation)",
        "genre": "Enka / Retro Pop",
        "case": case_dict["bdfc142ef136070839953ffa"],
        "use_supplied_abc": False
    }
]

out_list = []
out_dir = Path("outputs/acceptance_modes_benchmark/official_audios")
out_dir.mkdir(parents=True, exist_ok=True)

for m in modes_eval:
    c = m["case"]
    audio_url = "https://map-yue2.github.io/" + c.get("audio")
    official_file = out_dir / f"{m['mode_key']}_official.mp3"
    if not official_file.exists():
        print(f"Downloading {m['mode_key']} official audio...")
        try:
            req = urllib.request.Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, official_file.open("wb") as f:
                f.write(resp.read())
            print(f"  ✓ Saved {official_file.name}")
        except Exception as e:
            print(f"  ✗ Error downloading {audio_url}: {e}")
            
    abc_text = c.get("abc") if m["use_supplied_abc"] else None
    out_list.append({
        "mode_key": m["mode_key"],
        "mode_name": m["mode_name"],
        "cot": m["cot"],
        "title": m["title"],
        "genre": m["genre"],
        "official_id": m["official_id"],
        "official_audio": f"official_audios/{m['mode_key']}_official.mp3",
        "official_url": "https://map-yue2.github.io/",
        "style": c.get("tags"),
        "lyrics": c.get("lyrics"),
        "abc": abc_text,
        "seed": 2026
    })

Path("examples/acceptance_modes_cases.json").write_text(json.dumps(out_list, ensure_ascii=False, indent=2) + "\n")
print(f"Successfully generated config for {len(out_list)} acceptance modes!")

