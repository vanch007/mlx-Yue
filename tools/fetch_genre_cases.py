"""Fetch the 6 representative language cases from Genre Explorer."""
import json
from pathlib import Path
import urllib.request

url = "https://map-yue2.github.io/data/cases.js"
req = urllib.request.urlopen(url)
content = req.read().decode("utf-8")
start = content.find("{")
end = content.rfind("}") + 1
data = json.loads(content[start:end])
cases = data.get("cases", [])

selected_ids = {
    "zh": "115abed0e655f5fa9b1409b3",  # R&B / Synth-Pop
    "en": "ad4ec0a4f236082a2933b48d",  # 50s Rock & Roll
    "ja": "1b91eef30046ca2f0cf5633e",  # Easy Listening
    "ko": "df64fef4d0ba5a18670f6c19",  # Emo Hip-Hop
    "ru": "43fc3aa38700252b8d63982b",  # Folktronica
    "es": "0792ba06ebe2fd7cdd6c3e7b"   # Flamenco
}

out_cases = []
for c in cases:
    cid = c.get("id")
    for lang, target_id in selected_ids.items():
        if cid == target_id:
            slug = c.get("genre", "music").lower().replace(" ", "_").replace("-", "_").replace("&", "and")
            out_cases.append({
                "id": f"{lang}_{slug}",
                "case_id": cid,
                "language": lang,
                "language_name": c.get("languageLabel", lang),
                "genre": c.get("genre"),
                "title": c.get("title"),
                "style": c.get("tags"),
                "lyrics": c.get("lyrics"),
                "audio_url": "https://map-yue2.github.io/" + c.get("audio"),
                "seed": 1000 + len(out_cases) * 111,
                "cot": "melody"
            })

out_path = Path("examples/genre_explorer_cases.json")
out_path.write_text(json.dumps(out_cases, ensure_ascii=False, indent=2) + "\n")
print(f"Successfully saved {len(out_cases)} cases to {out_path}:")
for c in out_cases:
    print(f"  - [{c['language'].upper()}] {c['genre']}: {c['title']} (id: {c['id']})")

