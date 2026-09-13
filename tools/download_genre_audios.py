"""Download official audio clips for the 6 Genre Explorer cases."""
import json
from pathlib import Path
import urllib.request

cases = json.loads(Path("examples/genre_explorer_cases.json").read_text())
out_dir = Path("outputs/genre_explorer/official_audios")
out_dir.mkdir(parents=True, exist_ok=True)

for c in cases:
    cid = c["id"]
    target = out_dir / f"{cid}_official.mp3"
    if not target.exists() or target.stat().st_size == 0:
        print(f"Downloading {c['audio_url']} -> {target.name}...")
        try:
            req = urllib.request.Request(c["audio_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, target.open("wb") as f:
                f.write(resp.read())
            print(f"  ✓ {target.name} ({target.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print(f"  Already exists: {target.name} ({target.stat().st_size / 1024 / 1024:.2f} MB)")

