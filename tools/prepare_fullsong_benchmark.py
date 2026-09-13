"""Freeze complete official demo inputs; never invent lyrics or shorten scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://map-yue2.github.io/"
REPORT = ROOT / "reports/official-fullsong"
OUT = ROOT / "outputs/official_fullsong"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    import soundfile as sf

    REPORT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # A checkout already contains the selected source records. Rehydrate their
    # exact audio assets instead of silently fetching a new version of the site.
    frozen = REPORT / 'cases.json'
    if frozen.exists():
        for case in json.loads(frozen.read_text())['cases']:
            for prefix in ('official_audio', 'source_audio'):
                if prefix + '_path' not in case:
                    continue
                url = case[prefix + '_url']
                if not url.startswith(BASE + 'assets/'):
                    raise ValueError('Expected an official demo asset URL')
                target = ROOT / case[prefix + '_path']
                if not target.resolve().is_relative_to(OUT.resolve()):
                    raise ValueError('Asset path must stay inside the benchmark output')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    payload = urllib.request.urlopen(url, timeout=90).read()
                    if sha(payload) != case[prefix + '_sha256']:
                        raise ValueError('Official asset changed; keep existing benchmark frozen')
                    target.write_bytes(payload)
                if sha(target.read_bytes()) != case[prefix + '_sha256']:
                    raise ValueError('Cached official audio identity mismatch')
            print(case['id'], round(case['official_seconds'], 2), 'seconds (frozen)', flush=True)
        return
    snapshot = OUT / "official-cases.js"
    if not snapshot.exists():
        snapshot.write_bytes(urllib.request.urlopen(BASE + "data/cases.js", timeout=60).read())
    raw = snapshot.read_bytes()
    text = raw.decode()
    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    sources = {c["id"]: c for c in data["cases"] + data["covers"]}
    selection = [
        ("ja_score", "1b91eef30046ca2f0cf5633e", "full", "supplied", "ja"),
        ("ru_score", "43fc3aa38700252b8d63982b", "full", "supplied", "ru"),
        ("es_score", "0792ba06ebe2fd7cdd6c3e7b", "full", "supplied", "es"),
        ("zh_score", "115abed0e655f5fa9b1409b3", "full", "supplied", "zh"),
        ("ko_score", "df64fef4d0ba5a18670f6c19", "full", "supplied", "ko"),
        ("en_score", "ad4ec0a4f236082a2933b48d", "full", "supplied", "en"),
        ("full_generated", "2998556d4b082ca29c856652", "full", "generated", "en"),
        ("melody_generated", "115abed0e655f5fa9b1409b3", "melody", "generated", "zh"),
        ("melody_supplied", "jingle-bells-minor-no-chord", "melody", "supplied", "en"),
        ("off_direct", "bdfc142ef136070839953ffa", "off", "none", "ja"),
        ("cover_dagoujiao", "dagoujiao-chord", "full", "supplied", "zh"),
        ("cover_baikal", "baikal-lake-jazz", "full", "supplied", "zh"),
        ("cover_birthday", "happy-birthday-heavy-metal", "full", "supplied", "en"),
        ("audio_to_cover", "1b91eef30046ca2f0cf5633e", "full", "transcribed", "ja"),
        ("jazz_funk_score", "e3d76e66a92637ebb46e61ed", "full", "supplied", "ja"),
    ]
    cases = []
    for cid, source_id, cot, score, language in selection:
        source = sources[source_id]
        if score == "supplied" and not source.get("abc"):
            raise ValueError(f"Missing official score: {source_id}")
        if score == "none" and source.get("mode") != "direct":
            raise ValueError("Direct comparison requires an official direct-generation case")
        request = dict(id=cid, style=source["tags"], lyrics=source["lyrics"], cot=cot, seed=2026)
        if score == "supplied":
            request["abc"] = source["abc"]
        audio = OUT / "reference" / (source_id + ".mp3")
        audio.parent.mkdir(exist_ok=True)
        if not audio.exists():
            audio.write_bytes(urllib.request.urlopen(BASE + source["audio"], timeout=90).read())
        info = sf.info(audio)
        case = dict(id=cid, official_id=source_id, title=source["title"], genre=source["genre"],
                    language=language, score_mode=score, request=request,
                    official_mode=source.get("mode", "cover/edit"), source=source,
                    official_audio_url=BASE + source["audio"],
                    official_audio_path=str(audio.relative_to(ROOT)),
                    official_audio_sha256=sha(audio.read_bytes()), official_seconds=info.duration,
                    input_hashes={key: sha(value.encode()) for key, value in request.items()
                                  if key in ("style", "lyrics", "abc")})
        if score == "transcribed":
            recording = OUT / "reference" / (source_id + "-score.mp3")
            if not recording.exists():
                recording.write_bytes(urllib.request.urlopen(BASE + source["scoreAudio"], timeout=90).read())
            case.update(source_audio_url=BASE + source["scoreAudio"],
                        source_audio_path=str(recording.relative_to(ROOT)),
                        source_audio_sha256=sha(recording.read_bytes()))
        cases.append(case)
        print(cid, round(info.duration, 2), 'seconds', flush=True)
    manifest = dict(source_url=BASE + "data/cases.js", source_sha256=sha(raw),
                    official_seed="unknown", official_sampling="unknown",
                    official_candidate_selection="unknown for individual demo recordings",
                    selection_policy="Fixed seed 2026; execution failures may be retried with identical input and seed, preserving failures; no best-of-N selection",
                    cases=cases)
    destination = REPORT / "cases.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if destination.exists() and destination.read_text() != serialized:
        previous = json.loads(destination.read_text())
        current = {case['id']: case for case in cases}
        if (any(current.get(case['id']) != case for case in previous['cases']) or
                {k: v for k, v in previous.items() if k != 'cases'} !=
                {k: v for k, v in manifest.items() if k != 'cases'}):
            raise FileExistsError("Use a new report directory for changed source inputs")
    destination.write_text(serialized)


if __name__ == "__main__":
    main()
