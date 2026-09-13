"""Publish evidence status without interpreting missing results as passes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/acceptance-completion"
OUT = ROOT / "outputs/acceptance-completion"


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {"status": "missing evidence"}


numerical = []
for name, actual in [
    ("ar-short", "ar-short-mlx"), ("ar-negative", "ar-negative-mlx"),
    ("ar-song-negative-bounded", "ar-song-negative-bounded-mlx"), ("nar", "nar-mlx"),
    ("vae-default-512-fp64", "vae-default-512-fp64-budget-mlx"),
    ("vae-legacy-512-fp64", "vae-legacy-512-fp64-budget-mlx"),
    ("ar-long-layerwise", "ar-long-layerwise-mlx"),
    ("ar-valid-prefix-layerwise", "ar-valid-prefix-layerwise-mlx"),
]:
    fidelity, anchor = read(OUT / actual / "fidelity.json"), read(REPORT / (name + "-anchor.json"))
    numerical.append({"name": name, "reference_status": fidelity.get("status"),
                      "anchor_status": anchor.get("status"),
                      "reference_failures": fidelity.get("failures"),
                      "anchor_failures": anchor.get("failures"),
                      "reference_report": str(OUT / actual / "fidelity.json"),
                      "anchor_report": str(REPORT / (name + "-anchor.json"))})

tests = (REPORT / "final-tests.log").read_text().strip().splitlines()
summary = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "overall_status": "request_changes",
    "production_defects_fixed": ["transcription cancellation and resource propagation",
        "saved CLI generation config lost through None", "reserved and case-insensitive batch IDs",
        "Torch dependency during native VAE checkpoint export"],
    "final_validation_jobs": read(REPORT / "final-validation-jobs.json"),
    "pytest_summary": tests[-1],
    "installed_wheel": read(REPORT / "installed-wheel.json"),
    "output_checks": read(REPORT / "output-checks.json"),
    "sustained_execution": read(REPORT / "sustain-summary.json"),
    "numerical": numerical,
    "long_reference_attempts": [read(REPORT / name) for name in
        ("numerical-jobs.json", "numerical-retry-jobs.json", "final-numerical-jobs.json", "bounded-reference-jobs.json")],
    "bounded_full_prefix_ar": read(REPORT / "layerwise-cpu-anchor-jobs.json"),
    "complete_greedy_transcription": read(REPORT / "transcription-generated-comparison.json"),
    "attention_diagnostic": read(REPORT / "attention-promotion-diagnostic.json"),
    "original_complete_songs": {
        "english": read(OUT / "reference-fixed-score-bf16/reference.json"),
        "chinese": read(OUT / "reference-full-song-staged/reference.json"),
        "chinese_source_stage_lineage": "Known pinned command and token hashes, but no contemporaneous source-stage weights receipt; not strict lineage approval"},
    "human_listening": {"status": "pending", "ratings_received": False,
        "page": str(OUT / "listening/index.html"), "samples": read(OUT / "listening/mapping.json")},
    "open_acceptance_conditions": ["strict negative-branch AR/NAR numerical failures",
        "real-model transcription floating-point tolerance calibration", "human musical-quality review",
        "unchanged 180–240 second corpus duration gate"],
    "scope": "Local M3 Max only; no universal quality equivalence, hardware maximum, or overall speedup claim",
}
(REPORT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
paths = set(subprocess.check_output(["git", "ls-files", "src", "tools", "tests", "README.md", "docs",
    "validation/README.md", ".ai_project.md", ".ai_memory.md"], cwd=ROOT, text=True).splitlines())
for folder in ("src", "tools", "tests"):
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*.py"))
manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(paths)
            if (ROOT / name).is_file()}
(REPORT / "source-manifest.json").write_text(json.dumps({"base_head": subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "sha256": manifest}, indent=2) + "\n")
print(summary["overall_status"], summary["pytest_summary"], summary["installed_wheel"].get("status"))
