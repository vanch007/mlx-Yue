"""Compatibility entrypoint for the corrected full-song benchmark.

The former 800-token capped report is withdrawn. All results use the unified
page and source-validated runner. Run prepare_fullsong_benchmark.py first.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/run_fullsong_benchmark.py"),
                    "--only", *['ja_score', 'ru_score', 'es_score', 'zh_score', 'ko_score', 'en_score']], cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "tools/build_unified_showcase.py")],
                   cwd=root, check=True)
