"""Compatibility entrypoint for the three original cases with complete inputs."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/run_fullsong_benchmark.py"),
                    "--only", "full_generated", "cover_baikal", "cover_birthday"],
                   cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "tools/build_unified_showcase.py")],
                   cwd=root, check=True)
