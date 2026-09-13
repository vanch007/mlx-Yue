"""Serial installed-wheel validation, full regression and final artifact checks."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/acceptance-completion"
ENV = {**os.environ, "PYTHONPATH": "src:vendor/yue/src", "MLX_ENABLE_TF32": "0"}
rows = []
commands = [
    ("final-build", ["/Users/vanch/.local/bin/uv", "build", "--offline"], ROOT),
    ("final-install", ["/Users/vanch/.local/bin/uv", "pip", "install", "--python",
        "outputs/acceptance-audit-env/bin/python", "--offline", "--no-deps", "--reinstall-package",
        "lyra-yue2", "dist/lyra_yue2-0.1.0-py3-none-any.whl"], ROOT),
    ("installed-wheel-native", [str(ROOT / "outputs/acceptance-audit-env/bin/python"), "-I",
        str(REPORT / "test_installed_pipeline.py")], "/tmp"),
    ("final-tests", [".venv/bin/pytest", "-q"], ROOT),
    ("final-lint", [".venv/bin/ruff", "check", "src", "tests", "tools"], ROOT),
    ("final-artifacts", [".venv/bin/python", str(REPORT / "verify_outputs.py")], ROOT),
    ("final-listening", [".runtime/bin/python", str(REPORT / "build_listening.py")], ROOT),
    ("final-diff-check", ["git", "diff", "--check"], ROOT),
]
for name, command, cwd in commands:
    start = time.time()
    with (REPORT / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    row = {"name": name, "command": command, "status": "pass" if result.returncode == 0 else "fail",
           "exit_code": result.returncode, "seconds": time.time() - start}
    rows.append(row)
    (REPORT / "final-validation-jobs.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(name, row["status"], round(row["seconds"], 2), flush=True)
    if result.returncode and name in {"final-build", "final-install"}:
        raise SystemExit(result.returncode)
raise SystemExit(any(row["exit_code"] for row in rows))
