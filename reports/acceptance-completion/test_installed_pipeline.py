"""Run from /tmp with the freshly installed wheel and no repository import path."""
import importlib.util
import json
from pathlib import Path
import sys

import lyra
from lyra import YuE2Pipeline, GenerationConfig
from lyra.cli import main

ROOT = Path("/Users/vanch/yue2-mlx")
OUT = ROOT / "outputs/acceptance-completion"
assert "site-packages" in lyra.__file__ and "acceptance-audit-env" in lyra.__file__
assert importlib.util.find_spec("torch") is None


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network disabled during installed-wheel acceptance")


sys.addaudithook(deny_network)
paths = json.loads((ROOT / "models/paths.json").read_text())
config = GenerationConfig.from_dict({"ode_steps": 17, "semantic": {"max_tokens": 64, "min_tokens": 0}})
with YuE2Pipeline.from_pretrained(paths["model"], vae=paths["vae"], generation_config=config,
                                 local_files_only=True, require_ac=True) as pipe:
    pipe.save_pretrained(OUT / "saved-pipeline-native")
request = ROOT / "outputs/precision-comparison/fixed-score-bf16/request.json"
assert main(["generate", str(request), "--model", str(OUT / "saved-pipeline-native"),
             "--offline", "--require-ac", "--output", str(OUT / "installed-reload")]) == 0
effective = json.loads((OUT / "installed-reload/config.json").read_text())["generation"]
assert effective == config.to_dict()
assert "torch" not in sys.modules
result = json.loads((OUT / "installed-reload/result.json").read_text())
report = {"status": "pass", "import_path": lyra.__file__, "torch_installed": False,
          "network": "denied connect and DNS", "restored_generation_config": effective,
          "audio_seconds": result["audio_seconds"], "truncated": result["truncated"],
          "scope": "Actual save_pretrained, installed CLI reload and real capped generation; nondefault 17 steps are a configuration test only"}
(ROOT / "reports/acceptance-completion/installed-wheel.json").write_text(json.dumps(report, indent=2) + "\n")
