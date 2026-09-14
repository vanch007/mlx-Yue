"""No-GPU contract checks against the actual local YuE2 protocol and CLI."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from generate_music import configure, native_modules  # noqa: E402

native_modules(ROOT)
from lyra.cli import parser  # noqa: E402
from yue2.protocol import GenerationConfig  # noqa: E402

ABC = '''X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
V: Vocal
"C"C8D8E8G8|C32|
V: Ins
Z2|
'''


class Contract(unittest.TestCase):
    def setUp(self):
        self.request = {"style": "English piano pop", "lyrics": "[Chorus]\nStay with me", "seed": 42}
        self.path = SKILL / "assets/prompt.json"

    def config(self, **kwargs):
        return configure(self.request, self.path, **kwargs)

    def test_profiles_preserve_full_song_budget(self):
        for profile, steps in (("fast", 8), ("standard", 32), ("reference", 32)):
            with self.subTest(profile=profile):
                result = self.config(profile=profile)
                config = GenerationConfig.from_dict(result["generation_config"])
                self.assertEqual(config.ode_steps, steps)
                self.assertEqual(config.semantic.max_tokens, 9000)
                self.assertEqual(config.abc.max_tokens, 4096)

    def test_five_modes(self):
        for mode, abc in (("full", None), ("full", ABC), ("melody", None),
                          ("melody", ABC.replace('"C"', '')), ("off", None)):
            with self.subTest(mode=mode, supplied=bool(abc)):
                self.request.update(cot=mode, abc=abc)
                self.assertEqual(self.config()["abc"], abc)

    def test_user_overrides_win_and_input_not_mutated(self):
        self.request.update(generation_config={"ode_steps": 16}, abc_sampling={"temperature": 0.6}, cfg_scale=1.2)
        before = copy.deepcopy(self.request)
        result = self.config(profile="fast")
        self.assertEqual(result["generation_config"]["ode_steps"], 16)
        self.assertEqual(result["abc_sampling"], {"temperature": 0.6})
        self.assertEqual(self.request, before)

    def test_incompatible_modes_fail(self):
        for data in ({"cot": "off", "abc": ABC}, {"cot": "melody", "abc": ABC}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    configure({**self.request, **data}, self.path)

    def test_reject_foreign_controls(self):
        for key in ("bpm", "duration", "instrumental", "thinking", "repaint", "shift", "phonemes"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                configure({**self.request, key: 1}, self.path)

    def test_sampling_bounds(self):
        for sample in ({"top_k": 0}, {"max_tokens": 0}, {"temperature": float("nan")}, {"penalty_window": 101}):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                configure({**self.request, "semantic_sampling": sample}, self.path)

    def test_explicit_preview_is_labelled_by_call_and_bounded(self):
        result = self.config(preview_seconds=1)
        self.assertEqual(result["semantic_sampling"], {"max_tokens": 25, "min_tokens": 25})
        for seconds in (-1, 0, float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.config(preview_seconds=seconds)

    def test_cover_task_matching(self):
        for task, mode in (("full", "full"), ("melody-full", "melody"), ("melody-vocal", "melody")):
            with self.subTest(task=task):
                self.assertEqual(self.config(action="cover", task=task)["cot"], mode)
        self.request.update(cot="off")
        with self.assertRaises(ValueError):
            self.config(action="cover")
        with self.assertRaises(ValueError):
            self.config(action="plan")

    def test_abc_path_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            abc = Path(tmp) / "score.abc"
            abc.write_bytes(ABC.replace("\n", "\r\n").encode())
            result = configure({**self.request, "abc_path": "score.abc"}, Path(tmp) / "req.json")
            self.assertEqual(result["abc"].encode(), abc.read_bytes())
            with self.assertRaises(ValueError):
                configure({**self.request, "abc": ABC, "abc_path": str(abc)}, self.path)

    def test_native_cli_accepts_covered_commands(self):
        examples = [
            ["generate", "req.json"], ["plan", "req.json"], ["render-plan", "saved-plan"],
            ["replay", "saved-song", "--stage", "synthesize"],
            ["cover", "req.json", "--audio", "source.wav", "--task", "melody-full"],
            ["batch", "--input", "requests.jsonl", "--concurrency", "1"],
        ]
        for args in examples:
            with self.subTest(command=args[0]):
                parser().parse_args(args + ["--output", "out", "--model", "models", "--vae", "vae",
                                           "--precision", "8bit", "--offline", "--require-ac"])

    def test_dry_run_writes_nothing_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "unused"
            cmd = [sys.executable, str(SKILL / "scripts/generate_music.py"), "--request", str(self.path),
                   "--output", str(out), "--profile", "fast"]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["effective_request"]["generation_config"]["ode_steps"], 8)
            self.assertFalse(out.exists())
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("user work")
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(marker.read_text(), "user work")


if __name__ == "__main__":
    unittest.main()
