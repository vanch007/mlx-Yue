import json
from types import SimpleNamespace

import pytest

from lyra.cli import _request, main, parser
from lyra.commands import resume_result
from test_artifacts import saved_song


def test_inline_overrides_and_relative_abc(tmp_path):
    (tmp_path / "score.abc").write_bytes(b"X:1\r\nK:C\r\nCDEF|\r\n")
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"style": "pop", "lyrics": "hello", "abc_path": "score.abc"}))
    args = parser().parse_args(["generate", "--request", str(path), "--style", "jazz",
                               "--cot", "melody", "--seed", "42", "--output", "unused"])
    request = _request(args)
    assert request["style"] == "jazz" and request["seed"] == 42
    assert request["abc"] == "X:1\r\nK:C\r\nCDEF|\r\n"
    assert request["cot"] == "melody"


def test_resume_rejects_changed_request_and_modified_artifacts(tmp_path):
    path, original = saved_song(tmp_path)
    pipe = SimpleNamespace(_request=lambda **kwargs: original.semantic.plan.request.__class__(**kwargs),
                           effective_config=lambda *args: original.config, weights=original.weights)
    request = original.semantic.plan.request.to_dict()
    assert resume_result(pipe, request, path)["resumed"]
    with pytest.raises(ValueError):
        resume_result(pipe, dict(request, style="different"), path)
    (path / "audio.flac").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        resume_result(pipe, request, path)


def test_batch_rejects_duplicate_ids_before_loading_model(tmp_path):
    path = tmp_path / "batch.jsonl"
    path.write_text('{"id":"same"}\n{"id":"same"}\n')
    with pytest.raises(ValueError, match="unique"):
        main(["batch", "--input", str(path), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
