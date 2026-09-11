"""Saved plans and song artifacts are executable inputs, not display text."""
import json

import numpy as np
import pytest

from lyra.artifacts import (
    load_artifacts,
    load_plan_artifacts,
    save_plan_artifacts,
)
from lyra.pipeline import SongResult
from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.protocol import GenerationConfig, SongRequest
from yue2.storage import identity


def saved_song(tmp_path):
    # The score text intentionally does not define how to reconstruct its BPE.
    # A restored plan must retain the provided native IDs as authoritative.
    request = SongRequest("piano", "[Verse]\nA little light", abc="X:1\r\nK:C\r\nCDEF|\r\n")
    plan = SymbolicPlan(request, request.abc, [87, 12, 219], [151643, 5, 151847, 87, 12, 219, 151848, 151851])
    semantic = SemanticResult(plan, [0, 32767], {"seconds": 1.0}, True)
    config = {"generation": GenerationConfig().to_dict(), "validation_status": "unvalidated"}
    weights = {"mot": {"test": "generator"}, "vae": {"test": "decoder"}}
    timing = {
        "abc": plan.timing,
        "semantic": semantic.timing,
        "nar_seconds": 0.1,
        "vae_seconds": 0.1,
    }
    stamp = identity({
        "request": request.to_dict(),
        "config": config,
        "weights": weights,
    })
    result = SongResult(
        np.zeros((1920 * 2 - 64, 2), dtype=np.float32),
        48000,
        semantic,
        np.zeros((2, 64), dtype=np.float32),
        config,
        weights,
        timing,
        stamp,
        np.ones((2, 64), dtype=np.float32),
    )
    directory = tmp_path / "song"
    result.save_artifacts(directory)
    return directory, result


def test_restored_inputs_preserve_native_score_and_codec_boundaries(tmp_path):
    directory, original = saved_song(tmp_path)
    restored = load_artifacts(directory)
    assert restored.semantic.plan.abc == original.abc
    assert restored.semantic.plan.abc_ids == [87, 12, 219]
    assert restored.semantic.plan.prefix == original.semantic.plan.prefix
    assert restored.semantic.tokens == [0, 32767]
    assert restored.semantic.truncated is True
    np.testing.assert_array_equal(restored.noise, original.noise)
    assert restored.config == original.config
    assert restored.result["weights"] == original.weights


def test_saved_plan_retains_render_configuration_and_native_loader_compatibility(tmp_path):
    _, original = saved_song(tmp_path)
    config = GenerationConfig(ode_steps=17)
    directory = tmp_path / "plan"
    save_plan_artifacts(original.semantic.plan, directory, config)
    native = SymbolicPlan.load(directory)
    restored, restored_config = load_plan_artifacts(directory)
    assert native.prefix == original.semantic.plan.prefix
    assert restored.prefix == original.semantic.plan.prefix
    assert restored_config == config


def test_editing_saved_score_requires_new_external_input(tmp_path):
    directory, _ = saved_song(tmp_path)
    (directory / "score.abc").write_text("X:1\nK:G\nGABc|\n")
    with pytest.raises(ValueError):
        SymbolicPlan.load(directory)
    with pytest.raises(ValueError):
        load_artifacts(directory)


def test_changed_solver_noise_cannot_silently_replay(tmp_path):
    directory, original = saved_song(tmp_path)
    np.save(directory / "noise.npy", original.noise * 2)
    with pytest.raises(ValueError):
        load_artifacts(directory)


def test_result_receipt_cannot_change_weight_identity(tmp_path):
    directory, _ = saved_song(tmp_path)
    receipt = json.loads((directory / "result.json").read_text())
    receipt["weights"]["vae"] = {"test": "other-decoder"}
    (directory / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity disagrees"):
        load_artifacts(directory)


def test_replay_requires_semantic_stage_metadata(tmp_path):
    directory, _ = saved_song(tmp_path)
    receipt = json.loads((directory / "result.json").read_text())
    del receipt["timing"]["semantic"]
    (directory / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="stage timing"):
        load_artifacts(directory)


def test_export_rejects_empty_song_before_creating_artifacts(tmp_path):
    _, original = saved_song(tmp_path)
    original.semantic.tokens.clear()
    original.latents = np.empty((0, 64), dtype=np.float32)
    original.noise = np.empty((0, 64), dtype=np.float32)
    original.audio = np.empty((0, 2), dtype=np.float32)
    destination = tmp_path / "empty"
    with pytest.raises(ValueError, match="codec frames"):
        original.save_artifacts(destination)
    assert not destination.exists()


def test_artifacts_without_retained_solver_noise_remain_loadable(tmp_path):
    _, original = saved_song(tmp_path)
    original.noise = None
    directory = tmp_path / "without-noise"
    original.save_artifacts(directory)
    assert load_artifacts(directory).noise is None


def test_export_refuses_mixed_recording_directory(tmp_path):
    directory, original = saved_song(tmp_path)
    with pytest.raises(FileExistsError):
        original.save_artifacts(directory)
    assert json.loads((directory / "result.json").read_text())["identity"] == original.request_identity
