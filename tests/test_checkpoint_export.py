"""A native pipeline export must not require importing the Torch reference."""
import builtins
import json

import pytest

from lyra.checkpoints import copy_vae_checkpoint
from yue2.storage import model_identity


def test_vae_export_without_torch_retains_identity_and_file_boundary(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "saved"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "yue2_vae", "decoder_config": {}}))
    (source / "model.safetensors").write_bytes(b"checkpoint fixture")
    (source / "modeling_vae.py").write_text("raise RuntimeError('unreviewed source code')")
    (source / "private.txt").write_text("not part of the checkpoint")
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Native export imported Torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    copy_vae_checkpoint(source, target)
    assert model_identity(source) == model_identity(target)
    assert not (target / "private.txt").exists()
    assert "unreviewed source code" not in (target / "modeling_vae.py").read_text()
    with pytest.raises(FileExistsError):
        copy_vae_checkpoint(source, target)


def test_vae_export_rejects_unexpected_weight_names_before_writing(tmp_path):
    source, target = tmp_path / "source", tmp_path / "saved"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "yue2_vae", "decoder_config": {}}))
    (source / "unexpected.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="filename"):
        copy_vae_checkpoint(source, target)
    assert not target.exists()
