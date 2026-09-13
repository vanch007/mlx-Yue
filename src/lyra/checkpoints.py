"""Export native-runtime checkpoints without importing a reference backend."""
import json
from pathlib import Path
import shutil

from yue2 import storage


def copy_vae_checkpoint(source, destination):
    """Copy verified VAE data and reviewed compatibility code to an empty path.

    Configuration bytes are retained so checkpoint identity survives export.
    Reading that JSON or copying the installed compatibility module never imports
    its optional Torch backend, and arbitrary source-folder code is not exported.
    """
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("save_pretrained needs an empty VAE destination")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "yue2_vae" or not isinstance(config.get("decoder_config"), dict):
        raise ValueError("Expected a YuE2 VAE checkpoint configuration")
    weights = storage.model_identity(source)
    names = {"config.json", "model.safetensors", "model.safetensors.index.json",
             "LICENSE", "THIRD_PARTY_NOTICES.md"}
    names.update(name for name in weights["files"] if storage.SHARD_NAME.fullmatch(name))
    if set(weights["files"]) - names:
        raise ValueError("Unexpected VAE weight filename")
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    shutil.copyfile(Path(storage.__file__).with_name("modeling_vae.py"), destination / "modeling_vae.py")
    for name in sorted(storage.MODEL_LICENSES):
        if (source / "licenses" / name).is_file():
            (destination / "licenses").mkdir(exist_ok=True)
            shutil.copyfile(source / "licenses" / name, destination / "licenses" / name)
    storage.write_json(destination / "weights_manifest.json", {"files": weights["files"]})
