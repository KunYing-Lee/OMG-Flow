"""Frozen T5 caption embeddings shared by every training worker via a memory map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from omg.generation.conditions.t5 import FrozenT5TextEncoder


T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
# SHA256 of the files from google-t5/t5-base at T5_REVISION.
T5_SHA256 = {
    "config.json": "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033",
    "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    "tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    "model.safetensors": "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
}
MAX_TOKENS = 50
HIDDEN_DIM = 768


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BonesTextCache:
    def __init__(self, root, *, verify_files=False):
        root = Path(root)
        manifest_bytes = (root / "text.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if (manifest["encoder"], manifest["revision"], manifest["max_length"], manifest["encoder_sha256"]) != (
            "google-t5/t5-base", T5_REVISION, MAX_TOKENS, T5_SHA256
        ):
            raise ValueError("Unexpected caption encoder contract")
        self.signature = hashlib.sha256(manifest_bytes).hexdigest()
        self.ids = {text: idx for idx, text in enumerate(manifest["captions"])}
        if len(self.ids) != len(manifest["captions"]):
            raise ValueError("Caption cache contains duplicate captions")
        for name in ("tokens.npy", "lengths.npy"):
            if verify_files and _file_sha256(root / name) != manifest["files_sha256"][name]:
                raise ValueError(f"Caption cache checksum mismatch: {name}")
        self.tokens = torch.from_numpy(np.load(root / "tokens.npy", mmap_mode="c", allow_pickle=False))
        self.lengths = torch.from_numpy(np.load(root / "lengths.npy", mmap_mode="c", allow_pickle=False))
        if (self.tokens.shape != (len(self.ids), MAX_TOKENS, HIDDEN_DIM)
                or self.tokens.dtype != torch.float16 or self.lengths.shape != (len(self.ids),)
                or self.lengths.dtype != torch.uint8 or not torch.all((self.lengths > 0) & (self.lengths <= MAX_TOKENS))):
            raise ValueError("Caption cache shape, dtype or lengths differ from the encoder contract")

    def lookup(self, captions):
        ids = torch.tensor([self.ids[text] for text in captions], dtype=torch.long)
        return self.tokens[ids], torch.arange(MAX_TOKENS) < self.lengths[ids, None]


@torch.inference_mode()
def prepare_cache(captions, destination, model_path, device, batch_size):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    captions = sorted(set(captions))
    if not captions or batch_size < 1:
        raise ValueError("A nonempty caption list and positive batch size are required")
    if (destination / "text.json").exists():
        cache = BonesTextCache(destination, verify_files=True)
        if list(cache.ids) != captions:
            raise ValueError("A different caption list already exists at this destination")
        return

    # Load only verified snapshot files: local tokenizer overrides or alternate
    # weight files must not silently change the embeddings recorded under this revision.
    with TemporaryDirectory() as snapshot:
        for name, expected in T5_SHA256.items():
            path = (Path(model_path) / name).resolve(strict=True)
            if _file_sha256(path) != expected:
                raise ValueError(f"Unexpected SHA256 for {name} at pinned T5 revision {T5_REVISION}")
            (Path(snapshot) / name).symlink_to(path)
        encoder = FrozenT5TextEncoder(model_name=snapshot, frozen_eval=True, max_length=MAX_TOKENS).to(device).eval()
        device = torch.device(device)
        lengths = np.empty(len(captions), dtype=np.uint8)
        # Stream one batch at a time. Only readers map the complete embedding table.
        with (destination / "tokens.npy").open("wb") as output:
            np.lib.format.write_array_header_2_0(output, {
                "descr": np.dtype(np.float16).str, "fortran_order": False,
                "shape": (len(captions), MAX_TOKENS, HIDDEN_DIM),
            })
            for start in range(0, len(captions), batch_size):
                texts = captions[start:start + batch_size]
                with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    batch = encoder(texts, device=device)
                mask = batch["mask"].cpu()
                length = mask.sum(1)
                rows = batch["context"].half().cpu()
                if (rows.shape != (len(texts), MAX_TOKENS, HIDDEN_DIM) or not torch.isfinite(rows).all()
                        or not torch.all(length > 0) or not torch.equal(mask, torch.arange(MAX_TOKENS) < length[:, None])):
                    raise ValueError("T5 produced invalid embeddings or a non-right-padded attention mask")
                rows.numpy().tofile(output)
                lengths[start:start + len(texts)] = length.numpy()
                if start % (batch_size * 100) == 0:
                    print(f"encoded {start + len(texts)}/{len(captions)}", flush=True)
    np.save(destination / "lengths.npy", lengths)
    manifest = dict(
        encoder="google-t5/t5-base", revision=T5_REVISION, encoder_sha256=T5_SHA256,
        max_length=MAX_TOKENS, captions=captions,
        files_sha256={name: _file_sha256(destination / name) for name in ("tokens.npy", "lengths.npy")},
    )
    (destination / "text.tmp").write_text(json.dumps(manifest, ensure_ascii=False))
    (destination / "text.tmp").replace(destination / "text.json")
