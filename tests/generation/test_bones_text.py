import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from omg.generation import bones_text


class TinyTextEncoder(nn.Module):
    seen = []

    def __init__(self, *, model_name, frozen_eval, max_length):
        super().__init__()
        assert frozen_eval and max_length == 50
        from pathlib import Path
        assert set(p.name for p in Path(model_name).iterdir()) == set(bones_text.T5_SHA256)

    def forward(self, texts, device):
        self.seen.append(tuple(texts))
        lengths = torch.tensor([min(50, len(text) + 1) for text in texts])
        return {
            "context": lengths[:, None, None].float().expand(len(texts), 50, 768),
            "mask": torch.arange(50) < lengths[:, None],
        }


def test_cache_uses_pinned_assets_bounded_batches_and_mapped_rows(tmp_path, monkeypatch):
    model = tmp_path / "source"
    model.mkdir()
    expected = {}
    for name in bones_text.T5_SHA256:
        (model / name).write_bytes(name.encode())
        expected[name] = hashlib.sha256(name.encode()).hexdigest()
    # The builder presents only verified inputs to the encoder.
    (model / "tokenizer_config.json").write_text("this is not the approved snapshot")
    monkeypatch.setattr(bones_text, "T5_SHA256", expected)
    monkeypatch.setattr(bones_text, "FrozenT5TextEncoder", TinyTextEncoder)
    TinyTextEncoder.seen.clear()
    captions = ["ten letters", "", "a", "a", "b", "x" * 100]
    output = tmp_path / "cache"
    bones_text.prepare_cache(captions, output, model, "cpu", 2)
    assert TinyTextEncoder.seen == [("", "a"), ("b", "ten letters"), ("x" * 100,)]

    modes = []
    load = np.load

    def verify_mmap(*args, **kwargs):
        modes.append(kwargs["mmap_mode"])
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", verify_mmap)
    cache = bones_text.BonesTextCache(output, verify_files=True)
    assert modes == ["c", "c"]
    context, mask = cache.lookup(["b", "x" * 100, "b", ""])
    assert context.dtype == torch.float16 and context.shape == (4, 50, 768)
    assert torch.equal(mask.sum(1), torch.tensor([2, 50, 2, 1]))
    assert torch.equal(context[:, 0, 0], mask.sum(1).half())
    # Reuse does not encode again; it does verify the persisted complete payload.
    bones_text.prepare_cache(captions, output, model, "cpu", 2)
    assert len(TinyTextEncoder.seen) == 3
    with (output / "tokens.npy").open("r+b") as f:
        f.seek(-2, 2)
        f.write(b"ZZ")
    with pytest.raises(ValueError, match="checksum"):
        bones_text.prepare_cache(captions, output, model, "cpu", 2)


def test_wrong_weight_snapshot_is_rejected_before_encoding(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    dest = tmp_path / "target"
    with pytest.raises(ValueError, match="Unexpected SHA256"):
        bones_text.prepare_cache(["walk"], dest, tmp_path, "cpu", 2)
    assert not (dest / "text.json").exists()
