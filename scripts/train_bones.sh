#!/usr/bin/env bash
# Run in a dedicated four-GPU allocation with a complete prepared dataset.
set -euo pipefail
: "${BONES_DATA:?Path to complete prepared BONES-SEED}"
: "${BONES_TEXT_CACHE:?Path to the verified frozen T5 cache}"
: "${FLOW_OUTPUT:?Persistent run directory}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python="${PYTHON:-python}"
mkdir -p "$FLOW_OUTPUT"
trainer=("$python" -m torch.distributed.run --standalone --nproc_per_node=4
         -m omg.cli.generation.train --data "$BONES_DATA" --text-cache "$BONES_TEXT_CACHE"
         --output "$FLOW_OUTPUT")
if [[ -n "${FLOW_CONFIG:-}" ]]; then trainer+=(--config "$FLOW_CONFIG"); fi
"$python" - "$BONES_DATA" "$BONES_TEXT_CACHE" <<'PY'
import hashlib, json, sys
from pathlib import Path
from omg.generation.bones_text import BonesTextCache
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
if not manifest["complete"]:
    raise ValueError("Dedicated long-run launch requires the full source release")
for name, key in (("clips.parquet", "index_sha256"), ("qpos.f32", "qpos_sha256"), ("captions.json", "captions_sha256")):
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != manifest[key]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
BonesTextCache(sys.argv[2], verify_files=True)
PY
if [[ ! -e "$FLOW_OUTPUT/latest.pt" ]]; then
    "${trainer[@]}" --stop-after 12
fi
read -r step target < <("$python" - "$FLOW_OUTPUT/latest.pt" <<'PY'
import os, sys, torch, yaml
saved = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
with open(os.environ.get("FLOW_CONFIG") or "configs/flow/bones.yaml") as config:
    print(saved["step"], yaml.safe_load(config)["steps"])
PY
)
if ((step == target)); then exit 0; fi
if ((step < 16)); then
    "${trainer[@]}" --resume --stop-after 16
fi
exec "${trainer[@]}" --resume
