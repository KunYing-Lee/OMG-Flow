"""Validated JSON configuration; paths are relative to the configuration file."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

from .schedule import Schedule


@dataclass(frozen=True)
class StartGate:
    joint_rad: float = 0.3
    height_m: float = 0.10
    yaw_deg: float = 10.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"start_gate.{field.name} must be a finite nonnegative number")


@dataclass(frozen=True)
class StartHistory:
    path: Path
    key: str


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    prompt: str
    steps: int
    lowpass_hz: float | None
    start_history: StartHistory
    play_frames: int
    seed: int
    episode_id: int
    backend: str
    device: str
    text_cache: Path | None = None
    text_encoder: Path | None = None
    engine_directory: Path | None = None
    noise_policy: Path | None = None
    robot_xml: Path | None = None
    start_gate: StartGate = StartGate()
    switch_seconds: float = 0.5
    after_end: str = "hold"


def load_config(path, **overrides):
    """Read and validate once before constructing any model or opening a socket.

    Overrides use the same schema as JSON. Path strings, including overridden
    paths, resolve against the JSON file's directory.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    allowed = {field.name for field in fields(Config)}
    unknown = (data.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    data.update(overrides)
    required = {"checkpoint", "prompt", "steps", "lowpass_hz", "start_history",
                "play_frames", "seed", "episode_id", "backend", "device"}
    if required - data.keys():
        raise ValueError(f"Missing required config keys: {sorted(required - data.keys())}")
    for key in ("prompt", "backend", "device"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("steps", "play_frames", "seed", "episode_id"):
        if type(data[key]) is not int or data[key] < (1 if key in ("steps", "play_frames") else 0):
            raise ValueError(f"{key} must be a {'positive' if key in ('steps', 'play_frames') else 'nonnegative'} integer")
    for key in ("seed", "episode_id"):
        if data[key] >= 2**63:
            raise ValueError(f"{key} must be in [0, 2**63)")
    cutoff = data["lowpass_hz"]
    if cutoff is not None and (type(cutoff) not in (int, float) or not 0 < cutoff < 25):
        raise ValueError("lowpass_hz must be null or a number in (0, 25)")
    if data["backend"] not in ("torch-fp32", "torch-fp16", "tensorrt-fp16"):
        raise ValueError("backend must be torch-fp32, torch-fp16 or tensorrt-fp16")
    if data["backend"] != "torch-fp32" and not data["device"].startswith("cuda"):
        raise ValueError("FP16 backends require a CUDA device")
    if (data.get("text_cache") is None) == (data.get("text_encoder") is None):
        raise ValueError("Provide exactly one of text_cache or text_encoder")
    for key in ("checkpoint", "text_cache", "text_encoder", "engine_directory", "noise_policy", "robot_xml"):
        value = data.get(key)
        if value is None and key != "checkpoint":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        data[key] = (path.parent / value).resolve()
    if data["backend"] == "tensorrt-fp16" and data.get("engine_directory") is None:
        raise ValueError("tensorrt-fp16 requires engine_directory")
    history = data["start_history"]
    if not isinstance(history, dict) or history.keys() != {"path", "key"}:
        raise ValueError("start_history must contain exactly path and key")
    for key, value in history.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"start_history.{key} must be a nonempty string")
    data["start_history"] = StartHistory((path.parent / history["path"]).resolve(), history["key"])
    gate = data.get("start_gate", {})
    if not isinstance(gate, dict) or gate.keys() - {field.name for field in fields(StartGate)}:
        raise ValueError("start_gate must be an object with only joint_rad, height_m and yaw_deg")
    data["start_gate"] = StartGate(**gate)
    seconds = data.get("switch_seconds", 0.5)
    if type(seconds) not in (int, float):
        raise ValueError("switch_seconds must be a number")
    Schedule(seconds)
    if data.get("after_end", "hold") not in ("hold", "error"):
        raise ValueError("after_end must be hold or error")
    return Config(**data)
