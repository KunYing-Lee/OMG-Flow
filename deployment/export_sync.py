"""Build the fixed-shape, fixed-prompt OMG-Flow FP16 engine used by sync_flow.py."""

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch

from deployment.sync_flow import FlowGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    path = Path(config["engine_path"])
    generator = FlowGenerator({**config, "engine_path": None})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the public Flow reference API, including its world-frame codec, and
    # unroll precisely the configured Euler steps. No diffusion/Lite exporter.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        torch.onnx.export(
            generator, tuple(generator.inputs.values()), str(path.with_suffix(".onnx")),
            input_names=list(generator.inputs), output_names=["suffix"],
            opset_version=17, dynamo=False,
        )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path.with_suffix(".onnx"))):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    settings = builder.create_builder_config()
    settings.set_flag(trt.BuilderFlag.FP16)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    engine = builder.build_serialized_network(network, settings)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    path.write_bytes(bytes(engine))
    path.with_suffix(".json").write_text(json.dumps(generator.contract, indent=2) + "\n")
    print(f"Built {path}; run deployment.check_sync_backend before motor use")


if __name__ == "__main__":
    main()
