"""Static batch-one Flow velocity engine, ported from Posttrain 4f7ff92.

Only the velocity network is exported. Codec, context, Euler and filter stay
outside TensorRT; one engine serves both step counts and any prefix/prompt with
matching tensor shapes. No implicit backend substitution.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class VelocityExport(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, state, time, committed, context, context_mask):
        return self.flow._velocity(state, time, committed, context, context_mask)


class FlowTRT:
    def __init__(self, reference, tokens, mask, checkpoint_sha256, cache_dir):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        context, context_mask = reference.model._context(
            torch.zeros(1, reference.history, 38, device=reference.mean.device), tokens, mask)
        self.inputs = {
            "state": torch.zeros(1, reference.horizon, 38, device=reference.mean.device),
            "time": torch.zeros(1, device=reference.mean.device),
            "committed": torch.zeros(1, reference.horizon, device=reference.mean.device, dtype=torch.bool),
            "context": context.float().contiguous(), "context_mask": context_mask.contiguous(),
        }
        gpu = torch.cuda.get_device_properties(reference.mean.device)
        self.contract = {
            "format": "omg-flow-velocity-fp16-v1", "checkpoint_sha256": checkpoint_sha256,
            "gpu": gpu.name, "gpu_uuid": str(gpu.uuid), "capability": [gpu.major, gpu.minor],
            "tensorrt": trt.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in self.inputs.items()},
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__).relative_to(Path.cwd()), Path("src/omg/generation/flow.py"),
                Path("src/omg/generation/denoisers/transformer.py"), Path("src/omg/generation/conditions/position.py"))},
            "opset": 18, "workspace_bytes": 4 << 30,
        }
        key = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        self.path = Path(cache_dir) / checkpoint_sha256 / str(gpu.uuid) / key / "velocity.engine"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_hit = self.path.is_file()
        if self.cache_hit:
            receipt = json.loads(self.path.with_suffix(".json").read_text())
            if receipt["contract"] != self.contract or receipt["engine_sha256"] != hashlib.sha256(self.path.read_bytes()).hexdigest():
                raise ValueError("TensorRT cache contract or engine hash mismatch")
        else:
            # A CPU FP32 copy avoids ONNX constant-folding device mismatches.
            cpu = copy.deepcopy(reference.model).cpu().float().eval()
            with torch.inference_mode():
                torch.onnx.export(VelocityExport(cpu), tuple(v.cpu() for v in self.inputs.values()),
                                  str(self.path.with_suffix(".onnx")), opset_version=18,
                                  input_names=list(self.inputs), output_names=["velocity"], dynamo=False)
            builder = trt.Builder(self.logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse_from_file(str(self.path.with_suffix(".onnx"))):
                raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            settings = builder.create_builder_config()
            settings.set_flag(trt.BuilderFlag.FP16)
            settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            engine = builder.build_serialized_network(network, settings)
            if engine is None:
                raise RuntimeError("TensorRT velocity build failed")
            self.path.write_bytes(bytes(engine))
            self.path.with_suffix(".json").write_text(json.dumps({
                "contract": self.contract, "engine_sha256": hashlib.sha256(bytes(engine)).hexdigest(),
            }, indent=2) + "\n")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT velocity deserialization failed")
        self.context = self.engine.create_execution_context()
        self.output = torch.empty_like(self.inputs["state"])
        bindings = {**self.inputs, "velocity": self.output}
        if {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)} != bindings.keys():
            raise ValueError("Unexpected velocity engine I/O names")
        for name, tensor in bindings.items():
            dtype = torch.from_numpy(np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if tuple(self.engine.get_tensor_shape(name)) != tuple(tensor.shape) or dtype != tensor.dtype:
                raise ValueError(f"Unexpected velocity engine binding: {name}")
            self.context.set_tensor_address(name, tensor.data_ptr())

    def __call__(self, state, time, committed, context, context_mask):
        for target, value in zip(self.inputs.values(), (state, time, committed, context, context_mask)):
            target.copy_(value)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT velocity execution failed")
        return self.output
