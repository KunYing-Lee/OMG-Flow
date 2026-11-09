"""Build or validate the cached batch-one Flow velocity FP16 engine."""

import argparse
import json
from pathlib import Path

from deployment.sync_flow import FlowGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    generator = FlowGenerator({**json.loads(args.config.read_text()), "backend": "tensorrt-fp16"})
    print(json.dumps({"engine": str(generator.trt.path), "cache_hit": generator.trt.cache_hit,
                      "contract": generator.trt.contract}, indent=2))


if __name__ == "__main__":
    main()
