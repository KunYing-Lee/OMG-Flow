"""Build the frozen caption cache once, before starting torchrun."""
import argparse

from omg.data.bones import load_bones_captions
from omg.generation.bones_text import prepare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--t5", required=True, help="Local snapshot of pinned google-t5/t5-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    prepare_cache(load_bones_captions(args.data), args.output, args.t5, args.device, args.batch_size)


if __name__ == "__main__":
    main()
