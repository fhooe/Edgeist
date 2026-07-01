#!/usr/bin/env python3
"""Lightweight host-side NMCF header validator.

The authoritative validator is the C++ runtime. This script is useful in Python
pipelines to catch obviously malformed files before invoking C++ tests or target
flashing.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = 0x46434D4E
HEADER_PREFIX = "<HI8sHHIHIHI"
HEADER_PREFIX_SIZE = struct.calcsize(HEADER_PREFIX)


def validate(model_path: Path, trainable_path: Path | None) -> list[str]:
    errors: list[str] = []
    data = model_path.read_bytes()
    if len(data) < HEADER_PREFIX_SIZE:
        return ["model is shorter than the fixed header"]
    header = struct.unpack_from(HEADER_PREFIX, data, 0)
    header_size, magic, version, layer_count, channels_in, dim_x, dim_y, channels_out, out_x, out_y = header
    if magic != MAGIC:
        errors.append(f"invalid magic: 0x{magic:08x}")
    if layer_count == 0 or layer_count > 256:
        errors.append(f"invalid layer count: {layer_count}")
    min_header = HEADER_PREFIX_SIZE + layer_count * 4
    if header_size < min_header or header_size > len(data):
        errors.append(f"invalid headerSize: {header_size}; expected [{min_header}, {len(data)}]")
    if channels_in == 0 or dim_x == 0 or dim_y == 0 or channels_out == 0 or out_x == 0 or out_y == 0:
        errors.append("zero tensor dimension in model header")
    offsets: list[int] = []
    for i in range(layer_count):
        pos = HEADER_PREFIX_SIZE + i * 4
        if pos + 4 > len(data):
            errors.append("offset table is truncated")
            break
        (off,) = struct.unpack_from("<I", data, pos)
        offsets.append(off)
        if off < header_size or off >= len(data):
            errors.append(f"layer {i} offset {off} outside model image")
        if i and off <= offsets[i - 1]:
            errors.append(f"layer {i} offset is not strictly increasing")
    trainable_size = trainable_path.stat().st_size if trainable_path is not None else 0
    print(f"version={version!r} layers={layer_count} model_bytes={len(data)} trainable_bytes={trainable_size}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trainable", type=Path)
    args = parser.parse_args()
    errors = validate(args.model, args.trainable)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("basic Python validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
