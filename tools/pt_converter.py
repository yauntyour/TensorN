#!/usr/bin/env python3
"""
Bridge between PyTorch tensors and TensorN .pt binary format.

TensorN .pt binary format:
  - Magic:    "TENSORPT!" (9 bytes)
  - Version:  uint32 LE (4 bytes)
  - Dtype:    uint8        (1 byte):  0=f32, 1=f64, 2=i32, 3=i64, 4=u8, 5=i16
  - Ndims:    uint32 LE (4 bytes)
  - Shape:    uint64 LE[] (ndims * 8 bytes)
  - Data:     raw binary, row-major, LE

Usage:
  python pt_converter.py np2pt   <input.npy>    <output.pt>
  python pt_converter.py pt2np   <input.pt>     <output.npy>
  python pt_converter.py torch2pt  <input.pth>  <output.pt>
  python pt_converter.py pt2torch  <input.pt>   <output.pth>
"""

import struct
import sys
import numpy as np

MAGIC = b"TENSORPT!"
VERSION = 1

DTYPE_TO_ENUM = {
    np.dtype("float32"): 0,
    np.dtype("float64"): 1,
    np.dtype("int32"):   2,
    np.dtype("int64"):   3,
    np.dtype("uint8"):   4,
    np.dtype("int16"):   5,
}

ENUM_TO_DTYPE = {v: k for k, v in DTYPE_TO_ENUM.items()}


def save_tensorn_pt(filename: str, array: np.ndarray) -> None:
    dtype_enum = DTYPE_TO_ENUM.get(array.dtype)
    if dtype_enum is None:
        raise ValueError(
            f"Unsupported dtype: {array.dtype}. "
            f"Supported: {list(DTYPE_TO_ENUM.keys())}"
        )

    with open(filename, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<B", dtype_enum))
        f.write(struct.pack("<I", array.ndim))
        for dim in array.shape:
            f.write(struct.pack("<Q", dim))
        f.write(array.tobytes())


def load_tensorn_pt(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        magic = f.read(9)
        if magic != MAGIC:
            raise ValueError(
                f"Not a valid TensorN .pt file. "
                f"Expected magic {MAGIC!r}, got {magic!r}"
            )

        version = struct.unpack("<I", f.read(4))[0]
        if version != VERSION:
            raise ValueError(
                f"Unsupported version: {version}. Expected {VERSION}"
            )

        dtype_enum = struct.unpack("<B", f.read(1))[0]
        ndims = struct.unpack("<I", f.read(4))[0]

        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack("<Q", f.read(8))[0])

        dtype = ENUM_TO_DTYPE.get(dtype_enum)
        if dtype is None:
            raise ValueError(f"Unknown dtype enum: {dtype_enum}")

        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(shape)


def np2pt(input_npy: str, output_pt: str) -> None:
    arr = np.load(input_npy)
    save_tensorn_pt(output_pt, arr)
    print(f"Converted {input_npy} -> {output_pt}")


def pt2np(input_pt: str, output_npy: str) -> None:
    arr = load_tensorn_pt(input_pt)
    np.save(output_npy, arr)
    print(f"Converted {input_pt} -> {output_npy}")


def torch2pt(input_pth: str, output_pt: str) -> None:
    try:
        import torch
    except ImportError:
        print("Error: torch is required for this command. Install with: pip install torch")
        sys.exit(1)

    tensor = torch.load(input_pth, weights_only=True)
    if isinstance(tensor, dict):
        print(f"Warning: loaded a dict with keys: {list(tensor.keys())}. Using the first tensor found.")
        for v in tensor.values():
            if torch.is_tensor(v):
                tensor = v
                break
        else:
            raise ValueError("No tensor found in the loaded dict.")

    if not torch.is_tensor(tensor):
        raise ValueError(f"Expected a torch.Tensor, got {type(tensor)}")

    arr = tensor.detach().cpu().numpy()
    save_tensorn_pt(output_pt, arr)
    print(f"Converted {input_pth} -> {output_pt}")


def pt2torch(input_pt: str, output_pth: str) -> None:
    try:
        import torch
    except ImportError:
        print("Error: torch is required for this command. Install with: pip install torch")
        sys.exit(1)

    arr = load_tensorn_pt(input_pt)
    tensor = torch.from_numpy(arr)
    torch.save(tensor, output_pth)
    print(f"Converted {input_pt} -> {output_pth}")


def print_usage():
    print(__doc__)


COMMANDS = {
    "np2pt":    (np2pt,    2, "<input.npy> <output.pt>"),
    "pt2np":    (pt2np,    2, "<input.pt> <output.npy>"),
    "torch2pt": (torch2pt, 2, "<input.pth> <output.pt>"),
    "pt2torch": (pt2torch, 2, "<input.pt> <output.pth>"),
}


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)

    fn, nargs, usage = COMMANDS[cmd]
    args = sys.argv[2:]
    if len(args) != nargs:
        print(f"Usage: python pt_converter.py {cmd} {usage}")
        sys.exit(1)

    try:
        fn(*args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
