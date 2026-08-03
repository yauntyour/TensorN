#!/usr/bin/env python3
"""
Bridge between PyTorch tensors and TensorN .pt binary format.

TensorN .pt binary format (Version 1 - single tensor):
  - Magic:    "TENSORPT!" (9 bytes)
  - Version:  uint32 LE (4 bytes)
  - Dtype:    uint8        (1 byte):  0=f32, 1=f64, 2=i32, 3=i64, 4=u8, 5=i16
  - Ndims:    uint32 LE (4 bytes)
  - Shape:    uint64 LE[] (ndims * 8 bytes)
  - Data:     raw binary, row-major, LE

TensorN .pt binary format (Version 2 - multi-tensor):
  - Magic:        "TENSORPT!" (9 bytes)
  - Version:      uint32 LE (=2) (4 bytes)
  - Tensor Count: uint64 LE (8 bytes)
  - For each tensor:
      - Name Length: uint32 LE (4 bytes)
      - Name:        UTF-8 string
      - Dtype:       uint8 (1 byte)
      - Ndims:       uint32 LE (4 bytes)
      - Shape:       uint64 LE[]
      - Data Offset: uint64 LE (8 bytes)
      - Data Size:   uint64 LE (8 bytes)
  - Data:          raw binary for all tensors, row-major, LE

Usage:
  python pt_converter.py np2pt        <input.npy>     <output.pt>
  python pt_converter.py pt2np        <input.pt>      <output.npy>
  python pt_converter.py torch2pt     <input.pth>     <output.pt>
  python pt_converter.py pt2torch     <input.pt>      <output.pth>
  python pt_converter.py torch2pt_multi <input.pth>   <output.pt>
  python pt_converter.py pt_list      <input.pt>
"""

import struct
import sys
import numpy as np

MAGIC = b"TENSORPT!"
VERSION = 1
VERSION_MULTI = 2

def _build_dtype_map():
    m = {
        np.dtype("float32"): 0,
        np.dtype("float64"): 1,
        np.dtype("int32"):   2,
        np.dtype("int64"):   3,
        np.dtype("uint8"):   4,
        np.dtype("int16"):   5,
    }
    # extended low-precision types (depend on NumPy version)
    for name, enum in (("float16", 6), ("bfloat16", 7)):
        try:
            m[np.dtype(name)] = enum
        except TypeError:
            pass
    # 8: tf32 (no native NumPy dtype) — np.float8_e4m3fn / np.float8_e5m2 need NumPy >= 2.0
    for name, enum in (("float8_e4m3fn", 9), ("float8_e5m2", 10)):
        try:
            m[np.dtype(name)] = enum
        except TypeError:
            pass
    return m


DTYPE_TO_ENUM = _build_dtype_map()

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


def save_tensorn_pt_multi(filename: str, tensors: dict[str, np.ndarray]) -> None:
    if not tensors:
        raise ValueError("No tensors to save")

    with open(filename, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION_MULTI))
        f.write(struct.pack("<Q", len(tensors)))

        current_offset = 0
        entries = []
        for name, array in tensors.items():
            dtype_enum = DTYPE_TO_ENUM.get(array.dtype)
            if dtype_enum is None:
                raise ValueError(
                    f"Unsupported dtype for tensor '{name}': {array.dtype}. "
                    f"Supported: {list(DTYPE_TO_ENUM.keys())}"
                )
            raw_size = array.nbytes
            entries.append((name, array, dtype_enum, current_offset, raw_size))
            current_offset += raw_size

        for name, array, dtype_enum, offset, size in entries:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_enum))
            f.write(struct.pack("<I", array.ndim))
            for dim in array.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", size))

        for _, array, _, _, _ in entries:
            f.write(array.tobytes())


def load_tensorn_pt_multi(filename: str) -> dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        magic = f.read(9)
        if magic != MAGIC:
            raise ValueError(
                f"Not a valid TensorN .pt file. "
                f"Expected magic {MAGIC!r}, got {magic!r}"
            )

        version = struct.unpack("<I", f.read(4))[0]
        if version != VERSION_MULTI:
            if version == VERSION:
                arr = load_tensorn_pt(filename)
                return {"tensor": arr}
            raise ValueError(
                f"Not a multi-tensor .pt file (version={version})"
            )

        tensor_count = struct.unpack("<Q", f.read(8))[0]

        infos = []
        for _ in range(tensor_count):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_enum = struct.unpack("<B", f.read(1))[0]
            ndims = struct.unpack("<I", f.read(4))[0]
            shape = []
            for _ in range(ndims):
                shape.append(struct.unpack("<Q", f.read(8))[0])
            offset = struct.unpack("<Q", f.read(8))[0]
            size = struct.unpack("<Q", f.read(8))[0]
            infos.append((name, dtype_enum, shape, offset, size))

        data_base = f.tell()

        result = {}
        for name, dtype_enum, shape, offset, size in infos:
            dtype = ENUM_TO_DTYPE.get(dtype_enum)
            if dtype is None:
                raise ValueError(f"Unknown dtype enum: {dtype_enum}")
            f.seek(data_base + offset)
            data = f.read(size)
            result[name] = np.frombuffer(data, dtype=dtype).reshape(shape)

        return result


def pt_list_tensors(filename: str) -> list[str]:
    with open(filename, "rb") as f:
        magic = f.read(9)
        if magic != MAGIC:
            raise ValueError(
                f"Not a valid TensorN .pt file. "
                f"Expected magic {MAGIC!r}, got {magic!r}"
            )

        version = struct.unpack("<I", f.read(4))[0]
        if version == VERSION:
            return ["tensor"]
        if version != VERSION_MULTI:
            raise ValueError(f"Unsupported .pt version: {version}")

        tensor_count = struct.unpack("<Q", f.read(8))[0]

        names = []
        for _ in range(tensor_count):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            names.append(name)

            f.read(1)
            ndims = struct.unpack("<I", f.read(4))[0]
            f.read(ndims * 8)
            f.read(16)

        return names


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


def torch2pt_multi(input_pth: str, output_pt: str) -> None:
    try:
        import torch
    except ImportError:
        print("Error: torch is required for this command. Install with: pip install torch")
        sys.exit(1)

    obj = torch.load(input_pth, weights_only=True)
    tensors = {}

    if isinstance(obj, dict):
        for key, v in obj.items():
            if torch.is_tensor(v):
                tensors[str(key)] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                tensors[str(key)] = v
        if not tensors:
            raise ValueError("No tensors found in the loaded dict.")
    elif torch.is_tensor(obj):
        tensors["tensor"] = obj.detach().cpu().numpy()
    else:
        raise ValueError(f"Expected a dict or torch.Tensor, got {type(obj)}")

    save_tensorn_pt_multi(output_pt, tensors)
    print(f"Converted {input_pth} -> {output_pt} ({len(tensors)} tensors)")


def pt2torch_multi(input_pt: str, output_pth: str) -> None:
    try:
        import torch
    except ImportError:
        print("Error: torch is required for this command. Install with: pip install torch")
        sys.exit(1)

    tensors = load_tensorn_pt_multi(input_pt)
    torch_dict = {name: torch.from_numpy(arr) for name, arr in tensors.items()}
    torch.save(torch_dict, output_pth)
    print(f"Converted {input_pt} -> {output_pth} ({len(tensors)} tensors)")


def pt_list(input_pt: str) -> None:
    names = pt_list_tensors(input_pt)
    print(f"Tensors in {input_pt} ({len(names)}):")
    for name in names:
        print(f"  - {name}")


def print_usage():
    print(__doc__)


COMMANDS = {
    "np2pt":          (np2pt,          2, "<input.npy> <output.pt>"),
    "pt2np":          (pt2np,          2, "<input.pt> <output.npy>"),
    "torch2pt":       (torch2pt,       2, "<input.pth> <output.pt>"),
    "pt2torch":       (pt2torch,       2, "<input.pt> <output.pth>"),
    "torch2pt_multi": (torch2pt_multi, 2, "<input.pth> <output.pt>"),
    "pt2torch_multi": (pt2torch_multi, 2, "<input.pt> <output.pth>"),
    "pt_list":        (pt_list,        1, "<input.pt>"),
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
