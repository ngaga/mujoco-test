"""Tiny dependency-free PNG encoder for MuJoCo RGB camera previews."""
import struct
import zlib

import numpy as np

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(tag, payload):
    return (
        struct.pack(">I", len(payload))
        + tag + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def rgb_png(rgb, pixel=None):
    """Encode uint8 RGB with an optional green crosshair at detected pixel.

    Encoding is performed on the simulation thread, never in HTTP handlers.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB image")
    height, width, _ = rgb.shape
    preview = rgb.copy()
    if pixel is not None:
        u, v = (int(round(pixel[0])), int(round(pixel[1])))
        if 0 <= u < width and 0 <= v < height:
            preview[v, max(0, u - 12):min(width, u + 13)] = (30, 250, 75)
            preview[max(0, v - 12):min(height, v + 13), u] = (30, 250, 75)
    # PNG RGB rows: filtering type 0, one byte per row.
    rows = b"".join(b"\x00" + preview[y].tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows, level=1))
        + _chunk(b"IEND", b"")
    )
