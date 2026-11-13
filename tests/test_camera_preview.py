import struct
import unittest
import zlib

import numpy as np

from camera_preview import PNG_MAGIC, rgb_png


class CameraPreviewTests(unittest.TestCase):
    def test_rgb_png_is_valid_and_crosshair_visible(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[:] = (50, 70, 90)
        encoded = rgb_png(rgb, pixel=(2, 2))
        self.assertTrue(encoded.startswith(PNG_MAGIC))
        chunks = {}
        offset = len(PNG_MAGIC)
        while offset < len(encoded):
            size = struct.unpack_from(">I", encoded, offset)[0]
            tag = encoded[offset + 4:offset + 8]
            payload = encoded[offset + 8:offset + 8 + size]
            self.assertEqual(
                struct.unpack_from(">I", encoded, offset + 8 + size)[0],
                zlib.crc32(tag + payload) & 0xFFFFFFFF,
            )
            chunks[tag] = payload
            offset += 12 + size
        self.assertEqual(struct.unpack_from(">II", chunks[b"IHDR"]), (5, 4))
        scanlines = zlib.decompress(chunks[b"IDAT"])
        self.assertEqual(len(scanlines), 4 * (1 + 5 * 3))
        self.assertEqual(scanlines[2 * 16 + 1 + 2 * 3:2 * 16 + 1 + 3 * 3],
                         bytes([30, 250, 75]))

    def test_invalid_rgb_is_rejected(self):
        with self.assertRaises(ValueError):
            rgb_png(np.zeros((4, 5), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
