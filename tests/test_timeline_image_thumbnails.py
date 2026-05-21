import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image

from privacy import CRYPTO_AVAILABLE, decrypt_bytes


folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_input_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", folder_paths_stub)

from timeline_image_io import (  # noqa: E402
    THUMBNAIL_CACHE_PURPOSE,
    clear_thumbnail_cache,
    make_thumbnail,
)


class TimelineImageThumbnailTests(unittest.TestCase):
    def create_image(self, path):
        image = Image.new("RGB", (64, 48), (12, 120, 220))
        image.save(path, "PNG")

    def test_plain_mode_writes_and_reuses_webp_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            cache = root / "cache"
            self.create_image(source)

            first = make_thumbnail(source, cache_dir=cache)
            second = make_thumbnail(source, cache_dir=cache)

            self.assertEqual(first, second)
            self.assertEqual(first.suffix, ".webp")
            self.assertTrue(first.exists())
            self.assertTrue(first.read_bytes().startswith(b"RIFF"))
            self.assertEqual(list(cache.glob("*.webp")), [first])

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for encrypted thumbnail tests")
    def test_privacy_mode_writes_only_encrypted_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            cache = root / "cache"
            keys = root / "keys"
            self.create_image(source)

            thumbnail = make_thumbnail(source, privacy_mode=True, privacy_base_dir=keys, cache_dir=cache)
            encrypted_files = list(cache.glob("*.webp.enc"))

            self.assertTrue(thumbnail.startswith(b"RIFF"))
            self.assertEqual(len(encrypted_files), 1)
            self.assertEqual(list(cache.glob("*.webp")), [])
            encrypted_text = encrypted_files[0].read_text(encoding="utf-8")
            self.assertNotIn("RIFF", encrypted_text)
            self.assertNotIn("WEBP", encrypted_text)
            self.assertNotIn("source.png", encrypted_text)

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for encrypted thumbnail tests")
    def test_privacy_mode_cached_thumbnail_decrypts_to_webp_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            cache = root / "cache"
            keys = root / "keys"
            self.create_image(source)

            first = make_thumbnail(source, privacy_mode=True, privacy_base_dir=keys, cache_dir=cache)
            encrypted_file = next(cache.glob("*.webp.enc"))
            payload = json.loads(encrypted_file.read_text(encoding="utf-8"))
            decrypted = decrypt_bytes(payload, THUMBNAIL_CACHE_PURPOSE, keys)
            second = make_thumbnail(source, privacy_mode=True, privacy_base_dir=keys, cache_dir=cache)

            self.assertEqual(decrypted, first)
            self.assertEqual(second, first)
            self.assertTrue(decrypted.startswith(b"RIFF"))

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for encrypted thumbnail tests")
    def test_clear_thumbnail_cache_removes_plain_and_encrypted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            cache = root / "cache"
            keys = root / "keys"
            self.create_image(source)

            plain = make_thumbnail(source, cache_dir=cache)
            private = make_thumbnail(source, privacy_mode=True, privacy_base_dir=keys, cache_dir=cache)
            self.assertTrue(plain.exists())
            self.assertTrue(private.startswith(b"RIFF"))
            self.assertTrue(any(cache.iterdir()))

            clear_thumbnail_cache(cache)

            self.assertTrue(cache.exists())
            self.assertEqual(list(cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
