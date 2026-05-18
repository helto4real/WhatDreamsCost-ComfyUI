import json
import tempfile
import unittest
from pathlib import Path

from privacy import (
    CRYPTO_AVAILABLE,
    PrivacyError,
    crypto_status,
    decrypt_state,
    encrypt_state,
    is_encrypted_payload,
    key_path,
)


SAMPLE_STATE = {
    "global_prompt": "private global prompt",
    "timeline": {
        "segments": [
            {
                "id": "seg1",
                "start": 0,
                "length": 24,
                "type": "image",
                "prompt": "private segment prompt",
                "imageFile": "secret-file.png",
                "imageB64": "data:image/png;base64,SECRET_THUMBNAIL",
                "guideStrength": 0.75,
            }
        ],
        "audioSegments": [
            {
                "id": "aud1",
                "start": 0,
                "length": 24,
                "audioFile": "secret-audio.wav",
                "fileName": "secret-audio.wav",
            }
        ],
    },
}


class PrivacyTests(unittest.TestCase):
    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_key_generation_creates_stable_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = encrypt_state(SAMPLE_STATE, tmp)
            path = key_path(tmp)
            self.assertTrue(path.exists())
            key_data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(key_data["algorithm"], "AES-256-GCM")
            self.assertEqual(envelope["keyId"], key_data["keyId"])
            self.assertTrue(crypto_status(tmp)["keyExists"])

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_encrypt_decrypt_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = encrypt_state(SAMPLE_STATE, tmp)
            decrypted = decrypt_state(envelope, tmp)
            self.assertEqual(decrypted["global_prompt"], SAMPLE_STATE["global_prompt"])
            self.assertEqual(decrypted["timeline"]["segments"][0]["prompt"], "private segment prompt")

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_envelope_does_not_contain_private_plaintext(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = encrypt_state(SAMPLE_STATE, tmp)
            text = json.dumps(envelope)
            self.assertTrue(is_encrypted_payload(envelope))
            self.assertNotIn("private global prompt", text)
            self.assertNotIn("private segment prompt", text)
            self.assertNotIn("secret-file.png", text)
            self.assertNotIn("SECRET_THUMBNAIL", text)
            self.assertNotIn("secret-audio.wav", text)

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_wrong_key_fails_readably(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            envelope = encrypt_state(SAMPLE_STATE, one)
            encrypt_state(SAMPLE_STATE, two)
            with self.assertRaises(PrivacyError) as ctx:
                decrypt_state(envelope, two)
            self.assertIn("different local privacy key", str(ctx.exception))

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_missing_key_fails_readably(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = encrypt_state(SAMPLE_STATE, tmp)
            Path(key_path(tmp)).unlink()
            with self.assertRaises(PrivacyError) as ctx:
                decrypt_state(envelope, tmp)
            self.assertIn("missing", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
