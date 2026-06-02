import tempfile
import unittest

from ltx_director_privacy import derive_timeline_outputs, resolve_ltx_director_inputs
from privacy import CRYPTO_AVAILABLE, encrypt_state


class LTXDirectorPrivacyInputTests(unittest.TestCase):
    def test_derive_timeline_outputs_matches_commit_shape(self):
        timeline = {
            "segments": [
                {"id": "a", "start": 0, "length": 12, "type": "image", "prompt": "first @image1:character", "guideStrength": 0.8},
                {"id": "b", "start": 18, "length": 12, "type": "text", "prompt": "second"},
            ],
            "audioSegments": [],
            "referenceImages": [{"id": "ref1", "label": "image1", "kind": "character", "imageFile": "private.png"}],
        }
        derived = derive_timeline_outputs(timeline, 36)
        self.assertEqual(derived["local_prompts"], "first | second")
        self.assertEqual(derived["segment_lengths"], "18,18")
        self.assertEqual(derived["guide_strength"], "0.80")

    @unittest.skipUnless(CRYPTO_AVAILABLE, "cryptography package is required for privacy encryption tests")
    def test_encrypted_inputs_decrypt_and_derive_backend_values(self):
        timeline = {
            "segments": [
                {
                    "id": "source",
                    "start": 0,
                    "length": 9,
                    "type": "source_video",
                    "prompt": "",
                    "videoFile": "private.mp4",
                    "guideStrength": 0.85,
                },
                {"id": "prompt", "start": 9, "length": 15, "type": "text", "prompt": "private followup"},
            ],
            "audioSegments": [{"id": "audio", "start": 0, "length": 24, "audioFile": "private.wav"}],
            "referenceImages": [
                {"id": "ref1", "label": "image1", "kind": "character", "imageFile": "private-reference.png"}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            envelope = encrypt_state({"global_prompt": "private global", "timeline": timeline}, tmp)
            resolved = resolve_ltx_director_inputs(
                global_prompt="",
                timeline_data="{}",
                local_prompts="",
                segment_lengths="",
                guide_strength="",
                duration_frames=24,
                privacy_mode=True,
                privacy_payload=envelope,
                privacy_base_dir=tmp,
            )

        self.assertEqual(resolved["global_prompt"], "private global")
        self.assertEqual(resolved["local_prompts"], "private followup | private followup")
        self.assertEqual(resolved["segment_lengths"], "9,15")
        self.assertEqual(resolved["guide_strength"], "0.85")
        self.assertIn("private.mp4", resolved["timeline_data"])
        self.assertIn("private-reference.png", resolved["timeline_data"])


if __name__ == "__main__":
    unittest.main()
