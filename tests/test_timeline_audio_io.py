import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


timeline_audio_config_stub = types.ModuleType("timeline_audio_config")
timeline_audio_config_stub.AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
sys.modules.setdefault("timeline_audio_config", timeline_audio_config_stub)

import timeline_audio_io
from timeline_audio_io import audio_duration_seconds, list_audio


class TimelineAudioIoTests(unittest.TestCase):
    def test_list_audio_ignores_non_audio_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clip.wav").write_bytes(b"audio")
            (root / "notes.txt").write_text("not audio", encoding="utf-8")

            audios = list_audio(root)

            self.assertEqual([item["filename"] for item in audios], ["clip.wav"])
            self.assertNotIn("duration_seconds", audios[0])

    def test_list_audio_includes_duration_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clip.wav").write_bytes(b"audio")

            with mock.patch("timeline_audio_io.audio_duration_seconds", return_value=1.25):
                audios = list_audio(root, include_duration=True)

            self.assertEqual(audios[0]["filename"], "clip.wav")
            self.assertEqual(audios[0]["duration_seconds"], 1.25)

    def test_list_audio_duration_probe_failures_do_not_break_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clip.wav").write_bytes(b"audio")

            with mock.patch("timeline_audio_io.audio_duration_seconds", return_value=None):
                audios = list_audio(root, include_duration=True)

            self.assertEqual(audios[0]["filename"], "clip.wav")
            self.assertIsNone(audios[0]["duration_seconds"])

    def test_audio_duration_converts_container_microseconds_to_seconds(self):
        class FakeContainer:
            duration = 90_000_000
            streams = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch.object(timeline_audio_io.av, "time_base", 1_000_000):
            with mock.patch.object(timeline_audio_io.av, "open", return_value=FakeContainer()):
                self.assertEqual(audio_duration_seconds("clip.mp3"), 90.0)


if __name__ == "__main__":
    unittest.main()
