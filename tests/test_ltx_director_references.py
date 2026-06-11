import unittest

from ltx_director_references import (
    build_reference_guide_specs,
    build_segment_reference_usage,
    normalize_reference_images,
    parse_reference_tags,
    reference_usage_errors,
    replace_reference_tags,
    replace_reference_tags_in_prompt_list,
    strip_reference_tags,
    strip_reference_tags_from_prompt_list,
)


class LTXDirectorReferenceTests(unittest.TestCase):
    def test_parse_reference_tags_marks_supported_and_unsupported_kinds(self):
        tags = parse_reference_tags("walks with @image1:character[0.65] and @image2:style")

        self.assertEqual(
            tags,
            [
                {
                    "label": "image1",
                    "kind": "character",
                    "token": "@image1:character[0.65]",
                    "supported": True,
                    "strength_override": 0.65,
                },
                {
                    "label": "image2",
                    "kind": "style",
                    "token": "@image2:style",
                    "supported": False,
                    "strength_override": None,
                },
            ],
        )

    def test_strip_reference_tags_removes_control_syntax(self):
        self.assertEqual(
            strip_reference_tags("Alice @image1:character[0.5] walks, then @image2:style waves."),
            "Alice walks, then waves.",
        )

    def test_strip_reference_tags_from_prompt_list_keeps_segment_separator(self):
        self.assertEqual(
            strip_reference_tags_from_prompt_list("A @image1:character | B @image2:character"),
            "A | B",
        )

    def test_normalize_reference_images_assigns_labels_and_character_kind(self):
        refs = normalize_reference_images(
            [
                {"id": "a", "imageFile": "a.png"},
                {"id": "b", "label": "IMAGE7", "kind": "character", "strength": "0.5", "description": "Tall man"},
                {"id": "style", "label": "image9", "kind": "style"},
            ]
        )

        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]["label"], "image1")
        self.assertEqual(refs[0]["kind"], "character")
        self.assertEqual(refs[1]["label"], "image7")
        self.assertEqual(refs[1]["strength"], 0.5)
        self.assertEqual(refs[1]["description"], "Tall man")

    def test_replace_reference_tags_uses_descriptions(self):
        refs = [
            {"label": "image1", "kind": "character", "description": "a blonde woman in an olive jacket"},
        ]

        self.assertEqual(
            replace_reference_tags("The subject turns. @image1:character[0.35]", refs),
            "The subject turns. a blonde woman in an olive jacket",
        )

    def test_replace_reference_tags_strips_empty_descriptions(self):
        refs = [{"label": "image1", "kind": "character", "description": ""}]

        self.assertEqual(
            replace_reference_tags("The subject turns. @image1:character", refs),
            "The subject turns.",
        )

    def test_replace_reference_tags_handles_multiple_and_repeated_references(self):
        refs = [
            {"label": "image1", "kind": "character", "description": "a blonde woman"},
            {"label": "image2", "kind": "character", "description": "a dark-haired man"},
        ]

        self.assertEqual(
            replace_reference_tags_in_prompt_list(
                "@image1:character greets @image2:character | @image1:character smiles",
                refs,
            ),
            "a blonde woman greets a dark-haired man | a blonde woman smiles",
        )

    def test_build_segment_reference_usage_matches_enabled_known_references(self):
        timeline = {
            "referenceImages": [
                {"id": "a", "label": "image1", "kind": "character", "imageFile": "a.png", "strength": 0.8},
                {"id": "b", "label": "image2", "kind": "character", "enabled": False},
            ],
            "segments": [
                {
                    "id": "seg",
                    "start": 0,
                    "length": 24,
                    "prompt": "A @image1:character[0.45] meets @image2:character and @image3:character",
                }
            ],
        }

        usage = build_segment_reference_usage(timeline, 24)

        self.assertEqual(usage[0]["clean_prompt"], "A meets and")
        self.assertEqual([ref["label"] for ref in usage[0]["references"]], ["image1"])
        self.assertEqual(usage[0]["references"][0]["strength"], 0.45)
        self.assertEqual(usage[0]["references"][0]["strength_override"], 0.45)
        self.assertEqual([tag["token"] for tag in usage[0]["unknown_tags"]], ["@image2:character", "@image3:character"])

    def test_build_segment_reference_usage_uses_default_strength_without_override(self):
        timeline = {
            "referenceImages": [
                {"id": "a", "label": "image1", "kind": "character", "imageFile": "a.png", "strength": 0.8},
            ],
            "segments": [
                {
                    "id": "seg",
                    "start": 0,
                    "length": 24,
                    "prompt": "A @image1:character enters",
                }
            ],
        }

        usage = build_segment_reference_usage(timeline, 24)

        self.assertEqual(usage[0]["references"][0]["strength"], 0.8)
        self.assertIsNone(usage[0]["references"][0]["strength_override"])

    def test_reference_usage_errors_separates_missing_character_refs_from_future_kinds(self):
        timeline = {
            "referenceImages": [],
            "segments": [
                {
                    "id": "seg",
                    "start": 0,
                    "length": 24,
                    "prompt": "A @image1:character[0.25] with @image2:style",
                }
            ],
        }

        errors = reference_usage_errors(build_segment_reference_usage(timeline, 24))

        self.assertEqual(errors["unknown"], ["@image1:character[0.25]"])
        self.assertEqual(errors["unsupported"], ["@image2:style"])

    def test_build_reference_guide_specs_flattens_segment_scoped_references(self):
        timeline = {
            "referenceImages": [
                {"id": "ref-a", "label": "image1", "kind": "character", "imageFile": "a.png", "strength": 0.7},
            ],
            "segments": [
                {"id": "timeline-image", "start": 0, "length": 8, "type": "image", "imageFile": "scene.png", "prompt": "scene"},
                {"id": "uses-ref", "start": 8, "length": 8, "type": "text", "prompt": "runs @image1:character[0.35]"},
            ],
        }

        specs = build_reference_guide_specs(timeline, 16)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["id"], "ref-a")
        self.assertEqual(specs[0]["label"], "image1")
        self.assertEqual(specs[0]["segment_id"], "uses-ref")
        self.assertEqual(specs[0]["insert_frame"], 8)
        self.assertEqual(specs[0]["strength"], 0.35)
        self.assertEqual(specs[0]["strength_override"], 0.35)


if __name__ == "__main__":
    unittest.main()
