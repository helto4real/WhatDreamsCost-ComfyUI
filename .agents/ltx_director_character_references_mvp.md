# LTX Director Character Reference MVP

## Status

Last updated: 2026-06-02

| Step | Name | Planned | Implemented | User accepted |
| --- | --- | --- | --- | --- |
| 1 | Persist and track this plan | yes | yes | yes |
| 2 | Data model and prompt parsing | yes | yes | yes |
| 3 | Backend reference guide data | yes | yes | yes |
| 4 | Guide node IC-LoRA support | yes | yes | yes |
| 5 | Optional identity anchor integration | yes | yes | yes |
| 6 | Frontend reference image UI | yes | yes | yes |
| 7 | Docs and examples | yes | yes | pending |

## Acceptance Protocol

- Implement one step at a time.
- After each implemented step, update this file.
- Do not start the next step until the user accepts the current step.
- Keep status values explicit: `yes`, `no`, or `pending`.

## Summary

Add separate Director-managed character reference images that are not timeline segments. Segment prompts can reference them with tags like `@image1:character`; tags are stripped before LTX text encoding, and references apply only to tagged segments.

Use both supported likeness paths where possible:
- Standard/official LTX guide conditioning with IC-LoRA parameter support from ComfyUI's `LTXVAddGuide`.
- Existing bundled 10S identity-anchor helpers as an optional additional model patch path.

References checked:
- Local ComfyUI `/home/thhel/git/ComfyUI/comfy_extras/nodes_lt.py`
- Local ComfyUI `/home/thhel/git/ComfyUI/comfy/ldm/lightricks/model.py`
- ComfyUI LTX-2.3 docs: https://docs.comfy.org/tutorials/video/ltx/ltx-2-3
- Lightricks ComfyUI-LTXVideo: https://github.com/Lightricks/ComfyUI-LTXVideo
- ID-LoRA LTX-2.3: https://github.com/ID-LoRA/ID-LoRA

## Implementation Steps

### 1. Persist And Track This Plan

- Create `.agents/ltx_director_character_references_mvp.md`.
- Include the full step list plus a status table with `planned`, `implemented`, and `user_accepted`.
- After each implementation step, update the file and wait for user acceptance before starting the next step.

### 2. Data Model And Prompt Parsing

- Extend Director timeline JSON with `referenceImages: []`.
- Each reference stores stable id, display label like `image1`, kind fixed to `character`, folder alias/file metadata or input upload data, strength, and enabled flag.
- Add backend helpers to parse `@imageN:character`, strip those tags before `_encode_relay`, and build per-segment reference usage.

### 3. Backend Reference Guide Data

- Extend `guide_data` to carry separate `reference_images` entries while preserving existing `images`, `insert_frames`, and `strengths`.
- Load/resize/compress reference images using the same safe image-loading path as timeline images.
- For each tagged segment, attach the referenced character image as an LTX guide at that segment start with configurable strength.
- Do not change prompt optimizer/enhancer behavior for MVP.

### 4. Guide Node IC-LoRA Support

- Update `LTXDirectorGuide` to accept optional `iclora_parameters` and pass them through the same way ComfyUI's `LTXVAddGuide` does.
- Preserve existing workflow compatibility when the input is unconnected.
- Document recommended wiring: `LoRA Loader.model -> Get IC-LoRA Parameters -> LTX Director Guide.iclora_parameters`.

### 5. Optional Identity Anchor Integration

- Add a small node/helper that can select a Director reference image as `reference_image` for `LTXIdentityAnchorLatentAware`.
- Keep this optional so users can choose standard guides, 10S anchors, or both.
- Do not auto-patch the model inside `LTXDirector`; keep model-patch wiring explicit.

### 6. Frontend Reference Image UI

- Add a separate "References" panel/button in `js/ltx_director.js`, reusing the existing image browser.
- Let users add/remove/rename character references without placing them on the timeline.
- Show labels like `@image1:character` with a copy/insert affordance for the selected segment prompt.
- Store reference metadata in `timeline_data` and include it in privacy encryption state.
- In privacy mode, reference thumbnails must use the existing encrypted `privacy=1` thumbnail route, `Cache-Control: private, no-store`, and thumbnail cache clearing on privacy-mode changes.

### 7. Docs And Examples

- Update README with the syntax, MVP limitations, and recommended IC-LoRA/identity-anchor wiring.
- Note that mood/style references are intentionally out of scope for MVP.
- Add or update a compact example workflow only if the node interface changes require it.

## Test Plan

- Unit test tag parsing and stripping: multiple references, repeated tags, unknown tags, no tags.
- Unit test privacy derivation so `referenceImages` round-trips through encrypted payloads.
- Unit test guide-data construction with timeline images plus separate character references.
- Compile touched Python with `/home/thhel/.pyenv/versions/3.13.2/bin/python -m py_compile ...`.
- Run `/home/thhel/.pyenv/versions/3.13.2/bin/python -m unittest discover -s tests`.
- Browser/manual check: add references, copy/insert tag, save/reload workflow, enable privacy mode, decrypt, and verify references remain separate from timeline blocks.

## Assumptions And Defaults

- MVP supports only `:character`; `:style`, `:mood`, etc. are rejected or ignored with a clear warning.
- References are segment-scoped: only prompts containing `@imageN:character` receive that image guide.
- Tags are control syntax and are stripped before text encoding.
- No model downloads are bundled; IC-LoRA support is exposed through optional wiring.
- User acceptance is required after each implementation step before moving to the next.
