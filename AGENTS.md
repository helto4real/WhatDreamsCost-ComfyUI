# Agent Notes

- This repo is a ComfyUI custom-node pack for LTX/video workflows.
- Use `/home/thhel/git/ComfyUI` as the original ComfyUI source reference.
- For testing, use the ComfyUI repo Python environment: `/home/thhel/.pyenv/versions/3.13.2/bin/python`.
- `__init__.py` registers legacy `NODE_CLASS_MAPPINGS`, V3 `ComfyExtension` nodes, and route modules imported for side effects.
- Frontend extensions live in `js/`; `js/ltx_director.js` is the main LTX Director UI.
- Backend routes are in `timeline_image_routes.py`, `timeline_audio_routes.py`, `ltx_director_privacy_routes.py`, and `ltx_prompt_optimizer_routes.py`.
- `config/` and `thumbnail_cache/` are generated and ignored. Do not commit privacy keys, Hugging Face tokens, timing files, or thumbnail cache files.
- `vendor/tenstrip_10s/` contains adapted third-party code. Keep `THIRD_PARTY_NOTICES.md` current when changing vendored code.
- Prefer `python -m unittest discover -s tests` for tests. Compile touched Python with `python -m py_compile ...`.
- When adding or renaming nodes, update `__init__.py` mappings/extension list and any relevant JS.
