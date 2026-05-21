"""HTTP routes for the LTX Director prompt optimizer."""

from __future__ import annotations

try:
    from aiohttp import web
    import server

    from .ltx_prompt_optimizer import (
        PromptOptimizerError,
        clear_hf_token,
        get_model_statuses,
        get_optimizer_job_status,
        get_optimizer_settings_status,
        optimize_segments,
        save_hf_token,
        start_optimizer_job,
    )
except Exception:  # noqa: BLE001 - route registration is best-effort inside ComfyUI.
    web = None
    server = None


if web is not None and server is not None:
    ROUTE_PREFIX = "/wdc_ltx_prompt_optimizer"

    @server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/models")
    async def get_prompt_optimizer_models(_request):
        try:
            return web.json_response(get_model_statuses())
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/settings")
    async def get_prompt_optimizer_settings(_request):
        try:
            return web.json_response(get_optimizer_settings_status())
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/settings")
    async def post_prompt_optimizer_settings(request):
        try:
            payload = await request.json()
            if payload.get("clear"):
                return web.json_response(clear_hf_token())
            return web.json_response(save_hf_token(payload.get("hf_token", "")))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/optimize")
    async def post_prompt_optimizer_optimize(request):
        try:
            payload = await request.json()
            return web.json_response(optimize_segments(payload))
        except PromptOptimizerError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/optimize/start")
    async def post_prompt_optimizer_start(request):
        try:
            payload = await request.json()
            return web.json_response({"ok": True, "job_id": start_optimizer_job(payload)})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/optimize/status")
    async def get_prompt_optimizer_status(request):
        try:
            job_id = request.query.get("job_id", "")
            return web.json_response(get_optimizer_job_status(job_id))
        except PromptOptimizerError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
