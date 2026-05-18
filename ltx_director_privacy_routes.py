"""HTTP routes for LTX Director workflow privacy."""

from __future__ import annotations

try:
    from aiohttp import web
    import server

    from .privacy import PrivacyError, crypto_status, decrypt_state, encrypt_state
except Exception:  # noqa: BLE001 - route registration is best-effort inside ComfyUI.
    web = None
    server = None


if web is not None and server is not None:
    ROUTE_PREFIX = "/wdc_ltx_director/privacy"

    @server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/status")
    async def get_privacy_status(_request):
        return web.json_response({"ok": True, "status": crypto_status()})

    @server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/encrypt")
    async def post_privacy_encrypt(request):
        try:
            payload = await request.json()
            envelope = encrypt_state(payload.get("state", {}))
            return web.json_response({"ok": True, "envelope": envelope, "status": crypto_status()})
        except Exception as exc:  # noqa: BLE001 - API should return readable UI errors.
            return web.json_response({"ok": False, "error": str(exc), "status": crypto_status()}, status=400)

    @server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/decrypt")
    async def post_privacy_decrypt(request):
        try:
            payload = await request.json()
            state = decrypt_state(payload.get("payload", {}))
            return web.json_response({"ok": True, "state": state, "status": crypto_status()})
        except PrivacyError as exc:
            return web.json_response({"ok": False, "error": str(exc), "status": crypto_status()}, status=400)
        except Exception as exc:  # noqa: BLE001 - API should return readable UI errors.
            return web.json_response({"ok": False, "error": str(exc), "status": crypto_status()}, status=400)
