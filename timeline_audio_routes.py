import os
import urllib.parse

from aiohttp import web
import server

from .timeline_audio_config import (
    AUDIO_EXTENSIONS,
    add_folder,
    folder_by_alias,
    load_folders,
    remove_folder,
    resolve_audio_path,
)
from .timeline_audio_io import list_audio


ROUTE_PREFIX = "/wdc_timeline_audio"


def folder_payload():
    folders = []
    for folder in load_folders():
        exists = os.path.isdir(folder.path)
        folders.append(
            {
                "alias": folder.alias,
                "enabled": folder.enabled,
                "exists": exists,
                "audio_count": len(list_audio(folder.path)) if exists else 0,
            }
        )
    return folders


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/folders")
async def get_folders(request):
    return web.json_response({"folders": folder_payload()})


@server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/folders")
async def post_folder(request):
    try:
        data = await request.json()
        add_folder(data.get("alias"), data.get("path"))
        return web.json_response({"status": "ok", "folders": folder_payload()})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.delete(f"{ROUTE_PREFIX}/folders")
async def delete_folder(request):
    try:
        alias = request.query.get("alias", "")
        remove_folder(alias)
        return web.json_response({"status": "ok", "folders": folder_payload()})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/audios")
async def get_audios(request):
    try:
        alias = request.query.get("alias", "")
        recursive = request.query.get("recursive", "1").lower() not in {"0", "false", "no"}
        folder = folder_by_alias(alias)
        if not os.path.isdir(folder.path):
            return web.json_response({"audios": [], "warning": "Folder does not exist."})

        audios = list_audio(folder.path, recursive=recursive)
        for audio in audios:
            audio["audio_url"] = (
                f"{ROUTE_PREFIX}/audio?"
                + urllib.parse.urlencode({"alias": alias, "filename": audio["filename"], "t": int(audio["mtime"])})
            )
        return web.json_response({"audios": audios})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/refresh")
async def refresh(request):
    return web.json_response({"status": "ok", "folders": folder_payload()})


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/audio")
async def get_audio(request):
    try:
        alias = request.query.get("alias", "")
        filename = urllib.parse.unquote(request.query.get("filename", ""))
        path = resolve_audio_path(alias, filename)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return web.Response(status=400, text="Unsupported audio type")
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=300"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
