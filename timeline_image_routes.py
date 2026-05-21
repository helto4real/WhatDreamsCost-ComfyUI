import os
import secrets
import urllib.parse

from aiohttp import web
import server

from .timeline_image_config import (
    IMAGE_EXTENSIONS,
    add_folder,
    folder_by_alias,
    load_folders,
    remove_folder,
    resolve_image_path,
)
from .timeline_image_io import clear_thumbnail_cache, list_images, make_thumbnail


ROUTE_PREFIX = "/wdc_timeline_images"


def query_bool(value):
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def folder_payload():
    folders = []
    for folder in load_folders():
        exists = os.path.isdir(folder.path)
        folders.append(
            {
                "alias": folder.alias,
                "enabled": folder.enabled,
                "exists": exists,
                "image_count": len(list_images(folder.path)) if exists else 0,
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


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/images")
async def get_images(request):
    try:
        alias = request.query.get("alias", "")
        recursive = request.query.get("recursive", "1").lower() not in {"0", "false", "no"}
        privacy_mode = query_bool(request.query.get("privacy", ""))
        cache_bust = request.query.get("cacheBust", "")
        folder = folder_by_alias(alias)
        if not os.path.isdir(folder.path):
            return web.json_response({"images": [], "warning": "Folder does not exist."})

        images = list_images(folder.path, recursive=recursive)
        for image in images:
            thumb_params = {"alias": alias, "filename": image["filename"], "t": int(image["mtime"])}
            if privacy_mode:
                thumb_params["privacy"] = "1"
            if cache_bust:
                thumb_params["cacheBust"] = cache_bust
            image["thumb_url"] = (
                f"{ROUTE_PREFIX}/thumb?"
                + urllib.parse.urlencode(thumb_params)
            )
            image["image_url"] = (
                f"{ROUTE_PREFIX}/image?"
                + urllib.parse.urlencode({"alias": alias, "filename": image["filename"], "t": int(image["mtime"])})
            )
        return web.json_response({"images": images})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/refresh")
async def refresh(request):
    return web.json_response({"status": "ok", "folders": folder_payload()})


@server.PromptServer.instance.routes.post(f"{ROUTE_PREFIX}/thumb-cache/clear")
async def clear_thumb_cache(_request):
    try:
        clear_thumbnail_cache()
        return web.json_response({"status": "ok", "cacheBust": secrets.token_urlsafe(12)})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/thumb")
async def get_thumb(request):
    try:
        alias = request.query.get("alias", "")
        filename = urllib.parse.unquote(request.query.get("filename", ""))
        privacy_mode = query_bool(request.query.get("privacy", ""))
        path = resolve_image_path(alias, filename)
        thumb = make_thumbnail(path, privacy_mode=privacy_mode)
        if privacy_mode:
            return web.Response(
                body=thumb,
                headers={"Cache-Control": "private, no-store", "Content-Type": "image/webp"},
            )
        return web.FileResponse(thumb, headers={"Cache-Control": "public, max-age=86400", "Content-Type": "image/webp"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@server.PromptServer.instance.routes.get(f"{ROUTE_PREFIX}/image")
async def get_image(request):
    try:
        alias = request.query.get("alias", "")
        filename = urllib.parse.unquote(request.query.get("filename", ""))
        path = resolve_image_path(alias, filename)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return web.Response(status=400, text="Unsupported image type")
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=300"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
