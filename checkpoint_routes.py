"""Small browser API for checkpoint inventory and immutable branches."""

from __future__ import annotations

import asyncio
import gc

from .checkpoint_manifest import (
    branch_checkpoint_project,
    list_checkpoint_projects,
    public_checkpoint_status,
)


API_BASE = "/h3_digital_human/checkpoints"


def _error(web, exc, status=400):
    return web.json_response({"error": str(exc)}, status=status)


def _release_generation_resources():
    try:
        from comfy import model_management

        model_management.unload_all_models()
        try:
            model_management.soft_empty_cache(force=True)
        except TypeError:
            model_management.soft_empty_cache()
    finally:
        gc.collect()
    return {"released": True}


def register_checkpoint_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except (ImportError, ModuleNotFoundError):
        return False

    routes = PromptServer.instance.routes
    if getattr(routes, "_h3_digital_human_checkpoint_routes", False):
        return True

    async def status(request):
        try:
            prefix = request.query.get("prefix", "")
            value = await asyncio.to_thread(public_checkpoint_status, prefix)
            return web.json_response(value)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return _error(web, exc)

    async def branch(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("branch request must be a JSON object")
            value = await asyncio.to_thread(
                branch_checkpoint_project,
                body.get("prefix", ""),
                body.get("new_prefix", ""),
                body.get("through_clip", 0),
                body.get("take_sha256", ""),
            )
            return web.json_response(value)
        except FileExistsError as exc:
            return _error(web, exc, status=409)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return _error(web, exc)

    async def projects(_request):
        try:
            value = await asyncio.to_thread(list_checkpoint_projects)
            return web.json_response(value)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return _error(web, exc)

    async def cleanup(_request):
        try:
            # Run model-management calls on ComfyUI's server thread after the
            # executor has emitted terminal success. CUDA/model patchers are
            # not assumed safe from an arbitrary worker created by to_thread.
            value = _release_generation_resources()
            return web.json_response(value)
        except Exception as exc:  # Comfy backend/version specific
            return _error(web, exc, status=500)

    routes.get(API_BASE)(status)
    routes.get(API_BASE + "/projects")(projects)
    routes.post(API_BASE + "/branch")(branch)
    routes.post(API_BASE + "/cleanup")(cleanup)
    setattr(routes, "_h3_digital_human_checkpoint_routes", True)
    return True
