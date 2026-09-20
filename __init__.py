"""MiniMax H3 long-video digital-human generation and recovery nodes.

ComfyUI provides ``folder_paths`` before loading custom nodes.  Keeping the
package import harmless when that module is absent lets the repository's
standalone CPU/static tests run without pretending a full ComfyUI installation
is available.
"""

try:
    import folder_paths as _folder_paths  # noqa: F401
except ModuleNotFoundError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .checkpoint_routes import register_checkpoint_routes

    register_checkpoint_routes()

WEB_DIRECTORY = "./js"
__version__ = "1.0.0"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
