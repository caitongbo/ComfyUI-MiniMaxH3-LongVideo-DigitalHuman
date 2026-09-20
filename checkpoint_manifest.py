"""Atomic indexed H3 checkpoint inventory and dependency lineage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import PurePosixPath


FORMAT = "h3_digital_human_checkpoint_manifest"
_LOCK = threading.RLock()


def _output_root():
    import folder_paths

    return os.path.realpath(folder_paths.get_output_directory())


def prefix_base(checkpoint_prefix):
    prefix = str(checkpoint_prefix).strip().replace("\\", "/")
    pure = PurePosixPath(prefix)
    if not prefix or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            "h3_checkpoint_manifest: prefix must be relative to ComfyUI output"
        )
    if prefix.endswith(".safetensors"):
        prefix = prefix[:-len(".safetensors")]
    root = _output_root()
    base = os.path.realpath(os.path.join(root, *PurePosixPath(prefix).parts))
    if os.path.commonpath((root, base)) != root:
        raise ValueError("h3_checkpoint_manifest: prefix escapes output")
    return base


def manifest_path(checkpoint_prefix):
    return prefix_base(checkpoint_prefix) + ".manifest.json"


def checkpoint_path(checkpoint_prefix, index):
    return "%s_%05d.safetensors" % (prefix_base(checkpoint_prefix), int(index))


def file_sha256(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, document):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = "%s.part.%s.%s" % (path, os.getpid(), uuid.uuid4().hex)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_copy(source, destination):
    """Publish an immutable copy without exposing a partial destination."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = "%s.part.%s.%s" % (
        destination, os.getpid(), uuid.uuid4().hex
    )
    try:
        with open(source, "rb") as incoming, open(temporary, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 << 20)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError("h3_checkpoint_manifest: unsupported manifest format")
    if not isinstance(document.get("clips"), dict):
        raise ValueError("h3_checkpoint_manifest: clips inventory is invalid")
    return document


def _dependency_signature(prompt, unique_id):
    """Hash only the graph ancestry feeding this Save Latent node."""
    if not isinstance(prompt, dict) or unique_id is None:
        return ""
    visiting = set()
    memo = {}

    storage_only_root_inputs = {
        "filename_prefix", "checkpoint_prefix", "clip_index",
        "reuse_checkpoint",
    }
    def walk(node_id, root=False):
        key = str(node_id)
        if key in memo:
            return memo[key]
        if key in visiting:
            return {"cycle": key}
        node = prompt.get(key)
        if not isinstance(node, dict):
            return {"missing": key}
        visiting.add(key)
        inputs = {}
        class_type = str(node.get("class_type") or "")
        for name, value in sorted((node.get("inputs") or {}).items()):
            # Storage identity is operational only for the root checkpoint
            # being validated; generation dependencies remain content-based.
            if root and str(name) in storage_only_root_inputs:
                continue
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and str(value[0]) in prompt
                    and isinstance(value[1], int)):
                inputs[str(name)] = {
                    "output": int(value[1]),
                    "node": walk(value[0]),
                }
            else:
                inputs[str(name)] = value
        result = {"class_type": class_type, "inputs": inputs}
        visiting.remove(key)
        memo[key] = result
        return result

    try:
        payload = json.dumps(
            walk(unique_id, root=True), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=repr,
        ).encode("utf-8")
    except Exception:
        return ""
    return hashlib.sha256(payload).hexdigest()


def archive_current_take(checkpoint_prefix, index, path):
    """Preserve the active indexed checkpoint before it is overwritten."""
    index = int(index)
    if index < 1 or not os.path.isfile(path):
        return None
    target = manifest_path(checkpoint_prefix)
    with _LOCK:
        document = _load(target)
        if document is None:
            return None
        record = document["clips"].get(str(index))
        if not isinstance(record, dict):
            return None
        digest = file_sha256(path)
        if digest != str(record.get("sha256") or ""):
            # The process may have died after atomically replacing the tensor
            # file but before publishing its new manifest record. That file is
            # deliberately not a committed take; let the next successful save
            # replace it and repair the slot instead of permanently wedging
            # recovery behind the older record.
            return None
        archive_dir = prefix_base(checkpoint_prefix) + ".takes"
        archive = os.path.join(
            archive_dir, "clip_%05d.%s.safetensors" % (index, digest[:16])
        )
        if not os.path.isfile(archive):
            os.makedirs(archive_dir, exist_ok=True)
            try:
                os.link(path, archive)
            except OSError:
                _atomic_copy(path, archive)
        archived = dict(record)
        archived.pop("stale_after_clip", None)
        archived["path"] = os.path.relpath(
            archive, _output_root()
        ).replace(os.sep, "/")
        archived["archived_ns"] = time.time_ns()
        takes = document.setdefault("takes", {}).setdefault(str(index), [])
        if not any(item.get("sha256") == digest for item in takes
                   if isinstance(item, dict)):
            takes.append(archived)
            document["format"] = FORMAT
            document["updated_ns"] = time.time_ns()
            _atomic_json(target, document)
    return archive


def _record_content_usable(
    checkpoint_prefix, index, document, verify_hash=True,
):
    """Validate one manifest record, its file and predecessor lineage."""
    index = int(index)
    try:
        if document is None:
            return False, "manifest missing"
        record = document["clips"].get(str(index))
        if not isinstance(record, dict):
            return False, "record missing"
        if record.get("stale_after_clip") is not None:
            return False, "stale because Clip %s was replaced" % (
                record.get("stale_after_clip"),
            )
        path = checkpoint_path(checkpoint_prefix, index)
        if not os.path.isfile(path):
            return False, "checkpoint missing"
        stat = os.stat(path)
        if int(record.get("size", -1)) != int(stat.st_size):
            return False, "size changed"
        if (not verify_hash and int(record.get("mtime_ns") or 0) > 0
                and int(record.get("mtime_ns")) != int(stat.st_mtime_ns)):
            return False, "modification time changed"
        if verify_hash and file_sha256(path) != str(record.get("sha256") or ""):
            return False, "checkpoint content changed"
        if index > 1:
            predecessor = document["clips"].get(str(index - 1))
            if (not isinstance(predecessor, dict)
                    or predecessor.get("stale_after_clip") is not None
                    or str(record.get("predecessor_sha256") or "") !=
                    str(predecessor.get("sha256") or "")):
                return False, "predecessor lineage changed"
        return True, "ready"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)


def checkpoint_record_usable(
    checkpoint_prefix, index, prompt=None, unique_id=None, verify_hash=True,
):
    """Return ``(usable, reason)`` for one lazy resume/load-only gate."""
    index = int(index)
    try:
        with _LOCK:
            document = _load(manifest_path(checkpoint_prefix))
        usable, reason = _record_content_usable(
            checkpoint_prefix, index, document, verify_hash=verify_hash
        )
        if not usable:
            return usable, reason
        record = document["clips"][str(index)]
        expected_signature = _dependency_signature(prompt, unique_id)
        recorded_signature = str(record.get("generation_signature") or "")
        if (not bool(record.get("frozen_branch")) and expected_signature
                and recorded_signature != expected_signature):
            return False, "upstream workflow changed"
        return True, "ready"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)


def _available_memory_bytes():
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def public_checkpoint_status(checkpoint_prefix):
    """Browser-safe checkpoint/take inventory with no absolute paths."""
    target = manifest_path(checkpoint_prefix)
    with _LOCK:
        document = _load(target)
    if document is None:
        free = shutil.disk_usage(_output_root()).free
        return {
            "format": FORMAT,
            "checkpoint_prefix": str(checkpoint_prefix),
            "exists": False,
            "clips": [],
            "storage": {
                "active_bytes": 0,
                "take_bytes": 0,
                "total_bytes": 0,
                "free_bytes": int(free),
                "available_memory_bytes": _available_memory_bytes(),
            },
        }
    takes_by_clip = document.get("takes") or {}
    clips = []
    active_bytes = 0
    take_bytes = 0
    for key, record in sorted(
        document["clips"].items(), key=lambda item: int(item[0])
    ):
        if not isinstance(record, dict):
            continue
        index = int(key)
        path = checkpoint_path(checkpoint_prefix, index)
        takes = []
        for take in takes_by_clip.get(key, []):
            if not isinstance(take, dict):
                continue
            takes.append({
                "sha256": str(take.get("sha256") or ""),
                "published_ns": int(take.get("published_ns") or 0),
                "archived_ns": int(take.get("archived_ns") or 0),
                "video_shape": take.get("video_shape") or [],
                "size": int(take.get("size") or 0),
            })
            take_bytes += int(take.get("size") or 0)
        active_bytes += int(record.get("size") or 0)
        ready, reason = _record_content_usable(
            checkpoint_prefix, index, document,
            verify_hash=False,
        )
        clips.append({
            "index": index,
            "ready": bool(ready),
            "reason": str(reason),
            "stale_after_clip": record.get("stale_after_clip"),
            "frozen_branch": bool(record.get("frozen_branch")),
            "sha256": str(record.get("sha256") or ""),
            "published_ns": int(record.get("published_ns") or 0),
            "video_shape": record.get("video_shape") or [],
            "takes": takes,
        })
    return {
        "format": FORMAT,
        "checkpoint_prefix": str(checkpoint_prefix),
        "exists": True,
        "updated_ns": int(document.get("updated_ns") or 0),
        "clips": clips,
        "storage": {
            "active_bytes": active_bytes,
            "take_bytes": take_bytes,
            "total_bytes": active_bytes + take_bytes,
            "free_bytes": int(shutil.disk_usage(_output_root()).free),
            "available_memory_bytes": _available_memory_bytes(),
        },
    }


def list_checkpoint_projects(limit=500):
    """Discover only valid H3 manifests under output for the project picker."""
    root = _output_root()
    projects = []
    suffix = ".manifest.json"
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [
            name for name in folders
            if not name.startswith(".")
            and not os.path.islink(os.path.join(directory, name))
        ]
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(directory, name)
            try:
                document = _load(path)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if document is None:
                continue
            relative = os.path.relpath(path[:-len(suffix)], root).replace(os.sep, "/")
            clips = document.get("clips") or {}
            ready = sum(
                _record_content_usable(
                    relative, int(index), document,
                    verify_hash=False,
                )[0]
                for index, record in clips.items()
                if isinstance(record, dict)
            )
            projects.append({
                "prefix": relative,
                "clips": len(clips),
                "ready": ready,
                "updated_ns": int(document.get("updated_ns") or 0),
                "branched_from": document.get("branched_from"),
            })
            if len(projects) >= int(limit):
                break
        if len(projects) >= int(limit):
            break
    projects.sort(key=lambda item: (-item["updated_ns"], item["prefix"]))
    return {"projects": projects}


def branch_checkpoint_project(
    checkpoint_prefix, new_prefix, through_clip, take_sha256="",
):
    """Create an independent hard-linked branch through one selected take."""
    through_clip = int(through_clip)
    if through_clip < 1:
        raise ValueError("h3_checkpoint_manifest: branch needs at least one clip")
    source_manifest = manifest_path(checkpoint_prefix)
    destination_manifest = manifest_path(new_prefix)
    with _LOCK:
        source = _load(source_manifest)
        if source is None:
            raise FileNotFoundError("h3_checkpoint_manifest: source manifest missing")
        if os.path.exists(destination_manifest):
            raise FileExistsError(
                "h3_checkpoint_manifest: destination project already exists"
            )
        selected = []
        for index in range(1, through_clip + 1):
            record = source["clips"].get(str(index))
            if not isinstance(record, dict):
                raise ValueError(
                    "h3_checkpoint_manifest: source Clip %d is missing" % index
                )
            if index == through_clip and take_sha256:
                candidates = source.get("takes", {}).get(str(index), [])
                record = next((item for item in candidates
                               if isinstance(item, dict)
                               and item.get("sha256") == take_sha256), None)
                if record is None:
                    raise ValueError(
                        "h3_checkpoint_manifest: selected take was not found"
                    )
            selected.append(dict(record))
        previous = ""
        for index, record in enumerate(selected, start=1):
            if index > 1 and str(record.get("predecessor_sha256") or "") != previous:
                raise ValueError(
                    "h3_checkpoint_manifest: selected take does not continue "
                    "from the branch's preceding Clip %d" % (index - 1)
                )
            previous = str(record.get("sha256") or "")

        destination = {
            "format": FORMAT,
            "checkpoint_prefix": str(new_prefix).replace("\\", "/"),
            "clips": {},
            "takes": {},
            "branched_from": {
                "checkpoint_prefix": str(checkpoint_prefix).replace("\\", "/"),
                "through_clip": through_clip,
                "take_sha256": str(take_sha256 or ""),
            },
            "updated_ns": time.time_ns(),
        }
        for index, record in enumerate(selected, start=1):
            output_root = _output_root()
            source_path = os.path.realpath(os.path.join(
                output_root, *PurePosixPath(record["path"]).parts
            ))
            if os.path.commonpath((output_root, source_path)) != output_root:
                raise ValueError(
                    "h3_checkpoint_manifest: source take escapes output"
                )
            if (not os.path.isfile(source_path)
                    or file_sha256(source_path) != record.get("sha256")):
                raise ValueError(
                    "h3_checkpoint_manifest: source Clip %d failed integrity check"
                    % index
                )
            target_path = checkpoint_path(new_prefix, index)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                os.link(source_path, target_path)
            except OSError:
                _atomic_copy(source_path, target_path)
            copied = dict(record)
            copied.pop("stale_after_clip", None)
            copied["index"] = index
            # A branch is an explicit editorial choice to freeze this saved
            # take even if the currently visible prompt widgets have since
            # changed. Toggle reuse_checkpoint off to intentionally rerender
            # a frozen branch clip; the new publication removes this marker.
            copied["frozen_branch"] = True
            copied["path"] = os.path.relpath(
                target_path, output_root
            ).replace(os.sep, "/")
            copied["mtime_ns"] = int(os.stat(target_path).st_mtime_ns)
            destination["clips"][str(index)] = copied
        _atomic_json(destination_manifest, destination)
    return public_checkpoint_status(new_prefix)


def publish_checkpoint(
    checkpoint_prefix, index, path, video_shape, audio_shape,
    prompt=None, unique_id=None,
):
    index = int(index)
    if index < 1:
        return None
    expected = checkpoint_path(checkpoint_prefix, index)
    if os.path.realpath(path) != expected:
        raise ValueError(
            "h3_checkpoint_manifest: checkpoint path does not match its prefix/index"
        )
    stat = os.stat(path)
    digest = file_sha256(path)
    generation_signature = _dependency_signature(prompt, unique_id)
    target = manifest_path(checkpoint_prefix)
    root = _output_root()

    with _LOCK:
        document = _load(target) or {
            "format": FORMAT,
            "checkpoint_prefix": str(checkpoint_prefix).replace("\\", "/"),
            "clips": {},
        }
        document["format"] = FORMAT
        clips = document["clips"]
        old = clips.get(str(index))
        old_digest = str((old or {}).get("sha256") or "")
        old_signature = str((old or {}).get("generation_signature") or "")
        if old_digest != digest or old_signature != generation_signature:
            for key, record in clips.items():
                if int(key) > index and isinstance(record, dict):
                    record["stale_after_clip"] = index

        predecessor = clips.get(str(index - 1)) if index > 1 else None
        record = {
            "index": index,
            "path": os.path.relpath(path, root).replace(os.sep, "/"),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
            "video_shape": [int(value) for value in video_shape],
            "audio_shape": [int(value) for value in audio_shape],
            "predecessor_sha256": (
                str(predecessor.get("sha256") or "")
                if isinstance(predecessor, dict) else ""
            ),
            "generation_signature": generation_signature,
            "published_ns": time.time_ns(),
        }
        record["quality_mode"] = "turbo_8_step"
        clips[str(index)] = record
        document["updated_ns"] = time.time_ns()
        _atomic_json(target, document)
    return target


def validate_checkpoint_set(checkpoint_prefix, paths):
    target = manifest_path(checkpoint_prefix)
    with _LOCK:
        document = _load(target)
    if document is None:
        raise FileNotFoundError(
            "h3_checkpoint_manifest: manifest is missing. Regenerate/save "
            "the required clips with the current reusable Digital Human checkpoint pipeline."
        )

    clips = document["clips"]
    previous = ""
    for index, path in enumerate(paths, start=1):
        record = clips.get(str(index))
        if not isinstance(record, dict):
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d has no published record" % index
            )
        usable, reason = _record_content_usable(
            checkpoint_prefix, index, document,
            verify_hash=True,
        )
        if not usable:
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d is %s" % (index, reason)
            )
        expected = os.path.realpath(
            os.path.join(_output_root(), *PurePosixPath(record["path"]).parts)
        )
        if expected != os.path.realpath(path):
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d path differs from its record"
                % index
            )
        stat = os.stat(path)
        if int(record.get("size", -1)) != int(stat.st_size):
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d checkpoint size changed" % index
            )
        # Always verify content. Size/mtime are useful diagnostics and cache
        # invalidators, but neither is an integrity boundary: both can remain
        # unchanged after an interrupted copy or external replacement.
        if file_sha256(path) != str(record.get("sha256") or ""):
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d checkpoint content changed"
                % index
            )
        predecessor = str(record.get("predecessor_sha256") or "")
        if index > 1 and predecessor != previous:
            raise ValueError(
                "h3_checkpoint_manifest: Clip %d was generated from a different "
                "predecessor. Regenerate it and all later clips." % index
            )
        previous = str(record.get("sha256") or "")
    return document
