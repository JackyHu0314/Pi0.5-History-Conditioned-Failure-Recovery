#!/usr/bin/env python3
"""Hardlink identical immutable Orbax checkpoint data blobs.

Only finalized checkpoint step directories are considered.  Metadata, manifests,
locks, assets, and any file outside an OCDBT/TensorStore data directory are never
modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

INDEX_VERSION = 1
_BLOB_NAME = re.compile(r"[0-9a-f]{32}")
_PROCESS_DIR = re.compile(r"ocdbt\.process_\d+")
_ZARR_CHUNK_NAME = re.compile(r"\d+(?:\.\d+)*")
_FINALIZED_LOG_PATH = re.compile(
    r"Finished saving checkpoint \(finalized tmp dir\) to [`']([^`']+)[`']"
)
_TEMP_NAME_PARTS = (".tmp", ".orbax-checkpoint-tmp", "checkpoint_tmp")


@dataclasses.dataclass
class Stats:
    checkpoints: int = 0
    files_hashed: int = 0
    unique_blobs: int = 0
    duplicate_blobs: int = 0
    already_linked: int = 0
    relinked: int = 0
    logical_bytes_saved: int = 0
    allocated_bytes_saved: int = 0
    invalid_index_entries: int = 0
    cross_device_duplicates: int = 0


@dataclasses.dataclass(frozen=True)
class HashedFile:
    path: Path
    size: int
    allocated_size: int
    device: int
    inode: int
    links: int
    mtime_ns: int
    digest: str

    @property
    def key(self) -> str:
        return f"{self.size}:{self.digest}"


def _read_committed_metadata(checkpoint_dir: Path) -> bool:
    marker = checkpoint_dir / "_CHECKPOINT_METADATA"
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    commit = metadata.get("commit_timestamp_nsecs")
    return isinstance(commit, int) and commit > 0


def _is_temporary_path(path: Path) -> bool:
    return any(
        part.startswith(_TEMP_NAME_PARTS) or part.endswith(_TEMP_NAME_PARTS)
        for part in path.parts
    )


def _discover_from_root(root: Path) -> set[Path]:
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")
    checkpoints = set()
    for marker in root.rglob("_CHECKPOINT_METADATA"):
        checkpoint = marker.parent.resolve()
        if (
            checkpoint.name not in {"params", "train_state"}
            and not _is_temporary_path(checkpoint)
            and _read_committed_metadata(checkpoint)
        ):
            checkpoints.add(checkpoint)
    return checkpoints


def _discover_from_log(log_path: Path) -> set[Path]:
    checkpoints = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FINALIZED_LOG_PATH.search(line)
        if match:
            checkpoint = Path(match.group(1)).resolve()
            if checkpoint.is_dir() and not _is_temporary_path(checkpoint):
                checkpoints.add(checkpoint)
    return checkpoints


def discover_checkpoints(
    *,
    checkpoint_roots: Sequence[Path],
    checkpoint_dirs: Sequence[Path],
    completion_logs: Sequence[Path],
) -> list[Path]:
    checkpoints = set()
    for root in checkpoint_roots:
        checkpoints.update(_discover_from_root(root.resolve()))
    for log_path in completion_logs:
        checkpoints.update(_discover_from_log(log_path.resolve()))
    for checkpoint_dir in checkpoint_dirs:
        checkpoint = checkpoint_dir.resolve()
        if not checkpoint.is_dir():
            raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
        if _is_temporary_path(checkpoint):
            raise ValueError(f"refusing temporary checkpoint directory: {checkpoint}")
        if not _read_committed_metadata(checkpoint) and not checkpoint.name.isdecimal():
            raise ValueError(
                f"explicit checkpoint must be a terminal step directory: {checkpoint}"
            )
        checkpoints.add(checkpoint)
    return sorted(checkpoints)


def _is_data_blob(path: Path, checkpoint_dir: Path, storage_format: str) -> bool:
    try:
        parts = path.relative_to(checkpoint_dir).parts
    except ValueError:
        return False
    if storage_format == "ocdbt" and len(parts) == 3:
        item, data_dir, name = parts
        if (
            item in {"params", "train_state"}
            and data_dir == "d"
            and _BLOB_NAME.fullmatch(name) is not None
        ):
            return True
    if storage_format == "ocdbt" and len(parts) == 4:
        item, process_dir, data_dir, name = parts
        if (
            item in {"params", "train_state"}
            and _PROCESS_DIR.fullmatch(process_dir) is not None
            and data_dir == "d"
            and _BLOB_NAME.fullmatch(name) is not None
        ):
            return True
    if (
        storage_format == "zarr"
        and len(parts) >= 3
        and parts[0] in {"params", "train_state"}
        and _ZARR_CHUNK_NAME.fullmatch(parts[-1]) is not None
    ):
        return (path.parent / ".zarray").is_file()
    return False


def iter_data_blobs(
    checkpoint_dirs: Sequence[Path], *, storage_format: str
) -> Iterable[Path]:
    for checkpoint_dir in checkpoint_dirs:
        for item in ("params", "train_state"):
            item_dir = checkpoint_dir / item
            if not item_dir.is_dir():
                continue
            for path in item_dir.rglob("*"):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and _is_data_blob(path, checkpoint_dir, storage_format)
                ):
                    yield path


def _hash_stable_file(path: Path) -> HashedFile:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while hashing: {path}")
    return HashedFile(
        path=path,
        size=before.st_size,
        allocated_size=before.st_blocks * 512,
        device=before.st_dev,
        inode=before.st_ino,
        links=before.st_nlink,
        mtime_ns=before.st_mtime_ns,
        digest=digest.hexdigest(),
    )


def _load_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION or not isinstance(
        payload.get("entries"), dict
    ):
        raise ValueError(f"unsupported dedup index: {index_path}")
    return {str(key): str(value) for key, value in payload["entries"].items()}


def _write_index(index_path: Path, entries: dict[str, str]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": INDEX_VERSION, "entries": dict(sorted(entries.items()))}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, index_path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_index_candidate(
    key: str,
    entries: dict[str, str],
    *,
    expected: HashedFile,
    validation_cache: dict[tuple[str, int, int, int], bool],
) -> Path | None:
    stored_path = entries.get(key)
    if stored_path is None:
        return None
    candidate = Path(stored_path)
    try:
        candidate_stat = candidate.stat(follow_symlinks=False)
    except OSError:
        entries.pop(key, None)
        return None
    if (
        not stat.S_ISREG(candidate_stat.st_mode)
        or candidate_stat.st_size != expected.size
        or candidate_stat.st_dev != expected.device
    ):
        entries.pop(key, None)
        return None
    cache_key = (
        str(candidate),
        candidate_stat.st_ino,
        candidate_stat.st_size,
        candidate_stat.st_mtime_ns,
    )
    valid = validation_cache.get(cache_key)
    if valid is None:
        valid = _hash_stable_file(candidate).digest == expected.digest
        validation_cache[cache_key] = valid
    if not valid:
        entries.pop(key, None)
        return None
    return candidate


def _same_inode(first: Path, second: Path) -> bool:
    first_stat = first.stat(follow_symlinks=False)
    second_stat = second.stat(follow_symlinks=False)
    return (
        first_stat.st_dev == second_stat.st_dev
        and first_stat.st_ino == second_stat.st_ino
    )


def _atomic_relink(canonical: Path, target: HashedFile) -> None:
    current = target.path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
        target.device,
        target.inode,
        target.size,
        target.mtime_ns,
    ):
        raise RuntimeError(f"target changed before relink: {target.path}")
    temporary = target.path.with_name(
        f".{target.path.name}.dedup-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    try:
        os.link(canonical, temporary, follow_symlinks=False)
        os.replace(temporary, target.path)
    finally:
        temporary.unlink(missing_ok=True)


def deduplicate_finished_checkpoints(
    *,
    checkpoint_roots: Sequence[Path] = (),
    checkpoint_dirs: Sequence[Path] = (),
    completion_logs: Sequence[Path] = (),
    index_path: Path,
    storage_format: str = "ocdbt",
    dry_run: bool = False,
) -> dict[str, object]:
    if storage_format not in {"ocdbt", "zarr"}:
        raise ValueError(f"unsupported storage format: {storage_format}")
    checkpoints = discover_checkpoints(
        checkpoint_roots=checkpoint_roots,
        checkpoint_dirs=checkpoint_dirs,
        completion_logs=completion_logs,
    )
    entries = _load_index(index_path)
    validation_cache: dict[tuple[str, int, int, int], bool] = {}
    stats = Stats(checkpoints=len(checkpoints))

    for path in iter_data_blobs(checkpoints, storage_format=storage_format):
        hashed = _hash_stable_file(path)
        stats.files_hashed += 1
        had_entry = hashed.key in entries
        candidate = _validated_index_candidate(
            hashed.key,
            entries,
            expected=hashed,
            validation_cache=validation_cache,
        )
        if had_entry and candidate is None:
            stats.invalid_index_entries += 1
        if candidate is None:
            entries[hashed.key] = str(path)
            stats.unique_blobs += 1
            continue
        stats.duplicate_blobs += 1
        if _same_inode(candidate, path):
            stats.already_linked += 1
            continue
        if candidate.stat(follow_symlinks=False).st_dev != hashed.device:
            stats.cross_device_duplicates += 1
            continue
        if dry_run:
            stats.relinked += 1
        else:
            _atomic_relink(candidate, hashed)
            if not _same_inode(candidate, path):
                raise RuntimeError(f"relink verification failed: {path}")
            stats.relinked += 1
        if hashed.links == 1:
            stats.logical_bytes_saved += hashed.size
            stats.allocated_bytes_saved += hashed.allocated_size

    if not dry_run:
        _write_index(index_path, entries)
    return {
        **dataclasses.asdict(stats),
        "dry_run": dry_run,
        "storage_format": storage_format,
        "index_path": str(index_path.resolve()),
        "checkpoint_dirs": [str(path) for path in checkpoints],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        type=Path,
        help="Recursively discover checkpoints with committed _CHECKPOINT_METADATA. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        type=Path,
        help="Explicit terminal checkpoint step directory. Repeatable.",
    )
    parser.add_argument(
        "--completion-log",
        action="append",
        default=[],
        type=Path,
        help="Orbax log containing finalized checkpoint paths. Repeatable.",
    )
    parser.add_argument(
        "--index-path",
        required=True,
        type=Path,
        help="Persistent verified content index.",
    )
    parser.add_argument(
        "--storage-format",
        choices=("ocdbt", "zarr"),
        default="ocdbt",
        help="Checkpoint data layout. Defaults to the existing OCDBT format.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.checkpoint_root or args.checkpoint_dir or args.completion_log):
        parser.error(
            "at least one checkpoint root, checkpoint directory, or completion log is required"
        )
    return args


def main() -> None:
    args = _parse_args()
    result = deduplicate_finished_checkpoints(
        checkpoint_roots=args.checkpoint_root,
        checkpoint_dirs=args.checkpoint_dir,
        completion_logs=args.completion_log,
        index_path=args.index_path,
        storage_format=args.storage_format,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
