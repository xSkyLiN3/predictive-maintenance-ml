"""Canonical, atomic I/O helpers for small versioned ML artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """Raised when an existing artifact is incomplete, invalid or unexpectedly changed."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite numeric values."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a regular file in bounded chunks."""
    if not path.is_file():
        raise ArtifactIntegrityError(f"Expected a regular artifact file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactIntegrityError(f"Could not hash artifact: {path}") from error
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with a user-facing integrity error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"JSON artifact must contain an object: {path}")
    return payload


def write_bytes_atomic_if_changed(path: Path, content: bytes) -> None:
    """Atomically replace a small file only when its bytes changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return
    if path.exists() and not path.is_file():
        raise ArtifactIntegrityError(f"Expected a regular output file: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic_if_changed(path: Path, payload: Any) -> None:
    """Write canonical JSON atomically."""
    write_bytes_atomic_if_changed(path, canonical_json_bytes(payload))


def write_text_atomic_if_changed(path: Path, content: str) -> None:
    """Write UTF-8 text atomically."""
    write_bytes_atomic_if_changed(path, content.encode("utf-8"))
