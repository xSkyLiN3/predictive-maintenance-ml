"""Reproducible retrieval and integrity checks for the AI4I 2020 dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class DatasetSource:
    """Immutable identity and integrity information for a dataset snapshot."""

    name: str
    uci_repository_id: int
    landing_page_url: str
    source_url: str
    doi: str
    license_name: str
    license_url: str
    archive_filename: str
    archive_size_bytes: int
    archive_sha256: str
    csv_filename: str
    csv_size_bytes: int
    csv_sha256: str


OFFICIAL_DATASET = DatasetSource(
    name="AI4I 2020 Predictive Maintenance Dataset",
    uci_repository_id=601,
    landing_page_url=(
        "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
    ),
    source_url=(
        "https://archive.ics.uci.edu/static/public/601/"
        "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
    ),
    doi="10.24432/C5HS5C",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    archive_filename="ai4i_2020_predictive_maintenance.zip",
    archive_size_bytes=522_170,
    archive_sha256="f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252",
    csv_filename="ai4i2020.csv",
    csv_size_bytes=522_048,
    csv_sha256="dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e",
)

DEFAULT_RAW_DATA_DIR = Path("data/raw")
DOWNLOAD_METADATA_FILENAME = "download_metadata.json"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class DatasetIntegrityError(RuntimeError):
    """Raised when downloaded or local data does not match the fixed snapshot."""


@dataclass(frozen=True)
class DatasetPaths:
    """Paths produced by the data retrieval step."""

    archive: Path
    csv: Path
    metadata: Path


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_for(raw_dir: Path, source: DatasetSource) -> DatasetPaths:
    return DatasetPaths(
        archive=raw_dir / source.archive_filename,
        csv=raw_dir / source.csv_filename,
        metadata=raw_dir / DOWNLOAD_METADATA_FILENAME,
    )


def _validate_file(path: Path, expected_size: int, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise DatasetIntegrityError(f"Missing {label}: {path}")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DatasetIntegrityError(
            f"Unexpected {label} size for {path}: expected {expected_size}, got {actual_size}."
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise DatasetIntegrityError(
            f"Unexpected {label} SHA-256 for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def _temporary_path(parent: Path, suffix: str) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=".ai4i-",
        suffix=suffix,
    )
    os.close(file_descriptor)
    return Path(temporary_name)


def _copy_with_size_limit(source: BinaryIO, destination: BinaryIO, maximum_bytes: int) -> None:
    copied_bytes = 0
    while chunk := source.read(min(DOWNLOAD_CHUNK_SIZE, maximum_bytes + 1 - copied_bytes)):
        copied_bytes += len(chunk)
        if copied_bytes > maximum_bytes:
            raise DatasetIntegrityError(f"Received more than the expected {maximum_bytes} bytes.")
        destination.write(chunk)


def _download_archive(
    destination: Path,
    source: DatasetSource,
    timeout_seconds: float,
) -> None:
    temporary_path = _temporary_path(destination.parent, ".download")
    request = Request(
        source.source_url,
        headers={"User-Agent": "machine-failure-risk-classifier/0.1"},
    )

    try:
        with (
            urlopen(request, timeout=timeout_seconds) as response,  # noqa: S310
            temporary_path.open("wb") as file_handle,
        ):
            _copy_with_size_limit(response, file_handle, source.archive_size_bytes)
        _validate_file(
            temporary_path,
            source.archive_size_bytes,
            source.archive_sha256,
            "downloaded archive",
        )
        temporary_path.replace(destination)
    except (OSError, URLError) as error:
        raise DatasetIntegrityError(
            f"Could not download the official dataset from {source.source_url}: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_csv(archive_path: Path, destination: Path, source: DatasetSource) -> None:
    temporary_path = _temporary_path(destination.parent, ".csv")
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if member_names != [source.csv_filename]:
                raise DatasetIntegrityError(
                    "Unexpected archive contents: "
                    f"expected only {source.csv_filename!r}, got {member_names!r}."
                )
            member = members[0]
            unix_file_type = (member.external_attr >> 16) & 0o170000
            if member.is_dir() or unix_file_type == 0o120000 or member.flag_bits & 0x1:
                raise DatasetIntegrityError("The dataset ZIP member is not a regular plain file.")
            if member.file_size != source.csv_size_bytes:
                raise DatasetIntegrityError(
                    "Unexpected uncompressed CSV size declared by ZIP: "
                    f"expected {source.csv_size_bytes}, got {member.file_size}."
                )
            with (
                archive.open(member) as compressed_file,
                temporary_path.open("wb") as extracted_file,
            ):
                _copy_with_size_limit(
                    compressed_file,
                    extracted_file,
                    source.csv_size_bytes,
                )

        _validate_file(
            temporary_path,
            source.csv_size_bytes,
            source.csv_sha256,
            "extracted CSV",
        )
        temporary_path.replace(destination)
    except BadZipFile as error:
        raise DatasetIntegrityError(f"Invalid ZIP archive: {archive_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _metadata_payload(source: DatasetSource, retrieved_at_utc: str) -> dict[str, Any]:
    source_values = asdict(source)
    return {
        "schema_version": 1,
        "retrieved_at_utc": retrieved_at_utc,
        "dataset": {
            "name": source_values["name"],
            "uci_repository_id": source_values["uci_repository_id"],
            "landing_page_url": source_values["landing_page_url"],
            "source_url": source_values["source_url"],
            "doi": source_values["doi"],
            "license": {
                "name": source_values["license_name"],
                "url": source_values["license_url"],
            },
        },
        "archive": {
            "filename": source_values["archive_filename"],
            "size_bytes": source_values["archive_size_bytes"],
            "sha256": source_values["archive_sha256"],
        },
        "csv": {
            "filename": source_values["csv_filename"],
            "size_bytes": source_values["csv_size_bytes"],
            "sha256": source_values["csv_sha256"],
        },
    }


def _write_metadata(path: Path, source: DatasetSource) -> None:
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = _metadata_payload(source, retrieved_at)
    temporary_path = _temporary_path(path.parent, ".json")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_metadata(path: Path, source: DatasetSource) -> None:
    if not path.is_file():
        raise DatasetIntegrityError(f"Missing download metadata: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(f"Invalid download metadata: {path}") from error

    if not isinstance(payload, dict):
        raise DatasetIntegrityError(f"Invalid download metadata structure: {path}")
    retrieved_at = payload.get("retrieved_at_utc")
    if not isinstance(retrieved_at, str) or not retrieved_at.endswith("Z"):
        raise DatasetIntegrityError(f"Invalid UTC retrieval timestamp in metadata: {path}")
    try:
        parsed_timestamp = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise DatasetIntegrityError(
            f"Invalid UTC retrieval timestamp in metadata: {path}"
        ) from error
    if parsed_timestamp.tzinfo != UTC:
        raise DatasetIntegrityError(f"Invalid UTC retrieval timestamp in metadata: {path}")

    expected = _metadata_payload(source, retrieved_at)
    if payload != expected:
        raise DatasetIntegrityError(
            f"Download metadata does not match the configured dataset snapshot: {path}"
        )


def verify_dataset_files(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    *,
    source: DatasetSource = OFFICIAL_DATASET,
) -> DatasetPaths:
    """Verify the archive, extracted CSV and provenance metadata without network access."""
    paths = _paths_for(raw_dir, source)
    _validate_file(
        paths.archive,
        source.archive_size_bytes,
        source.archive_sha256,
        "dataset archive",
    )
    _validate_file(paths.csv, source.csv_size_bytes, source.csv_sha256, "dataset CSV")
    _validate_metadata(paths.metadata, source)
    return paths


def retrieve_dataset(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    *,
    source: DatasetSource = OFFICIAL_DATASET,
    timeout_seconds: float = 60.0,
) -> DatasetPaths:
    """Retrieve the fixed dataset snapshot once and refuse silent replacement."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths_for(raw_dir, source)
    archive_exists = paths.archive.exists()
    csv_exists = paths.csv.exists()
    metadata_exists = paths.metadata.exists()

    if archive_exists and not metadata_exists:
        raise DatasetIntegrityError(
            "Found an unmanaged dataset archive without retrieval metadata. "
            "Its retrieval date cannot be inferred; remove or move it before downloading."
        )
    if metadata_exists and not archive_exists:
        raise DatasetIntegrityError("Found retrieval metadata without its dataset archive.")
    if csv_exists and not archive_exists:
        raise DatasetIntegrityError("Found an extracted CSV without its dataset archive.")

    created_paths: list[Path] = []
    try:
        if archive_exists:
            _validate_file(
                paths.archive,
                source.archive_size_bytes,
                source.archive_sha256,
                "existing dataset archive",
            )
        else:
            _download_archive(paths.archive, source, timeout_seconds)
            created_paths.append(paths.archive)

        if csv_exists:
            _validate_file(
                paths.csv,
                source.csv_size_bytes,
                source.csv_sha256,
                "existing dataset CSV",
            )
        else:
            _extract_csv(paths.archive, paths.csv, source)
            created_paths.append(paths.csv)

        if metadata_exists:
            _validate_metadata(paths.metadata, source)
        else:
            _write_metadata(paths.metadata, source)
            created_paths.append(paths.metadata)

        return verify_dataset_files(raw_dir, source=source)
    except Exception:
        for created_path in reversed(created_paths):
            created_path.unlink(missing_ok=True)
        raise
