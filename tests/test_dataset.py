"""Offline tests for retrieval, integrity and provenance metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from predictive_maintenance.dataset import (
    DatasetIntegrityError,
    DatasetSource,
    retrieve_dataset,
    verify_dataset_files,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _create_source(tmp_path: Path) -> tuple[Path, DatasetSource, bytes]:
    csv_content = b"column_a,column_b\n1,2\n"
    source_archive = tmp_path / "source.zip"
    with ZipFile(source_archive, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ai4i2020.csv", csv_content)
    archive_content = source_archive.read_bytes()
    source = DatasetSource(
        name="Test dataset",
        uci_repository_id=601,
        landing_page_url="https://example.test/dataset",
        source_url=source_archive.as_uri(),
        doi="10.example/test",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        archive_filename="snapshot.zip",
        archive_size_bytes=len(archive_content),
        archive_sha256=_sha256(archive_content),
        csv_filename="ai4i2020.csv",
        csv_size_bytes=len(csv_content),
        csv_sha256=_sha256(csv_content),
    )
    return source_archive, source, csv_content


def test_retrieve_dataset_writes_verified_files_and_metadata(tmp_path: Path) -> None:
    source_archive, source, csv_content = _create_source(tmp_path)
    raw_dir = tmp_path / "raw"

    paths = retrieve_dataset(raw_dir, source=source)

    assert paths.archive.read_bytes() == source_archive.read_bytes()
    assert paths.csv.read_bytes() == csv_content
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["dataset"]["source_url"] == source.source_url
    assert metadata["dataset"]["license"]["name"] == "CC BY 4.0"
    assert metadata["archive"]["sha256"] == source.archive_sha256
    assert metadata["csv"]["sha256"] == source.csv_sha256
    assert metadata["retrieved_at_utc"].endswith("Z")
    assert verify_dataset_files(raw_dir, source=source) == paths


def test_retrieve_dataset_reuses_valid_local_snapshot_without_source(tmp_path: Path) -> None:
    source_archive, source, _ = _create_source(tmp_path)
    raw_dir = tmp_path / "raw"
    first_paths = retrieve_dataset(raw_dir, source=source)
    original_metadata = first_paths.metadata.read_bytes()
    source_archive.unlink()

    second_paths = retrieve_dataset(raw_dir, source=source)

    assert second_paths == first_paths
    assert second_paths.metadata.read_bytes() == original_metadata


def test_retrieve_dataset_rejects_unexpected_archive_hash(tmp_path: Path) -> None:
    _, source, _ = _create_source(tmp_path)
    invalid_source = replace(source, archive_sha256="0" * 64)
    raw_dir = tmp_path / "raw"

    with pytest.raises(DatasetIntegrityError, match="SHA-256"):
        retrieve_dataset(raw_dir, source=invalid_source)

    assert not (raw_dir / source.archive_filename).exists()
    assert not list(raw_dir.glob(".ai4i-*"))


def test_verify_dataset_rejects_a_mutated_csv(tmp_path: Path) -> None:
    _, source, _ = _create_source(tmp_path)
    raw_dir = tmp_path / "raw"
    paths = retrieve_dataset(raw_dir, source=source)
    paths.csv.write_bytes(b"changed")

    with pytest.raises(DatasetIntegrityError, match="size"):
        verify_dataset_files(raw_dir, source=source)


def test_retrieve_dataset_rejects_an_archive_with_extra_members(tmp_path: Path) -> None:
    _, source, csv_content = _create_source(tmp_path)
    invalid_archive = tmp_path / "invalid.zip"
    with ZipFile(invalid_archive, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ai4i2020.csv", csv_content)
        archive.writestr("unexpected.txt", b"not part of the dataset")
    invalid_content = invalid_archive.read_bytes()
    invalid_source = replace(
        source,
        source_url=invalid_archive.as_uri(),
        archive_size_bytes=len(invalid_content),
        archive_sha256=_sha256(invalid_content),
    )

    with pytest.raises(DatasetIntegrityError, match="Unexpected archive contents"):
        retrieve_dataset(tmp_path / "raw", source=invalid_source)

    assert not (tmp_path / "raw" / invalid_source.archive_filename).exists()
    assert not list((tmp_path / "raw").glob(".ai4i-*"))


def test_retrieve_dataset_refuses_an_unmanaged_archive_without_metadata(tmp_path: Path) -> None:
    source_archive, source, _ = _create_source(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    unmanaged_archive = raw_dir / source.archive_filename
    unmanaged_archive.write_bytes(source_archive.read_bytes())

    with pytest.raises(DatasetIntegrityError, match="retrieval date cannot be inferred"):
        retrieve_dataset(raw_dir, source=source)

    assert unmanaged_archive.exists()
