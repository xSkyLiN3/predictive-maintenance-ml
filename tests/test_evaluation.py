"""Tests for guarded final M3 holdout consumption using synthetic files only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance.evaluation as evaluation
import predictive_maintenance.modeling as modeling
from predictive_maintenance.artifact_io import ArtifactIntegrityError, sha256_path
from predictive_maintenance.evaluation import (
    FinalEvaluationError,
    compute_final_metrics,
    evaluate_final_holdout,
)
from predictive_maintenance.modeling import build_candidate_pipelines, run_training_selection
from predictive_maintenance.splitting import load_holdout_partition_for_final_evaluation
from predictive_maintenance.validation import MODELING_COLUMNS, TARGET_COLUMN


def _frame(row_count: int, *, positive_every: int = 5) -> pd.DataFrame:
    target = np.array(
        [1 if index % positive_every == 0 else 0 for index in range(row_count)], dtype=int
    )
    return pd.DataFrame(
        {
            "Type": [["L", "M", "H"][index % 3] for index in range(row_count)],
            "Air temperature [K]": [
                299.0 + 2.0 * target[index] + (index % 5) / 10 for index in range(row_count)
            ],
            "Process temperature [K]": [
                309.0 + target[index] + (index % 7) / 10 for index in range(row_count)
            ],
            "Rotational speed [rpm]": [
                1500 - 120 * target[index] + index % 30 for index in range(row_count)
            ],
            "Torque [Nm]": [
                35.0 + 20.0 * target[index] + (index % 4) / 10 for index in range(row_count)
            ],
            "Tool wear [min]": [
                50 + 100 * target[index] + index % 40 for index in range(row_count)
            ],
            TARGET_COLUMN: target,
        },
        columns=MODELING_COLUMNS,
    )


def _small_pipelines():
    pipelines = build_candidate_pipelines()
    pipelines["random_forest"].set_params(model__n_estimators=8, model__max_depth=4)
    return pipelines


def _ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "holdout_access" / f"{'c' * 64}.json"


def _selection_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pd.DataFrame, dict, object]:
    monkeypatch.chdir(tmp_path)
    training = _frame(150)
    manifest = {
        "source": {"filename": "ai4i2020.csv", "sha256": "a" * 64, "rows": 180},
        "split": {"random_seed": 42},
        "training": {"filename": "train.csv", "rows": 150, "sha256": "b" * 64},
        "holdout": {"filename": "holdout.csv", "rows": 30, "sha256": "c" * 64},
    }
    monkeypatch.setattr(modeling, "load_training_partition", lambda *_: (training.copy(), manifest))
    monkeypatch.setattr(modeling, "build_candidate_pipelines", _small_pipelines)
    artifacts = run_training_selection(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        evaluation,
        "load_training_partition",
        lambda *_: (training.copy(), manifest),
    )
    return training, manifest, artifacts


def test_final_metrics_match_known_scores_and_inclusive_threshold() -> None:
    metrics = compute_final_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.5, 0.5, 0.9]),
        0.5,
    )

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["accuracy"] == 0.75
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(0.875)


@pytest.mark.parametrize(
    "invalid_target",
    [
        pytest.param(np.array([0, 0.5, 1]), id="fractional"),
        pytest.param(np.array(["0", "0", "1"]), id="strings"),
        pytest.param(np.array([0, np.nan, 1]), id="nan"),
    ],
)
def test_final_metrics_reject_invalid_targets_before_integer_conversion(
    invalid_target: np.ndarray,
) -> None:
    with pytest.raises(FinalEvaluationError):
        compute_final_metrics(invalid_target, np.array([0.1, 0.5, 0.9]), 0.5)


def test_final_evaluation_creates_ticket_before_one_read_and_then_uses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, _ = _selection_bundle(tmp_path, monkeypatch)
    calls: list[str] = []

    def holdout_loader(processed_dir: Path, manifest_path: Path):
        del processed_dir, manifest_path
        assert _ledger_path(tmp_path).is_file()
        calls.append("read")
        return _frame(30), manifest

    first = evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=holdout_loader,
    )
    assert calls == ["read"]
    assert first.cached is False
    assert first.evaluation.is_file()
    receipt = json.loads(first.evaluation.read_text(encoding="utf-8"))
    assert receipt["receipt"] == "final_holdout_evaluation_complete"
    assert receipt["holdout"]["application_reads_for_evaluation"] == 1

    def forbidden_loader(*_):
        raise AssertionError("cached evaluation must not read holdout")

    second = evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=forbidden_loader,
    )
    assert second.cached is True
    assert calls == ["read"]


def test_selection_tampering_is_rejected_before_ticket_or_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, training_artifacts = _selection_bundle(tmp_path, monkeypatch)
    artifact_dir = tmp_path / "artifacts"
    ticket = _ledger_path(tmp_path)

    def forbidden_loader(*_):
        raise AssertionError("tampering must be rejected before reading holdout")

    original_cv_results = training_artifacts.cv_results.read_bytes()
    cv_results = json.loads(original_cv_results)
    cv_results["candidates"]["dummy"]["average_precision_mean"] = 0.999
    training_artifacts.cv_results.write_text(json.dumps(cv_results), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash|receipt|report"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            artifact_dir,
            confirm_final_evaluation=True,
            holdout_loader=forbidden_loader,
        )
    assert not ticket.exists()
    training_artifacts.cv_results.write_bytes(original_cv_results)

    artifact_manifest = json.loads(training_artifacts.artifact_manifest.read_text(encoding="utf-8"))
    artifact_manifest["selected_model"] = "dummy"
    training_artifacts.artifact_manifest.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="pipeline bundle|selected run"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            artifact_dir,
            confirm_final_evaluation=True,
            holdout_loader=forbidden_loader,
        )
    assert not ticket.exists()


@pytest.mark.parametrize("artifact_name", ["selection_report", "cv_figure", "oof_figure"])
def test_selection_report_and_figures_are_hashed_before_holdout_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    _, _, artifacts = _selection_bundle(tmp_path, monkeypatch)
    paths = {
        "selection_report": artifacts.selection_report,
        "cv_figure": artifacts.figures[0],
        "oof_figure": artifacts.figures[1],
    }
    paths[artifact_name].write_bytes(b"tampered selection artifact")

    with pytest.raises(ArtifactIntegrityError, match="Selection report hashes"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("holdout must remain unread"),
        )

    assert not _ledger_path(tmp_path).exists()


@pytest.mark.parametrize("receipt_name", ["cv", "threshold"])
def test_coherently_rehashed_malformed_selection_receipt_fails_before_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_name: str,
) -> None:
    _, _, artifacts = _selection_bundle(tmp_path, monkeypatch)
    run_manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    if receipt_name == "cv":
        receipt_path = artifacts.cv_results
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["candidates"]["dummy"]["average_precision_by_fold"] = [0.1]
        hash_key = "cv_results_sha256"
    else:
        receipt_path = artifacts.threshold_selection
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["oof_metrics"]["f1"] = "not-a-number"
        hash_key = "threshold_selection_sha256"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    run_manifest["selection_receipts"][hash_key] = sha256_path(receipt_path)
    artifacts.run_manifest.write_text(json.dumps(run_manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="AP folds|CV|OOF|threshold|receipt"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("holdout must remain unread"),
        )

    assert not _ledger_path(tmp_path).exists()


def test_corrupted_versioned_outputs_are_recovered_from_bundle_without_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, _ = _selection_bundle(tmp_path, monkeypatch)

    first = evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=lambda *_: (_frame(30), manifest),
    )
    expected = {
        first.evaluation: first.evaluation.read_bytes(),
        first.report: first.report.read_bytes(),
        first.confusion_figure: first.confusion_figure.read_bytes(),
    }
    first.evaluation.write_text("{}", encoding="utf-8")
    first.report.write_text("corrupted report", encoding="utf-8")
    first.confusion_figure.write_bytes(b"corrupted png")

    def forbidden_loader(*_):
        raise AssertionError("bundle recovery must not read holdout")

    recovered = evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=forbidden_loader,
    )

    assert recovered.cached is True
    assert expected == {path: path.read_bytes() for path in expected}


def test_bundle_manifest_with_wrong_run_is_rejected_without_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, artifacts = _selection_bundle(tmp_path, monkeypatch)
    evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=lambda *_: (_frame(30), manifest),
    )
    bundle_manifest_path = (
        tmp_path
        / "artifacts"
        / artifacts.run_id
        / evaluation.FINAL_BUNDLE_DIRECTORY
        / evaluation.FINAL_BUNDLE_MANIFEST_FILENAME
    )
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    bundle_manifest["run_id"] = "0" * 16
    bundle_manifest_path.write_text(json.dumps(bundle_manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="bundle"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("holdout must remain unread"),
        )


def test_publication_failure_after_local_bundle_recovers_without_second_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, training_artifacts = _selection_bundle(tmp_path, monkeypatch)
    artifact_dir = tmp_path / "artifacts"
    bundle = artifact_dir / training_artifacts.run_id / evaluation.FINAL_BUNDLE_DIRECTORY
    reads: list[str] = []

    def holdout_loader(*_):
        reads.append("read")
        return _frame(30), manifest

    original_publish = evaluation._publish_versioned_bundle

    def fail_after_bundle(*_):
        assert bundle.is_dir()
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(evaluation, "_publish_versioned_bundle", fail_after_bundle)
    with pytest.raises(OSError, match="synthetic publication failure"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            artifact_dir,
            confirm_final_evaluation=True,
            holdout_loader=holdout_loader,
        )
    assert reads == ["read"]
    assert bundle.is_dir()

    monkeypatch.setattr(evaluation, "_publish_versioned_bundle", original_publish)
    recovered = evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        artifact_dir,
        confirm_final_evaluation=True,
        holdout_loader=lambda *_: pytest.fail("recovery must not reread holdout"),
    )

    assert recovered.cached is True
    assert reads == ["read"]
    assert recovered.evaluation.is_file()
    assert recovered.report.is_file()
    assert recovered.confusion_figure.is_file()


def test_different_requested_split_is_rejected_before_ticket_or_holdout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, manifest, training_artifacts = _selection_bundle(tmp_path, monkeypatch)
    different_manifest = json.loads(json.dumps(manifest))
    different_manifest["holdout"]["sha256"] = "d" * 64
    monkeypatch.setattr(
        evaluation,
        "load_training_partition",
        lambda *_: (training.copy(), different_manifest),
    )
    artifact_dir = tmp_path / "artifacts"
    ticket = _ledger_path(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="Requested split metadata differs"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "different-split.json",
            tmp_path / "reports",
            artifact_dir,
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("mismatched split must not read holdout"),
        )

    assert not ticket.exists()


def test_global_ledger_blocks_a_second_run_and_different_output_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, _ = _selection_bundle(tmp_path, monkeypatch)
    reads: list[str] = []

    evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports",
        tmp_path / "artifacts",
        confirm_final_evaluation=True,
        holdout_loader=lambda *_: (reads.append("run-a") or _frame(30), manifest),
    )
    original_versions = modeling._runtime_versions()
    monkeypatch.setattr(
        modeling,
        "_runtime_versions",
        lambda: {**original_versions, "python": "3.12.synthetic-second-run"},
    )
    second = run_training_selection(
        tmp_path / "processed",
        tmp_path / "split.json",
        tmp_path / "reports-b",
        tmp_path / "artifacts-b",
    )

    with pytest.raises(FinalEvaluationError, match="another model run"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports-b",
            tmp_path / "artifacts-b",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("a second run must not open holdout"),
        )

    assert second.run_id != json.loads(_ledger_path(tmp_path).read_text())["run_id"]
    assert reads == ["run-a"]


def test_unsafe_run_id_and_active_run_change_are_rejected_before_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, artifacts = _selection_bundle(tmp_path, monkeypatch)
    active_path = tmp_path / "reports" / "active_run.json"
    active_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "../escape",
                "directory": "../escape",
                "baseline_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FinalEvaluationError, match="incomplete|dummy gate"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("unsafe run id must not reach holdout"),
        )
    assert not _ledger_path(tmp_path).exists()

    active_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": artifacts.run_id,
                "directory": artifacts.run_id,
                "baseline_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    original_load = evaluation.joblib.load

    def racing_load(path: Path):
        pipeline = original_load(path)
        original_predict = pipeline.predict_proba

        def mutate_active_then_predict(features):
            active_path.write_bytes(active_path.read_bytes() + b" ")
            return original_predict(features)

        pipeline.predict_proba = mutate_active_then_predict
        return pipeline

    monkeypatch.setattr(evaluation.joblib, "load", racing_load)
    with pytest.raises(FinalEvaluationError, match="active training run changed"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("race must abort before holdout"),
        )
    assert not _ledger_path(tmp_path).exists()


def test_incomplete_ticket_blocks_retry_without_reading_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _ = _selection_bundle(tmp_path, monkeypatch)

    with pytest.raises(OSError, match="synthetic pre-bundle failure"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: (_ for _ in ()).throw(
                OSError("synthetic pre-bundle failure")
            ),
        )
    assert _ledger_path(tmp_path).is_file()

    with pytest.raises(FinalEvaluationError, match="ledger"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            tmp_path / "reports",
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("holdout must remain unread"),
        )


def test_gate_failure_stops_before_any_holdout_loader(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "active_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "blocked",
                "baseline_gate_passed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FinalEvaluationError, match="dummy gate"):
        evaluate_final_holdout(
            tmp_path / "processed",
            tmp_path / "split.json",
            report_dir,
            tmp_path / "artifacts",
            confirm_final_evaluation=True,
            holdout_loader=lambda *_: pytest.fail("holdout must remain unread"),
        )


def test_holdout_loader_reads_synthetic_file_once_and_validates_from_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    holdout_path = processed_dir / "holdout.csv"
    holdout = _frame(2_000, positive_every=20)
    holdout.to_csv(holdout_path, index=False, lineterminator="\n")
    holdout_sha = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": {
            "filename": "ai4i2020.csv",
            "sha256": "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e",
            "rows": 10_000,
        },
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "random_seed": 42,
            "holdout_fraction": 0.2,
            "stratified_by": TARGET_COLUMN,
            "rows_sorted_within_partitions": True,
        },
        "training": {
            "filename": "train.csv",
            "rows": 8_000,
            "sha256": "a" * 64,
            "columns": list(MODELING_COLUMNS),
        },
        "holdout": {
            "filename": "holdout.csv",
            "rows": 2_000,
            "sha256": holdout_sha,
            "columns": list(MODELING_COLUMNS),
            "access_policy": "Do not inspect until the final M3 evaluation.",
        },
        "versions": {"python": "3.12.0", "pandas": "3.0.5", "scikit_learn": "1.9.0"},
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_read_bytes = Path.read_bytes
    reads = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == holdout_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    loaded, _ = load_holdout_partition_for_final_evaluation(processed_dir, manifest_path)

    assert len(loaded) == 2_000
    assert tuple(loaded.columns) == MODELING_COLUMNS
    assert reads == 1
