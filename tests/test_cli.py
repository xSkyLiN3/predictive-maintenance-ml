"""Tests for user-facing failures in the local data CLI."""

import json
from pathlib import Path

import pytest

from predictive_maintenance.cli import main
from predictive_maintenance.eda import EDAArtifacts
from predictive_maintenance.evaluation import FinalEvaluationArtifacts
from predictive_maintenance.modeling import TrainingArtifacts
from predictive_maintenance.splitting import SplitArtifacts


def test_cli_reports_an_invalid_raw_directory_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_raw_dir = tmp_path / "not-a-directory"
    invalid_raw_dir.write_text("file", encoding="utf-8")

    exit_code = main(["download", "--raw-dir", str(invalid_raw_dir)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err


def test_cli_split_delegates_paths_and_prints_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest = tmp_path / "split.json"
    expected = SplitArtifacts(
        training_csv=processed_dir / "train.csv",
        holdout_csv=processed_dir / "holdout.csv",
        manifest=manifest,
        training_rows=80,
        holdout_rows=20,
        training_sha256="a" * 64,
        holdout_sha256="b" * 64,
    )
    received: list[tuple[Path, Path, Path]] = []

    def fake_materialize(raw: Path, processed: Path, output_manifest: Path) -> SplitArtifacts:
        received.append((raw, processed, output_manifest))
        return expected

    monkeypatch.setattr("predictive_maintenance.cli.materialize_split", fake_materialize)

    exit_code = main(
        [
            "split",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--manifest",
            str(manifest),
        ]
    )

    assert exit_code == 0
    assert received == [(raw_dir, processed_dir, manifest)]
    assert json.loads(capsys.readouterr().out) == expected.to_dict()


def test_cli_eda_delegates_paths_and_prints_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    manifest = tmp_path / "split.json"
    report_dir = tmp_path / "reports"
    expected = EDAArtifacts(
        report=report_dir / "EDA_REPORT.md",
        summary=report_dir / "summary.json",
        figures=(report_dir / "figure.png",),
        training_rows=80,
        positive_rows=3,
    )
    received: list[tuple[Path, Path, Path]] = []

    def fake_generate(processed: Path, source_manifest: Path, output: Path) -> EDAArtifacts:
        received.append((processed, source_manifest, output))
        return expected

    monkeypatch.setattr("predictive_maintenance.eda.generate_eda", fake_generate)

    exit_code = main(
        [
            "eda",
            "--processed-dir",
            str(processed_dir),
            "--manifest",
            str(manifest),
            "--report-dir",
            str(report_dir),
        ]
    )

    assert exit_code == 0
    assert received == [(processed_dir, manifest, report_dir)]
    assert json.loads(capsys.readouterr().out) == expected.to_dict()


def test_cli_train_delegates_without_holdout_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    manifest = tmp_path / "split.json"
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "artifacts"
    expected = TrainingArtifacts(
        run_id="run",
        selected_model="logistic_regression",
        threshold=0.4,
        pipeline=artifact_dir / "run" / "pipeline.joblib",
        artifact_manifest=artifact_dir / "run" / "artifact_manifest.json",
        run_manifest=report_dir / "run_manifest.json",
        cv_results=report_dir / "cv_results.json",
        threshold_selection=report_dir / "threshold_selection.json",
        selection_report=report_dir / "SELECTION_REPORT.md",
        figures=(report_dir / "figure.png",),
    )
    received: list[tuple[Path, Path, Path, Path]] = []

    def fake_train(processed: Path, source: Path, reports: Path, artifacts: Path):
        received.append((processed, source, reports, artifacts))
        return expected

    monkeypatch.setattr("predictive_maintenance.modeling.run_training_selection", fake_train)
    exit_code = main(
        [
            "train",
            "--processed-dir",
            str(processed_dir),
            "--manifest",
            str(manifest),
            "--report-dir",
            str(report_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    assert exit_code == 0
    assert received == [(processed_dir, manifest, report_dir, artifact_dir)]
    assert json.loads(capsys.readouterr().out) == expected.to_dict()


def test_cli_evaluate_requires_and_forwards_explicit_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    manifest = tmp_path / "split.json"
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "artifacts"
    expected = FinalEvaluationArtifacts(
        run_id="run",
        selected_model="logistic_regression",
        threshold=0.4,
        pipeline=artifact_dir / "run" / "pipeline.joblib",
        evaluation=report_dir / "final_evaluation.json",
        report=report_dir / "M3_REPORT.md",
        confusion_figure=report_dir / "figure.png",
        cached=False,
    )
    confirmations: list[bool] = []

    def fake_evaluate(
        processed: Path,
        source: Path,
        reports: Path,
        artifacts: Path,
        *,
        confirm_final_evaluation: bool,
    ):
        assert (processed, source, reports, artifacts) == (
            processed_dir,
            manifest,
            report_dir,
            artifact_dir,
        )
        confirmations.append(confirm_final_evaluation)
        return expected

    monkeypatch.setattr("predictive_maintenance.evaluation.evaluate_final_holdout", fake_evaluate)
    exit_code = main(
        [
            "evaluate-holdout",
            "--processed-dir",
            str(processed_dir),
            "--manifest",
            str(manifest),
            "--report-dir",
            str(report_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--confirm-final-evaluation",
        ]
    )

    assert exit_code == 0
    assert confirmations == [True]
    assert json.loads(capsys.readouterr().out) == expected.to_dict()
