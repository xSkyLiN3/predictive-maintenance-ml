"""Command-line entry points for reproducible local data milestones."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from predictive_maintenance.artifact_io import ArtifactIntegrityError
from predictive_maintenance.config import (
    DEFAULT_EDA_REPORT_DIR,
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
)
from predictive_maintenance.dataset import (
    DEFAULT_RAW_DATA_DIR,
    DatasetIntegrityError,
    retrieve_dataset,
    verify_dataset_files,
)
from predictive_maintenance.splitting import SplitIntegrityError, materialize_split
from predictive_maintenance.validation import DataValidationError, read_dataset, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the data-management CLI parser."""
    parser = argparse.ArgumentParser(
        prog="machine-failure-data",
        description="Retrieve, partition and analyze the fixed AI4I 2020 dataset snapshot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("download", "Download once, verify checksums, extract and validate the dataset."),
        ("validate", "Validate already-downloaded files without network access."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument(
            "--raw-dir",
            type=Path,
            default=DEFAULT_RAW_DATA_DIR,
            help=f"Raw-data directory (default: {DEFAULT_RAW_DATA_DIR}).",
        )

    split_parser = subparsers.add_parser(
        "split",
        help="Materialize the fixed stratified train/holdout partition.",
    )
    split_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    split_parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DATA_DIR)
    split_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
    )

    eda_parser = subparsers.add_parser(
        "eda",
        help="Generate the training-only M2 report and figures.",
    )
    eda_parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DATA_DIR)
    eda_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
    )
    eda_parser.add_argument("--report-dir", type=Path, default=DEFAULT_EDA_REPORT_DIR)

    train_parser = subparsers.add_parser(
        "train",
        help="Run training-only CV selection, OOF thresholding and final fitting.",
    )
    train_parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DATA_DIR)
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    train_parser.add_argument("--report-dir", type=Path, default=DEFAULT_MODEL_REPORT_DIR)
    train_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_MODEL_ARTIFACT_DIR)

    evaluate_parser = subparsers.add_parser(
        "evaluate-holdout",
        help="Consume the holdout once for the final selected-model evaluation.",
    )
    evaluate_parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DATA_DIR)
    evaluate_parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    evaluate_parser.add_argument("--report-dir", type=Path, default=DEFAULT_MODEL_REPORT_DIR)
    evaluate_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_MODEL_ARTIFACT_DIR)
    evaluate_parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        required=True,
        help="Confirm the single final read of the sealed holdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a data command and return a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "download":
            paths = retrieve_dataset(arguments.raw_dir)
            frame = read_dataset(paths.csv)
            summary = validate_dataset(frame)
            output = {
                "archive": str(paths.archive),
                "csv": str(paths.csv),
                "metadata": str(paths.metadata),
                "validation": summary.to_dict(),
            }
        elif arguments.command == "validate":
            paths = verify_dataset_files(arguments.raw_dir)
            frame = read_dataset(paths.csv)
            summary = validate_dataset(frame)
            output = {
                "archive": str(paths.archive),
                "csv": str(paths.csv),
                "metadata": str(paths.metadata),
                "validation": summary.to_dict(),
            }
        elif arguments.command == "split":
            split_artifacts = materialize_split(
                arguments.raw_dir,
                arguments.processed_dir,
                arguments.manifest,
            )
            output = split_artifacts.to_dict()
        elif arguments.command == "eda":
            from predictive_maintenance.eda import generate_eda

            eda_artifacts = generate_eda(
                arguments.processed_dir,
                arguments.manifest,
                arguments.report_dir,
            )
            output = eda_artifacts.to_dict()
        elif arguments.command == "train":
            from predictive_maintenance.modeling import run_training_selection

            training_artifacts = run_training_selection(
                arguments.processed_dir,
                arguments.manifest,
                arguments.report_dir,
                arguments.artifact_dir,
            )
            output = training_artifacts.to_dict()
        elif arguments.command == "evaluate-holdout":
            from predictive_maintenance.evaluation import evaluate_final_holdout

            evaluation_artifacts = evaluate_final_holdout(
                arguments.processed_dir,
                arguments.manifest,
                arguments.report_dir,
                arguments.artifact_dir,
                confirm_final_evaluation=arguments.confirm_final_evaluation,
            )
            output = evaluation_artifacts.to_dict()
        else:  # pragma: no cover - argparse constrains this value.
            parser.error(f"Unsupported command: {arguments.command}")
    except (
        ArtifactIntegrityError,
        DataValidationError,
        DatasetIntegrityError,
        SplitIntegrityError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        from predictive_maintenance.evaluation import FinalEvaluationError
        from predictive_maintenance.modeling import ModelingError

        if not isinstance(error, (FinalEvaluationError, ModelingError)):
            raise
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0
