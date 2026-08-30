"""Regression gate for English text on editable public surfaces."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SURFACES = (
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("artifacts/README.md"),
    Path("data/README.md"),
    Path("notebooks/README.md"),
    Path("reports/README.md"),
    Path("reports/eda/EDA_REPORT.md"),
    Path("src/predictive_maintenance/web/index.html"),
    Path("src/predictive_maintenance/web/app.js"),
)

# These project and statistical names are valid in English output and are removed before checks.
ALLOWED_PROPER_NAMES = ("AI4I", "Wilson", "Spearman", "Cristóbal")
SPANISH_DIACRITICS = re.compile("[áéíóúüñ¿¡]", re.IGNORECASE)
SPANISH_PUBLIC_WORDS = frozenset(
    {
        "abrir",
        "aire",
        "alcance",
        "algoritmos",
        "analizadas",
        "analizado",
        "candidato",
        "candidatos",
        "clasificación",
        "clasificado",
        "celdas",
        "clase",
        "congelado",
        "consultar",
        "conteos",
        "datos",
        "decisión",
        "desgaste",
        "entrenados",
        "entrenamiento",
        "evaluación",
        "fallo",
        "filas",
        "fuente",
        "fuera",
        "informe",
        "matriz",
        "mediana",
        "modelo",
        "modelos",
        "negativos",
        "observaciones",
        "operacional",
        "positivos",
        "prevalencia",
        "protocolo",
        "puerta",
        "resultados",
        "riesgos",
        "sintético",
        "sintéticos",
        "semilla",
        "soporte",
        "tasa",
        "temperaturas",
        "validación",
        "umbral",
        "velocidad",
        "visualizaciones",
    }
)
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _public_text_paths() -> tuple[Path, ...]:
    paths = {PROJECT_ROOT / path for path in PUBLIC_TEXT_SURFACES}
    paths.update((PROJECT_ROOT / "docs").glob("*.md"))
    paths.update((PROJECT_ROOT / "src/predictive_maintenance").glob("*.py"))
    return tuple(sorted(paths))


def test_editable_public_surfaces_are_english_only() -> None:
    """Reject Spanish text without inspecting sealed reports or generated JSON receipts."""
    violations = []
    for path in _public_text_paths():
        relative_path = path.relative_to(PROJECT_ROOT)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            checked_line = line
            for proper_name in ALLOWED_PROPER_NAMES:
                checked_line = checked_line.replace(proper_name, "")
            spanish_words = sorted(
                set(WORD_PATTERN.findall(checked_line.casefold())) & SPANISH_PUBLIC_WORDS
            )
            if not SPANISH_DIACRITICS.search(checked_line) and not spanish_words:
                continue
            preview = " ".join(line.split())[:120]
            violations.append(
                f"{relative_path.as_posix()}:{line_number}: {preview!r}"
                f" (Spanish words: {spanish_words})"
            )

    assert not violations, "Spanish text found in editable public surfaces:\n" + "\n".join(
        violations
    )
