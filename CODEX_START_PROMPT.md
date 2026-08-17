# Prompt para iniciar el proyecto en una tarea nueva de Codex

Lee completamente `AGENTS.md` y toda la documentación enlazada desde `README.md` antes de modificar archivos.

Trabaja exclusivamente en local y ejecuta solo los hitos M0 y M1 de `docs/ROADMAP.md`. En concreto:

1. Valida el entorno disponible y usa explícitamente Python 3.12.
2. Inicializa `.venv` sin modificar el Python global.
3. Crea un `pyproject.toml` mínimo y reproducible con estructura `src/`, pytest y Ruff.
4. Implementa la descarga programática del dataset UCI AI4I 2020 desde una fuente oficial.
5. Conserva los datos descargados fuera de Git y registra su procedencia y checksum.
6. Implementa validación de esquema, target, categorías y columnas prohibidas por leakage.
7. Añade pruebas para descarga/lectura, validación y exclusión de columnas.
8. Ejecuta Ruff y pytest, corrige los fallos y actualiza `TASKS.md` y `docs/DECISIONS.md`.

No hagas todavía EDA extensa, entrenamiento de modelos, API, interfaz, Docker, despliegue, GitHub ni cambios en el VPS.

Al terminar, detente y entrega un informe breve con:

- estructura creada;
- dependencias añadidas y justificación;
- comandos exactos para reproducir el entorno y las pruebas;
- origen, tamaño y checksum del dataset;
- resultado de Ruff y pytest;
- riesgos o decisiones pendientes antes de comenzar M2.
