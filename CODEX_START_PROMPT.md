# Prompt para continuar el proyecto en una tarea nueva de Codex

Lee completamente `AGENTS.md`, `README.md`, `docs/MODEL_CARD.md`,
`docs/PORTFOLIO_REVIEW.md` y el resto de la documentación enlazada antes de modificar archivos.

M0–M5 están completados y el MVP local quedó cerrado. M6 de publicación y demo fue autorizado
expresamente por el usuario el 2026-08-19. El resultado congelado es el run
`b15bab7b54bc2e1f`, con random forest y umbral `0.6965799216184142`. No abras de nuevo
`holdout.csv`, no recalcules métricas finales y no cambies modelo, features, target o umbral.
La API 1.0 se abstiene de puntuar fuera de la envolvente marginal de training documentada;
conserva esa distinción entre validación semántica, aplicabilidad educativa y límites físicos.

Conserva estas fronteras:

1. La licencia del código es MIT y no sustituye la `CC BY 4.0` declarada para AI4I.
2. GitHub, CI, un contenedor mínimo y una demo pública stateless están dentro de M6.
3. No publiques hasta que el snapshot staged pase pruebas y auditoría de secretos/dependencias.
4. No añadas autenticación, base de datos, telemetría, tracking de experimentos, LLM, GPU ni
   infraestructura distribuida salvo que el usuario cambie expresamente el alcance.
5. No describas la demo educativa como un sistema de producción ni como validación industrial.

Al completar M6, verifica y actualiza `TASKS.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md` y esta guía.
No inventes resultados ni describas AI4I como validación industrial.
