# Machine Failure Risk Classifier

Proyecto local de portfolio para aprender y demostrar un flujo completo y reproducible de machine learning con datos tabulares.

## Estado

**Fase actual: propuesta y preparación local.**

Todavía no existe un modelo entrenado, una demo pública ni un despliegue. El repositorio se mantendrá local hasta superar los criterios de calidad descritos en la documentación.

## Qué problema resolverá

El sistema estimará la probabilidad de `Machine failure` para una observación de operación utilizando el dataset sintético **AI4I 2020 Predictive Maintenance** de UCI.

No pretende predecir cuánto tiempo falta para una avería, analizar una secuencia temporal ni acreditar utilidad industrial real. Su objetivo es demostrar fundamentos de ingeniería de ML:

- descarga y validación reproducible de datos;
- prevención de fuga de información;
- baseline y comparación de modelos;
- evaluación apropiada para clases desbalanceadas;
- API e interfaz local;
- pruebas, documentación y reproducibilidad.

## Documentación del proyecto

- [Propuesta](docs/PROJECT_PROPOSAL.md)
- [Contrato de datos y evaluación](docs/DATA_EVALUATION.md)
- [Hoja de ruta](docs/ROADMAP.md)
- [Registro de decisiones](docs/DECISIONS.md)
- [Tareas](TASKS.md)
- [Prompt inicial para Codex](CODEX_START_PROMPT.md)

## Cómo comenzar en Codex

1. Abrir esta carpeta como un proyecto independiente.
2. Iniciar una tarea nueva.
3. Copiar el contenido de `CODEX_START_PROMPT.md`.
4. Revisar el resultado del primer hito antes de avanzar al entrenamiento.

Codex leerá automáticamente `AGENTS.md` desde la raíz del proyecto y aplicará sus reglas de alcance y calidad.

## Tecnología prevista

- Python 3.12
- scikit-learn
- pandas
- FastAPI, en una fase posterior
- pytest y Ruff
- CPU; no se necesita GPU

## Fuente de datos

UCI Machine Learning Repository, dataset 601:  
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maint

Licencia del dataset: CC BY 4.0. La atribución exacta se incorporará al README definitivo y a la ficha del modelo.
