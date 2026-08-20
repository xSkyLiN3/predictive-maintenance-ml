"use strict";

const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const resultPanel = document.querySelector("#result-panel");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const resultStatus = document.querySelector("#result-status");
const errorSummary = document.querySelector("#form-errors");
const errorList = document.querySelector("#error-list");
const riskScore = document.querySelector("#risk-score");
const scoreRaw = document.querySelector("#score-raw");
const riskMeter = document.querySelector("#risk-meter");
const thresholdMeter = document.querySelector("#threshold-meter");
const thresholdValue = document.querySelector("#threshold-value");
const classificationValue = document.querySelector("#classification-value");
const decisionPill = document.querySelector("#decision-pill");
const warningsBox = document.querySelector("#warnings-box");
const warningsList = document.querySelector("#warnings-list");
const domainNotice = document.querySelector("#domain-notice");
const riskScaleWrap = document.querySelector("#risk-scale-wrap");
const thresholdExplanation = document.querySelector("#threshold-explanation");
let hasRenderedResult = false;

const DOMAIN_WITHIN = "within_reference_envelope";
const DOMAIN_OUTSIDE = "outside_reference_envelope";
const integerFieldNames = new Set(["rotational_speed_rpm", "tool_wear_min"]);
const positiveTemperatureFieldNames = new Set([
  "air_temperature_k",
  "process_temperature_k",
]);

const fieldLabels = {
  type: "Tipo de producto",
  air_temperature_k: "Temperatura del aire",
  process_temperature_k: "Temperatura del proceso",
  rotational_speed_rpm: "Velocidad rotacional",
  torque_nm: "Torque",
  tool_wear_min: "Desgaste de herramienta",
};

const fieldErrorElements = {
  type: document.querySelector("#type-error"),
  air_temperature_k: document.querySelector("#air-temperature-error"),
  process_temperature_k: document.querySelector("#process-temperature-error"),
  rotational_speed_rpm: document.querySelector("#rotational-speed-error"),
  torque_nm: document.querySelector("#torque-error"),
  tool_wear_min: document.querySelector("#tool-wear-error"),
};

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  if (isLoading) {
    submitButton.setAttribute("aria-label", "Calculando score de riesgo");
  } else {
    submitButton.removeAttribute("aria-label");
  }
  resultPanel.setAttribute("aria-busy", String(isLoading));
}

function clearErrors() {
  errorSummary.hidden = true;
  errorList.replaceChildren();
  for (const field of form.elements) {
    if (field instanceof HTMLElement) {
      field.removeAttribute("aria-invalid");
    }
  }

  for (const fieldError of Object.values(fieldErrorElements)) {
    fieldError.textContent = "";
    fieldError.hidden = true;
  }
}

function appendErrorItem(error) {
  const item = document.createElement("li");

  if (error.field && Object.hasOwn(fieldLabels, error.field)) {
    const field = form.elements.namedItem(error.field);
    if (field instanceof HTMLElement) {
      item.dataset.field = error.field;
      const link = document.createElement("a");
      link.href = `#${field.id}`;
      link.textContent = error.message;
      link.addEventListener("click", () => field.focus());
      item.append(link);
      errorList.append(item);
      return;
    }
  }

  item.textContent = error.message;
  errorList.append(item);
}

function setFieldError(name, message) {
  const field = form.elements.namedItem(name);
  const fieldError = fieldErrorElements[name];
  if (!(field instanceof HTMLElement) || !fieldError) {
    return;
  }

  field.setAttribute("aria-invalid", "true");
  fieldError.textContent = message;
  fieldError.hidden = false;
}

function showErrors(errors) {
  errorList.replaceChildren();
  const messagesByField = new Map();

  for (const error of errors) {
    appendErrorItem(error);
    if (error.field && Object.hasOwn(fieldLabels, error.field)) {
      const messages = messagesByField.get(error.field) ?? [];
      messages.push(error.message);
      messagesByField.set(error.field, messages);
    }
  }

  for (const [name, messages] of messagesByField) {
    setFieldError(name, messages.join(" "));
  }

  errorSummary.hidden = false;
  errorSummary.focus({ preventScroll: true });
  errorSummary.scrollIntoView({ block: "center" });
}

function getControlError(control) {
  const label = fieldLabels[control.name] ?? control.name;
  let reason = null;

  if (control.validity.badInput) {
    reason = "debe ser un número válido";
  } else if (control.validity.valueMissing) {
    reason = "es obligatorio";
  } else if (control.validity.rangeUnderflow) {
    reason = `debe ser igual o superior a ${control.min}`;
  } else if (control.validity.rangeOverflow) {
    reason = `debe ser igual o inferior a ${control.max}`;
  } else if (control.validity.stepMismatch) {
    reason = integerFieldNames.has(control.name)
      ? "debe ser un número entero"
      : "debe respetar la precisión indicada";
  } else if (!control.validity.valid) {
    reason = "revisa el valor ingresado";
  } else if (control instanceof HTMLInputElement) {
    const numericValue = Number(control.value);
    if (!Number.isFinite(numericValue)) {
      reason = "debe ser un número finito";
    } else if (positiveTemperatureFieldNames.has(control.name) && numericValue <= 0) {
      reason = "debe ser mayor que 0";
    } else if (integerFieldNames.has(control.name) && !Number.isInteger(numericValue)) {
      reason = "debe ser un número entero";
    } else if (integerFieldNames.has(control.name) && !Number.isSafeInteger(numericValue)) {
      reason = "supera el rango entero seguro admitido por esta interfaz";
    }
  }

  return reason ? { message: `${label}: ${reason}.`, field: control.name } : null;
}

function collectClientErrors() {
  const errors = [];

  for (const control of form.elements) {
    if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement)) {
      continue;
    }

    const error = getControlError(control);
    if (error) {
      errors.push(error);
    }
  }

  return errors;
}

function revalidateInvalidField(control) {
  if (control.getAttribute("aria-invalid") !== "true") {
    return;
  }

  const error = getControlError(control);
  const matchingItems = [...errorList.children].filter(
    (item) => item instanceof HTMLLIElement && item.dataset.field === control.name,
  );

  if (error) {
    setFieldError(control.name, error.message);
    if (matchingItems.length === 0) {
      appendErrorItem(error);
    } else {
      const link = matchingItems[0].querySelector("a");
      if (link) {
        link.textContent = error.message;
      } else {
        matchingItems[0].textContent = error.message;
      }
      for (const duplicate of matchingItems.slice(1)) {
        duplicate.remove();
      }
    }
    return;
  }

  control.removeAttribute("aria-invalid");
  const fieldError = fieldErrorElements[control.name];
  if (fieldError) {
    fieldError.textContent = "";
    fieldError.hidden = true;
  }
  for (const item of matchingItems) {
    item.remove();
  }
  errorSummary.hidden = errorList.children.length === 0;
}

function buildPayload() {
  return {
    type: form.elements.namedItem("type").value,
    air_temperature_k: Number(form.elements.namedItem("air_temperature_k").value),
    process_temperature_k: Number(form.elements.namedItem("process_temperature_k").value),
    rotational_speed_rpm: Number(form.elements.namedItem("rotational_speed_rpm").value),
    torque_nm: Number(form.elements.namedItem("torque_nm").value),
    tool_wear_min: Number(form.elements.namedItem("tool_wear_min").value),
  };
}

function asFiniteUnitInterval(value, fieldName) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`La API devolvió un ${fieldName} inválido.`);
  }
  return value;
}

function normalizeWarnings(value) {
  if (Array.isArray(value)) {
    return value.filter((item) => typeof item === "string" && item.trim()).map((item) => item.trim());
  }
  if (typeof value === "string" && value.trim()) {
    return [value.trim()];
  }
  return [];
}

function renderWarnings(value, isOutsideDomain) {
  const warnings = normalizeWarnings(value);
  if (isOutsideDomain && warnings.length === 0) {
    warnings.push(
      "La observación queda fuera de la envolvente de referencia obtenida del training AI4I.",
      "No se aplica el umbral ni se emite una clasificación.",
    );
  }

  warningsList.replaceChildren();
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = warning;
    warningsList.append(item);
  }
  warningsBox.hidden = warnings.length === 0;
}

function renderResult(response) {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    throw new Error("La API devolvió un resultado inválido.");
  }

  if (response.domain_status !== DOMAIN_WITHIN && response.domain_status !== DOMAIN_OUTSIDE) {
    throw new Error("La API devolvió un estado de dominio inválido.");
  }
  if (typeof response.decision_applicable !== "boolean") {
    throw new Error("La API devolvió un estado de decisión inválido.");
  }

  const threshold = asFiniteUnitInterval(response.threshold, "umbral");
  const isOutsideDomain = response.domain_status === DOMAIN_OUTSIDE;
  const decimalFormat = new Intl.NumberFormat("es-CL", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });

  if (isOutsideDomain) {
    if (
      response.decision_applicable !== false ||
      response.risk_score !== null ||
      response.predicted_failure !== null
    ) {
      throw new Error("La API devolvió un resultado fuera de dominio inconsistente.");
    }

    domainNotice.hidden = false;
    riskScaleWrap.hidden = true;
    riskScore.textContent = "No disponible";
    riskScore.classList.add("is-unavailable");
    scoreRaw.textContent = "No se calcula fuera de la envolvente de referencia.";
    thresholdValue.textContent = "No aplicable";
    classificationValue.textContent = "No aplicable";
    decisionPill.textContent = "Sin decisión";
    decisionPill.className = "decision-pill is-outside";
    thresholdExplanation.textContent =
      "No se aplica el umbral ni se emite una clasificación fuera de la envolvente de referencia.";
    resultStatus.textContent = "Fuera del dominio observado";
    resultStatus.className = "result-status result-status-outside";
    resultPanel.classList.add("is-outside");
    renderWarnings(response.warnings ?? response.warning, true);
  } else {
    if (response.decision_applicable !== true) {
      throw new Error("La API devolvió una decisión no aplicable dentro del dominio.");
    }

    const score = asFiniteUnitInterval(response.risk_score, "score de riesgo");
    if (typeof response.predicted_failure !== "boolean") {
      throw new Error("La API devolvió una clasificación inválida.");
    }

    const predictedFailure = response.predicted_failure;
    domainNotice.hidden = true;
    riskScaleWrap.hidden = false;
    riskScore.textContent = decimalFormat.format(score);
    riskScore.classList.remove("is-unavailable");
    scoreRaw.textContent = "escala 0–1 · no evaluada como probabilidad calibrada";
    thresholdValue.textContent = decimalFormat.format(threshold);
    classificationValue.textContent = predictedFailure ? "Fallo clasificado" : "Sin fallo clasificado";
    decisionPill.textContent = predictedFailure ? "Sobre el umbral" : "Bajo el umbral";
    decisionPill.className = `decision-pill ${predictedFailure ? "is-positive" : "is-negative"}`;
    riskMeter.value = score;
    riskMeter.textContent = decimalFormat.format(score);
    riskMeter.setAttribute("aria-label", `Score de riesgo: ${decimalFormat.format(score)}`);
    riskMeter.classList.toggle("is-positive", predictedFailure);
    thresholdMeter.value = threshold;
    thresholdMeter.textContent = decimalFormat.format(threshold);
    thresholdMeter.setAttribute("aria-label", `Umbral de decisión: ${decimalFormat.format(threshold)}`);
    thresholdExplanation.textContent =
      "La clasificación es positiva cuando el score es igual o superior al umbral. Este resultado describe una observación y no un diagnóstico mecánico.";
    resultStatus.textContent = predictedFailure ? "Clasificación positiva" : "Clasificación negativa";
    resultStatus.className = `result-status ${
      predictedFailure ? "result-status-positive" : "result-status-negative"
    }`;
    resultPanel.classList.remove("is-outside");
    renderWarnings(response.warnings ?? response.warning, false);
  }
  resultPanel.classList.remove("is-stale");
  resultEmpty.hidden = true;
  resultContent.hidden = false;
  hasRenderedResult = true;
}

function markResultAsStale() {
  if (!hasRenderedResult) {
    return;
  }
  resultPanel.classList.add("is-stale");
  resultStatus.textContent = "Entrada modificada";
  resultStatus.className = "result-status result-status-stale";
}

function formatApiErrors(payload, status) {
  if (status >= 500) {
    return [{ message: "La API no pudo completar la solicitud. Inténtalo de nuevo." }];
  }

  const details = payload && Array.isArray(payload.detail) ? payload.detail : [];
  if (details.length > 0) {
    const errors = [];

    for (const detail of details) {
      const path = Array.isArray(detail.loc) ? detail.loc : [];
      const field = path.at(-1);
      const label = typeof field === "string" ? (fieldLabels[field] ?? field) : "Entrada";
      const reason = translateValidationReason(detail);
      errors.push({
        message: `${label}: ${reason}.`,
        field: typeof field === "string" && Object.hasOwn(fieldLabels, field) ? field : undefined,
      });
    }

    return errors;
  }

  if (payload && typeof payload.detail === "string") {
    return [{ message: payload.detail }];
  }

  return [{ message: `La API respondió con un error (${status}). Revisa los valores e inténtalo otra vez.` }];
}

function translateValidationReason(detail) {
  const context = detail && typeof detail.ctx === "object" && detail.ctx ? detail.ctx : {};
  const validationType = typeof detail.type === "string" ? detail.type : "";

  if (validationType === "missing") {
    return "es obligatorio";
  }
  if (validationType === "literal_error") {
    return "debe ser una categoría admitida";
  }
  if (validationType === "greater_than_equal") {
    return `debe ser igual o superior a ${context.ge ?? "el mínimo admitido"}`;
  }
  if (validationType === "greater_than") {
    return `debe ser superior a ${context.gt ?? "el mínimo admitido"}`;
  }
  if (validationType === "less_than_equal") {
    return `debe ser igual o inferior a ${context.le ?? "el máximo admitido"}`;
  }
  if (validationType === "less_than") {
    return `debe ser inferior a ${context.lt ?? "el máximo admitido"}`;
  }
  if (validationType === "int_type" || validationType === "int_parsing") {
    return "debe ser un número entero";
  }
  if (
    validationType === "float_type" ||
    validationType === "float_parsing" ||
    validationType === "finite_number"
  ) {
    return "debe ser un número finito";
  }
  if (validationType === "extra_forbidden") {
    return "no pertenece al contrato de entrada";
  }
  return "contiene un valor inválido";
}

async function parseResponseBody(response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    throw new Error("La API devolvió una respuesta inesperada.");
  }
  return response.json();
}

form.addEventListener("input", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
    revalidateInvalidField(event.target);
  }
  markResultAsStale();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  markResultAsStale();
  clearErrors();

  const clientErrors = collectClientErrors();
  if (clientErrors.length > 0) {
    showErrors(clientErrors);
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildPayload()),
    });

    if (response.status >= 500) {
      showErrors(formatApiErrors(null, response.status));
      return;
    }

    const payload = await parseResponseBody(response);

    if (!response.ok) {
      showErrors(formatApiErrors(payload, response.status));
      return;
    }

    renderResult(payload);
  } catch (error) {
    const message =
      error instanceof TypeError
        ? "No se pudo conectar con la API. Confirma que la aplicación esté en ejecución."
        : error instanceof SyntaxError
          ? "La API devolvió JSON no válido."
        : error instanceof Error
          ? error.message
          : "Ocurrió un error inesperado al calcular el score.";
    showErrors([{ message }]);
  } finally {
    setLoading(false);
  }
});
