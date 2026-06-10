from __future__ import annotations

import html
import json
import os
import io
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ICU_Patient_Monitoring_Mortality_Prediction_15000 (1).csv"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
STYLE_PATH = BASE_DIR / "static" / "style.css"
BASE_PATH = os.environ.get("ICU_APP_BASE_PATH", "/icu-risk")

CATEGORICAL_FIELDS = ["gender", "admission_type"]
TARGET_COLUMN = "mortality_label"
IDENTITY_FIELDS = ["name"]
REQUIRED_FIELDS = {
    "name",
    "age",
    "gender",
    "admission_type",
    "comorbidity_score",
    "heart_rate_mean",
    "spo2_mean",
}

FIELD_GROUPS = [
    (
        "Assessment Details",
        [
            "name",
            "age",
            "gender",
            "admission_type",
            "comorbidity_score",
            "heart_rate_mean",
            "spo2_mean",
        ],
    )
]

FIELD_META = {
    "name": {
        "label": "Patient Name",
        "type": "text",
        "placeholder": "e.g. Jordan Lee",
        "hint": "Used only for the local assessment view.",
    },
    "age": {"label": "Age", "type": "number", "step": "1", "unit": "years"},
    "gender": {
        "label": "Gender",
        "type": "select",
        "options": [("Female", "Female"), ("Male", "Male")],
    },
    "admission_type": {
        "label": "Admission Type",
        "type": "select",
        "options": [
            ("Emergency", "Emergency"),
            ("Urgent", "Urgent"),
            ("Elective", "Elective"),
        ],
    },
    "comorbidity_score": {
        "label": "Comorbidity Score",
        "type": "number",
        "step": "0.01",
        "unit": "/10",
    },
    "heart_rate_mean": {
        "label": "Mean Heart Rate",
        "type": "number",
        "step": "0.01",
        "unit": "bpm",
    },
    "heart_rate_std": {
        "label": "Heart Rate Variability",
        "type": "number",
        "step": "0.01",
        "unit": "std",
    },
    "heart_rate_max": {
        "label": "Max Heart Rate",
        "type": "number",
        "step": "0.01",
        "unit": "bpm",
    },
    "heart_rate_min": {
        "label": "Min Heart Rate",
        "type": "number",
        "step": "0.01",
        "unit": "bpm",
    },
    "systolic_bp_mean": {
        "label": "Mean Systolic BP",
        "type": "number",
        "step": "0.01",
        "unit": "mmHg",
    },
    "systolic_bp_std": {
        "label": "Systolic BP Variability",
        "type": "number",
        "step": "0.01",
        "unit": "std",
    },
    "respiratory_rate_mean": {
        "label": "Respiratory Rate",
        "type": "number",
        "step": "0.01",
        "unit": "breaths/min",
    },
    "spo2_mean": {
        "label": "Mean SpO2",
        "type": "number",
        "step": "0.01",
        "unit": "%",
    },
    "temperature_mean": {
        "label": "Temperature",
        "type": "number",
        "step": "0.01",
        "unit": "C",
    },
    "glucose_mean": {
        "label": "Glucose",
        "type": "number",
        "step": "0.01",
        "unit": "mg/dL",
    },
    "lactate_mean": {
        "label": "Lactate",
        "type": "number",
        "step": "0.01",
        "unit": "mmol/L",
    },
    "urine_output_total": {
        "label": "Urine Output",
        "type": "number",
        "step": "0.01",
        "unit": "mL",
    },
    "ventilation_required": {
        "label": "Ventilation Required",
        "type": "select",
        "options": [("0", "No"), ("1", "Yes")],
    },
    "vasopressor_used": {
        "label": "Vasopressor Used",
        "type": "select",
        "options": [("0", "No"), ("1", "Yes")],
    },
    "length_of_stay_days": {
        "label": "Length Of Stay",
        "type": "number",
        "step": "0.01",
        "unit": "days",
    },
    "apache_score": {
        "label": "APACHE Score",
        "type": "number",
        "step": "0.01",
        "unit": "severity",
    },
    "sofa_score": {
        "label": "SOFA Score",
        "type": "number",
        "step": "0.01",
        "unit": "severity",
    },
    "sepsis_flag": {
        "label": "Sepsis Flag",
        "type": "select",
        "options": [("0", "No"), ("1", "Yes")],
    },
}

DISPLAY_NAMES = {
    "gender_Male": "Male sex profile",
    "admission_type_Emergency": "Emergency admission",
    "admission_type_Urgent": "Urgent admission",
    "hr_spo2_ratio": "Heart rate to oxygen mismatch",
    "bp_hr_ratio": "Blood pressure to heart rate ratio",
    "hr_range": "Heart rate range",
    "tachycardia_flag": "Tachycardia threshold crossed",
    "hypoxia_flag": "Hypoxia threshold crossed",
    "severity_index": "Combined severity burden",
}


@dataclass
class ModelArtifacts:
    feature_columns: list[str]
    feature_weights: np.ndarray
    intercept: float
    means: np.ndarray
    scales: np.ndarray
    numeric_medians: dict[str, float]
    categorical_defaults: dict[str, str]
    binary_defaults: dict[str, str]
    field_ranges: dict[str, tuple[float, float]]
    metrics: dict[str, float]
    dataset_summary: dict[str, float]
    threshold: float


def h(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def normalize_base_path(value: str) -> str:
    cleaned = (value or "/").strip()
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    cleaned = cleaned.rstrip("/")
    return cleaned or "/"


def route_path(path: str) -> str:
    base = normalize_base_path(BASE_PATH)
    suffix = path if path.startswith("/") else f"/{path}"
    if base == "/":
        return suffix
    return f"{base}{suffix}"


def format_number(value: float, digits: int = 1) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hr_spo2_ratio"] = enriched["heart_rate_mean"] / (enriched["spo2_mean"] + 1.0)
    enriched["bp_hr_ratio"] = enriched["systolic_bp_mean"] / (enriched["heart_rate_mean"] + 1.0)
    enriched["hr_range"] = enriched["heart_rate_max"] - enriched["heart_rate_min"]
    enriched["tachycardia_flag"] = (enriched["heart_rate_mean"] > 100).astype(int)
    enriched["hypoxia_flag"] = (enriched["spo2_mean"] < 92).astype(int)
    enriched["severity_index"] = (enriched["apache_score"] + enriched["sofa_score"]) / 2.0
    return enriched


def prepare_feature_frame(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(frame, columns=CATEGORICAL_FIELDS, drop_first=True, dtype=float)
    if feature_columns is None:
        return encoded
    return encoded.reindex(columns=feature_columns, fill_value=0.0)


def stratified_split(target: np.ndarray, test_ratio: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in np.unique(target):
        indices = np.where(target == label)[0]
        rng.shuffle(indices)
        cutoff = int(len(indices) * (1 - test_ratio))
        train_parts.append(indices[:cutoff])
        test_parts.append(indices[cutoff:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def train_weighted_logistic_regression(
    features: np.ndarray,
    target: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 1600,
    regularization: float = 0.002,
) -> np.ndarray:
    weights = np.zeros(features.shape[1], dtype=float)
    positives = max(float(target.sum()), 1.0)
    pos_weight = (len(target) - positives) / positives
    sample_weights = np.where(target == 1.0, pos_weight, 1.0)

    for _ in range(epochs):
        predictions = sigmoid(features @ weights)
        error = (predictions - target) * sample_weights
        gradient = (features.T @ error) / len(target)
        gradient[1:] += regularization * weights[1:]
        weights -= learning_rate * gradient
    return weights


def compute_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    positive_mask = y_true == 1
    positives = float(positive_mask.sum())
    negatives = float((~positive_mask).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    return float((ranks[positive_mask].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": compute_auc(y_true, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.25, 0.75, 51):
        metrics = classification_metrics(y_true, probabilities, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def mode_or_default(series: pd.Series, default: str) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return default
    mode_values = non_null.mode()
    if mode_values.empty:
        return default
    return str(mode_values.iloc[0])


def build_model() -> ModelArtifacts:
    dataset = load_training_data()
    enriched = engineer_features(dataset.copy())
    feature_frame = prepare_feature_frame(
        enriched.drop(columns=[TARGET_COLUMN, "patient_id"], errors="ignore"),
    )
    target = enriched[TARGET_COLUMN].to_numpy(dtype=float)

    train_idx, test_idx = stratified_split(target)
    x_train_df = feature_frame.iloc[train_idx].copy()
    x_test_df = feature_frame.iloc[test_idx].copy()
    y_train = target[train_idx]
    y_test = target[test_idx]

    means = x_train_df.mean().to_numpy(dtype=float)
    scales = x_train_df.std().replace(0, 1).to_numpy(dtype=float)
    x_train = (x_train_df.to_numpy(dtype=float) - means) / scales
    x_test = (x_test_df.to_numpy(dtype=float) - means) / scales
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

    weights = train_weighted_logistic_regression(x_train, y_train)
    train_probs = sigmoid(x_train @ weights)
    threshold = select_threshold(y_train, train_probs)
    metrics = classification_metrics(y_test, sigmoid(x_test @ weights), threshold)

    numeric_columns = [
        field
        for field, meta in FIELD_META.items()
        if meta["type"] == "number" and field not in IDENTITY_FIELDS
    ]
    numeric_medians: dict[str, float] = {}
    field_ranges: dict[str, tuple[float, float]] = {}
    for column in numeric_columns:
        if column in dataset.columns and not dataset[column].dropna().empty:
            numeric_medians[column] = float(dataset[column].median())
            field_ranges[column] = (
                float(dataset[column].min()),
                float(dataset[column].max()),
            )
        else:
            numeric_medians[column] = 0.0
            field_ranges[column] = (-1_000_000.0, 1_000_000.0)

    categorical_defaults = {
        field: mode_or_default(
            dataset[field] if field in dataset.columns else pd.Series(dtype=object),
            FIELD_META[field]["options"][0][0],
        )
        for field in CATEGORICAL_FIELDS
    }
    binary_defaults = {
        field: mode_or_default(
            dataset[field] if field in dataset.columns else pd.Series(dtype=object),
            "0",
        )
        for field in ["ventilation_required", "vasopressor_used", "sepsis_flag"]
    }
    dataset_summary = {
        "records": float(len(dataset)),
        "mortality_rate": float(dataset[TARGET_COLUMN].mean()),
        "median_age": float(dataset["age"].median()),
        "median_apache": float(dataset["apache_score"].median()),
        "median_sofa": float(dataset["sofa_score"].median()),
    }

    return ModelArtifacts(
        feature_columns=list(feature_frame.columns),
        feature_weights=weights[1:],
        intercept=float(weights[0]),
        means=means,
        scales=scales,
        numeric_medians=numeric_medians,
        categorical_defaults=categorical_defaults,
        binary_defaults=binary_defaults,
        field_ranges=field_ranges,
        metrics=metrics,
        dataset_summary=dataset_summary,
        threshold=threshold,
    )


def risk_band(probability: float) -> dict[str, str]:
    if probability >= 0.7:
        return {
            "label": "Critical Watch",
            "class_name": "critical",
            "summary": "Very elevated mortality signal. Treat this as an urgent review candidate.",
        }
    if probability >= 0.55:
        return {
            "label": "High Risk",
            "class_name": "high",
            "summary": "Elevated mortality signal. Close review and escalation planning are appropriate.",
        }
    if probability >= 0.3:
        return {
            "label": "Moderate Risk",
            "class_name": "moderate",
            "summary": "Intermediate mortality signal. Trend closely with bedside context and labs.",
        }
    return {
        "label": "Low Risk",
        "class_name": "low",
        "summary": "Lower mortality signal in this cohort-based model, though monitoring still matters.",
    }


def contributor_label(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    if name in FIELD_META:
        return FIELD_META[name]["label"]
    return name.replace("_", " ").title()


def parse_request_values(body: str) -> dict[str, str]:
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0].strip() for key, values in parsed.items()}


def parse_patient_input(values: dict[str, str], model: ModelArtifacts) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    autofilled: list[str] = []
    patient: dict[str, object] = {}

    for field, meta in FIELD_META.items():
        raw_value = values.get(field, "").strip()
        if meta["type"] == "text":
            if field in REQUIRED_FIELDS and not raw_value:
                errors.append(f"{meta['label']} is required.")
            patient[field] = raw_value
            continue

        if meta["type"] == "select":
            options = {option for option, _ in meta["options"]}
            default_value = model.categorical_defaults.get(field) or model.binary_defaults.get(field)
            chosen = raw_value or default_value
            if chosen not in options:
                errors.append(f"{meta['label']} has an invalid value.")
                continue
            if not raw_value and field not in REQUIRED_FIELDS:
                autofilled.append(meta["label"])
            patient[field] = chosen
            continue

        if raw_value == "":
            if field in REQUIRED_FIELDS:
                errors.append(f"{meta['label']} is required.")
                continue
            patient[field] = model.numeric_medians[field]
            autofilled.append(meta["label"])
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError:
            errors.append(f"{meta['label']} must be numeric.")
            continue

        minimum, maximum = model.field_ranges[field]
        if numeric_value < minimum or numeric_value > maximum:
            errors.append(
                f"{meta['label']} should stay within the cohort range of "
                f"{format_number(minimum, 2)} to {format_number(maximum, 2)}."
            )
            continue
        patient[field] = numeric_value

    patient["name"] = values.get("name", "").strip()
    return patient, errors, autofilled


def make_feature_vector(patient: dict[str, object], model: ModelArtifacts) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in patient.items()
                if key not in IDENTITY_FIELDS
            }
        ]
    )
    for field in ["ventilation_required", "vasopressor_used", "sepsis_flag"]:
        frame[field] = frame[field].astype(int)
    enriched = engineer_features(frame)
    encoded = prepare_feature_frame(enriched, model.feature_columns)
    scaled = (encoded.to_numpy(dtype=float) - model.means) / model.scales
    return encoded, scaled[0]


def render_form_field(field: str, values: dict[str, str], model: ModelArtifacts) -> str:
    meta = FIELD_META[field]
    label = meta["label"]
    required = "required" if field in REQUIRED_FIELDS else ""
    helper = meta.get("hint", "")
    status = "Required" if field in REQUIRED_FIELDS else "Optional"
    field_header = (
        "<span class='field-header'>"
        f"<span class='field-label'>{h(label)}</span>"
        f"<span class='field-tag'>{h(status)}</span>"
        "</span>"
    )
    if meta["type"] == "text":
        return (
            f"<label class='field'>"
            f"{field_header}"
            f"<input type='text' name='{h(field)}' placeholder='{h(meta.get('placeholder', ''))}' "
            f"value='{h(values.get(field, ''))}' {required}>"
            f"{f'<small>{h(helper)}</small>' if helper else ''}"
            f"</label>"
        )

    if meta["type"] == "select":
        current = values.get(field, "")
        options = []
        for option_value, option_label in meta["options"]:
            selected = "selected" if current == option_value else ""
            options.append(
                f"<option value='{h(option_value)}' {selected}>{h(option_label)}</option>"
            )
        if not current and field not in REQUIRED_FIELDS:
            placeholder = "<option value='' selected>Use cohort default</option>"
            options.insert(0, placeholder)
        return (
            f"<label class='field'>"
            f"{field_header}"
            f"<select name='{h(field)}' {required}>{''.join(options)}</select>"
            f"{f'<small>{h(helper)}</small>' if helper else ''}"
            f"</label>"
        )

    current_value = values.get(field, "")
    minimum, maximum = model.field_ranges[field]
    placeholder = f"Median {format_number(model.numeric_medians[field], 2)}"
    unit = meta.get("unit", "")
    hints: list[str] = []
    range_hint = f"Range {format_number(minimum, 2)} to {format_number(maximum, 2)}"
    if unit:
        range_hint = f"{range_hint} {unit}"
    hints.append(range_hint)
    if field not in REQUIRED_FIELDS:
        hints.append("Blank uses cohort median")

    return (
        f"<label class='field'>"
        f"{field_header}"
        f"<input type='number' name='{h(field)}' step='{h(meta.get('step', '0.01'))}' "
        f"min='{h(format_number(minimum, 2))}' max='{h(format_number(maximum, 2))}' "
        f"placeholder='{h(placeholder)}' value='{h(current_value)}' {required}>"
        f"<small>{h(' | '.join(hints))}</small>"
        f"</label>"
    )


def render_form_sections(values: dict[str, str], model: ModelArtifacts) -> str:
    sections = []
    for _, (title, fields) in enumerate(FIELD_GROUPS):
        fields_html = "".join(render_form_field(field, values, model) for field in fields)
        sections.append(
            f"<section class='section-card'>"
            f"<div class='section-heading'>"
            f"<p class='section-kicker'>Core Inputs</p>"
            f"<h3>{h(title)}</h3>"
            f"<p>Capture the patient profile used by the bedside assessment model.</p>"
            f"</div>"
            f"<div class='field-grid'>{fields_html}</div>"
            f"</section>"
        )
    return "".join(sections)


def build_alerts(patient: dict[str, object]) -> list[str]:
    alerts = []
    if float(patient["spo2_mean"]) < 92:
        alerts.append("SpO2 is below the hypoxia threshold.")
    if float(patient["lactate_mean"]) > 4:
        alerts.append("Lactate is elevated and may reflect perfusion stress.")
    if float(patient["apache_score"]) > 30 or float(patient["sofa_score"]) > 12:
        alerts.append("Severity scores are in a high-burden range.")
    if int(patient["ventilation_required"]) == 1:
        alerts.append("Ventilatory support is already required.")
    if int(patient["vasopressor_used"]) == 1:
        alerts.append("Vasopressor therapy is active.")
    if int(patient["sepsis_flag"]) == 1:
        alerts.append("Sepsis flag is active and increases clinical concern.")
    return alerts


def explain_patient(
    patient: dict[str, object],
    encoded_features: pd.DataFrame,
    scaled_vector: np.ndarray,
    model: ModelArtifacts,
    autofilled: list[str],
) -> dict[str, object]:
    probability = float(sigmoid(np.array([model.intercept + scaled_vector @ model.feature_weights]))[0])
    band = risk_band(probability)
    contributions = scaled_vector * model.feature_weights
    feature_scores = [
        {
            "name": contributor_label(name),
            "score": float(score),
            "raw_name": name,
        }
        for name, score in zip(model.feature_columns, contributions)
        if abs(float(score)) >= 0.03
    ]
    positive_drivers = [
        item for item in sorted(feature_scores, key=lambda item: item["score"], reverse=True) if item["score"] > 0
    ][:4]
    protective_drivers = [
        item for item in sorted(feature_scores, key=lambda item: item["score"]) if item["score"] < 0
    ][:3]

    patient_rows = []
    for field in [field for _, group in FIELD_GROUPS for field in group if field != "name"]:
        value = patient[field]
        label = FIELD_META[field]["label"]
        if FIELD_META[field]["type"] == "select":
            display = "Yes" if str(value) == "1" else ("No" if str(value) == "0" else str(value))
        else:
            display = format_number(float(value), 2) if isinstance(value, (int, float)) else str(value)
        patient_rows.append((label, display))

    alerts = build_alerts(patient)
    summary_lines = [
        f"The model estimates a {probability * 100:.1f}% mortality probability "
        f"and a {(1 - probability) * 100:.1f}% estimated survival probability.",
        band["summary"],
    ]
    if positive_drivers:
        summary_lines.append(
            "Largest upward pressure comes from "
            + ", ".join(item["name"].lower() for item in positive_drivers[:3])
            + "."
        )
    if protective_drivers:
        summary_lines.append(
            "Protective pressure in this profile comes from "
            + ", ".join(item["name"].lower() for item in protective_drivers[:2])
            + "."
        )
    if autofilled:
        summary_lines.append(
            "Blank advanced fields were completed with cohort medians for "
            + ", ".join(label.lower() for label in autofilled[:5])
            + "."
        )

    return {
        "probability": probability,
        "survival": 1 - probability,
        "band": band,
        "alerts": alerts,
        "positive_drivers": positive_drivers,
        "protective_drivers": protective_drivers,
        "patient_rows": patient_rows,
        "summary": " ".join(summary_lines),
        "entered_fields": sum(1 for value in patient.values() if value not in ("", None)),
        "autofilled": autofilled,
        "feature_vector": encoded_features.iloc[0].to_dict(),
    }


def stat_card(title: str, value: str, note: str = "") -> str:
    note_html = f"<small>{h(note)}</small>" if note else ""
    return (
        "<div class='stat-card'>"
        f"<span class='stat-label'>{h(title)}</span>"
        f"<strong>{h(value)}</strong>"
        f"{note_html}"
        "</div>"
    )


def render_chips(items: list[str], class_name: str) -> str:
    if not items:
        return "<li class='muted'>No notable signals were triggered from the supplied values.</li>"
    return "".join(f"<li class='{h(class_name)}'>{h(item)}</li>" for item in items)


def render_contributors(items: list[dict[str, object]], tone: str) -> str:
    if not items:
        return "<p class='muted'>No stable contributor pattern was strong enough to display.</p>"
    rows = []
    for item in items:
        strength = min(abs(float(item["score"])) * 100, 100)
        rows.append(
            "<div class='contributor'>"
            f"<div><strong>{h(item['name'])}</strong></div>"
            f"<div class='contributor-bar {h(tone)}'><span style='width:{strength:.1f}%'></span></div>"
            "</div>"
        )
    return "".join(rows)


def render_probability_bars(probability: float, survival: float) -> str:
    return (
        "<div class='prob-chart'>"
        "<h3>Outcome Probabilities</h3>"
        "<div class='prob-row'>"
        "<span>Mortality</span>"
        f"<div class='prob-track mortality'><span style='width:{probability:.1f}%'></span></div>"
        f"<strong>{probability:.1f}%</strong>"
        "</div>"
        "<div class='prob-row'>"
        "<span>Survival</span>"
        f"<div class='prob-track survival'><span style='width:{survival:.1f}%'></span></div>"
        f"<strong>{survival:.1f}%</strong>"
        "</div>"
        "</div>"
    )


def render_risk_bar_graph_fallback(probability: float, survival: float, note: str | None = None) -> str:
    note_html = f"<p class='muted'>{h(note)}</p>" if note else ""
    return (
        "<div class='risk-bar-graph risk-bar-graph-fallback'>"
        "<h3>Comparative Chart</h3>"
        f"{note_html}"
        "<div class='prob-chart'>"
        "<div class='prob-row'>"
        "<span>Mortality</span>"
        f"<div class='prob-track mortality'><span style='width:{probability:.1f}%'></span></div>"
        f"<strong>{probability:.1f}%</strong>"
        "</div>"
        "<div class='prob-row'>"
        "<span>Survival</span>"
        f"<div class='prob-track survival'><span style='width:{survival:.1f}%'></span></div>"
        f"<strong>{survival:.1f}%</strong>"
        "</div>"
        "</div>"
        "</div>"
    )


def render_risk_bar_graph(probability: float, survival: float) -> str:
    if plt is None:
        return render_risk_bar_graph_fallback(
            probability,
            survival,
            "Matplotlib is unavailable, so this chart is shown in a simplified inline format.",
        )

    try:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
        fig.patch.set_facecolor("#f3f7fb")
        ax.set_facecolor("#ffffff")
        categories = ["Mortality", "Survival"]
        values = [probability, survival]
        colors = ["#d9485f", "#1f6feb"]

        bars = ax.bar(categories, values, color=colors, width=0.55)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#d8e1eb", alpha=1.0)
        ax.tick_params(colors="#496071", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#d3dde7")
        ax.set_ylabel("Probability (%)", color="#496071", fontsize=9)
        ax.set_title("Risk Comparison", color="#17232d", fontsize=11, pad=8)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                color="#17232d",
                fontsize=9,
                fontweight="bold",
            )

        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            "<div class='risk-bar-graph matplotlib'>"
            "<h3>Comparative Chart</h3>"
            f"<img src='data:image/png;base64,{encoded}' alt='Mortality and survival bar graph'>"
            "</div>"
        )
    except Exception:
        return render_risk_bar_graph_fallback(
            probability,
            survival,
            "The full chart could not be rendered, so a simplified comparison is shown instead.",
        )


def render_results(result: dict[str, object], patient: dict[str, object], model: ModelArtifacts) -> str:
    band = result["band"]
    probability = float(result["probability"]) * 100
    survival = float(result["survival"]) * 100
    patient_name = patient["name"] or "Unidentified Patient"
    patient_table = "".join(
        f"<div class='fact-row'><span>{h(label)}</span><strong>{h(value)}</strong></div>"
        for label, value in result["patient_rows"]
    )
    score_cards = (
        stat_card("Mortality risk", f"{probability:.1f}%", band["label"])
        + stat_card("Survival estimate", f"{survival:.1f}%", "Cohort-based model output")
        + stat_card("Fields completed", str(result["entered_fields"]), "Autofill used where optional")
    )
    alert_count = len(result["alerts"])
    autofill_note = (
        f"{len(result['autofilled'])} optional values were completed from cohort medians."
        if result["autofilled"]
        else "All displayed values were supplied directly."
    )

    return (
        "<section class='panel results-panel'>"
        "<div class='results-hero'>"
        "<div class='results-copy'>"
        "<p class='eyebrow'>Assessment Output</p>"
        "<h2>Clinical Risk Review</h2>"
        f"<p class='patient-name'>{h(patient_name)}</p>"
        f"<p class='lede'>{h(band['summary'])}</p>"
        "<div class='assessment-strip'>"
        f"<span class='band-pill {h(band['class_name'])}'>{h(band['label'])}</span>"
        f"<span>{alert_count} clinical flag{'s' if alert_count != 1 else ''}</span>"
        f"<span>{h(autofill_note)}</span>"
        "</div>"
        "</div>"
        f"<div class='risk-meter {h(band['class_name'])}' style='--score:{probability:.1f}'>"
        f"<div><span>Mortality</span><strong>{probability:.1f}%</strong><small>Survival {survival:.1f}%</small></div>"
        "</div>"
        "</div>"
        f"<div class='score-grid'>{score_cards}</div>"
        "<div class='results-grid'>"
        "<div class='info-card section-span'>"
        f"<p>{h(result['summary'])}</p>"
        "</div>"
        "<div class='info-card'>"
        "<h3>Patient Factors</h3>"
        f"<div class='facts'>{patient_table}</div>"
        "</div>"
        "<div class='info-card'>"
        + render_probability_bars(probability, survival)
        + "</div>"
        "<div class='info-card section-span'>"
        + render_risk_bar_graph(probability, survival)
        + "</div>"
        "<div class='info-card'>"
        "<h3>Clinical Flags</h3>"
        f"<ul class='chip-list'>{render_chips(result['alerts'], 'chip-alert')}</ul>"
        "</div>"
        "<div class='info-card'>"
        "<h3>Upward Risk Drivers</h3>"
        f"{render_contributors(result['positive_drivers'], 'hot')}"
        "</div>"
        "<div class='info-card'>"
        "<h3>Protective Signals</h3>"
        f"{render_contributors(result['protective_drivers'], 'cool')}"
        "</div>"
        "<div class='info-card caution'>"
        "<h3>Use With Clinical Judgement</h3>"
        "<p>This assessment supports triage discussion and review. It should not replace bedside evaluation, imaging, laboratory interpretation, or escalation protocols.</p>"
        "</div>"
        "</div>"
        "</section>"
    )


def render_overview(model: ModelArtifacts) -> str:
    mini_cards = (
        stat_card("Validation AUC", format_percent(model.metrics["auc"]))
        + stat_card("Records", f"{int(model.dataset_summary['records']):,}")
        + stat_card("Baseline mortality", format_percent(model.dataset_summary["mortality_rate"]))
    )
    return (
        "<section class='panel overview-panel'>"
        "<p class='eyebrow'>Operational Overview</p>"
        "<h2>Built For Fast Clinical Review</h2>"
        "<p class='lede'>A compact ICU mortality workspace for intake, signal review, and probability-based decision support.</p>"
        f"<div class='mini-stats'>{mini_cards}</div>"
        "<div class='overview-grid'>"
        "<article class='overview-card'>"
        "<span>Workflow</span>"
        "<strong>Structured intake</strong>"
        "<p>Collect the patient profile, validate ranges against the cohort, and generate a bedside-ready risk summary.</p>"
        "</article>"
        "<article class='overview-card'>"
        "<span>Decision framing</span>"
        f"<strong>Threshold {format_percent(model.threshold, 0)}</strong>"
        "<p>Outputs are grouped into low, moderate, high, and critical watch bands for clearer triage communication.</p>"
        "</article>"
        "<article class='overview-card'>"
        "<span>Cohort signals</span>"
        f"<strong>Median age {format_number(model.dataset_summary['median_age'])}</strong>"
        "<p>The current model is anchored to cohort medians and feature contribution scoring for transparent review.</p>"
        "</article>"
        "</div>"
        "</section>"
    )


def render_home_section(model: ModelArtifacts) -> str:
    records = int(model.dataset_summary["records"])
    mortality_rate = format_percent(model.dataset_summary["mortality_rate"], 1)
    auc = format_percent(model.metrics["auc"], 1)
    threshold = format_percent(model.threshold, 0)
    return (
        "<section class='panel overview-panel home-panel'>"
        "<p class='eyebrow'>Home</p>"
        "<h2>About This App</h2>"
        "<p class='lede'>This workspace estimates ICU mortality risk from bedside variables and presents an interpretable clinical summary for triage discussion.</p>"
        "<div class='overview-grid'>"
        "<article class='overview-card'>"
        "<span>What does it do?</span>"
        "<strong>Predicts mortality probability</strong>"
        "<p>After intake, the app estimates mortality and survival probability, maps the case into a risk band, and highlights major risk drivers and protective signals.</p>"
        "</article>"
        "<article class='overview-card'>"
        "<span>How does it work?</span>"
        "<strong>Feature engineering + logistic model</strong>"
        f"<p>Input values are validated against cohort ranges, engineered into clinical indicators, standardized, and scored by a weighted logistic regression model (validation AUC {auc}, decision threshold {threshold}).</p>"
        "</article>"
        "<article class='overview-card'>"
        "<span>What are the patient records?</span>"
        f"<strong>{records:,} ICU encounters</strong>"
        f"<p>The model is trained on {records:,} rows from the project dataset, with baseline mortality of {mortality_rate}. Optional missing fields are auto-filled using cohort medians or mode defaults.</p>"
        "</article>"
        "<article class='overview-card'>"
        "<span>How is the project done?</span>"
        "<strong>End-to-end local clinical web app</strong>"
        "<p>The project includes data loading, preprocessing, model training, threshold selection, contributor scoring, risk visualization, and a browser interface served locally through a Python HTTP server.</p>"
        "</article>"
        "</div>"
        "</section>"
    )


def render_errors(errors: list[str]) -> str:
    if not errors:
        return ""
    items = "".join(f"<li>{h(error)}</li>" for error in errors)
    return (
        "<div class='error-banner'>"
        "<strong>Check the highlighted inputs before running the assessment.</strong>"
        f"<ul>{items}</ul>"
        "</div>"
    )


def render_page(values: dict[str, str], model: ModelArtifacts, results_html: str, errors: list[str]) -> str:
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.safe_substitute(
        page_title="ICU Risk Studio",
        static_css_path=route_path("/static/style.css"),
        predict_path=route_path("/predict"),
        home_section=render_home_section(model),
        form_sections=render_form_sections(values, model),
        overview_panel=render_overview(model),
        results_panel=results_html,
        error_panel=render_errors(errors),
    )


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        base = normalize_base_path(BASE_PATH)
        home_paths = {"/", "/index.html", base, f"{base}/"}
        if parsed.path in home_paths:
            self.respond_html(render_page({}, MODEL, "", []))
            return
        if parsed.path in {"/health", route_path("/health")}:
            self.respond_json(
                {
                    "status": "ok",
                    "records": int(MODEL.dataset_summary["records"]),
                    "auc": round(MODEL.metrics["auc"], 4),
                }
            )
            return
        if parsed.path == route_path("/static/style.css"):
            self.respond_file(STYLE_PATH, "text/css; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != route_path("/predict"):
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        values = parse_request_values(body)
        patient, errors, autofilled = parse_patient_input(values, MODEL)
        results_html = ""

        if not errors:
            encoded, scaled = make_feature_vector(patient, MODEL)
            results = explain_patient(patient, encoded, scaled, MODEL, autofilled)
            results_html = render_results(results, patient, MODEL)

        self.respond_html(render_page(values, MODEL, results_html, errors))

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def respond_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_json(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run() -> None:
    host = os.environ.get("ICU_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("ICU_APP_PORT", "8000")))
    base = normalize_base_path(BASE_PATH)
    server = ThreadingHTTPServer((host, port), AppHandler)
    public_url = os.environ.get("ICU_APP_PUBLIC_URL", "").strip()
    if public_url:
        root_url = public_url.rstrip("/")
    else:
        browser_host = "localhost" if host in {"0.0.0.0", "::"} else host
        root_url = f"http://{browser_host}:{port}"
    app_url = root_url if base == "/" else f"{root_url}{base}"
    print(f"ICU Risk Studio running at {app_url}")
    server.serve_forever()


MODEL = build_model()


if __name__ == "__main__":
    run()
