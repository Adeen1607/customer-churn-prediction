"""Train, select, and evaluate telecom churn classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=563)
    features = dataset.data.features.copy()
    target = dataset.data.targets.squeeze().copy()

    combined = pd.concat([features, target.rename("target")], axis=1).drop_duplicates()
    features = combined.drop(columns="target")
    target = combined["target"]

    labels = sorted(target.dropna().unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"Expected a binary churn target, found {labels}.")
    if set(labels) != {0, 1}:
        target = target.map({labels[0]: 0, labels[1]: 1})
    return features, target.astype(int)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def select_threshold(
    labels: pd.Series,
    probabilities: pd.Series,
    minimum_recall: float,
) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    candidates = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    eligible = candidates[candidates["recall"].ge(minimum_recall)]
    if eligible.empty:
        raise ValueError("No validation threshold satisfies the recall requirement.")
    selected = eligible.sort_values(
        ["precision", "threshold"], ascending=[False, False]
    ).iloc[0]
    return (
        float(selected["threshold"]),
        float(selected["precision"]),
        float(selected["recall"]),
    )


def evaluate(
    labels: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_plots(
    labels: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
    output_dir: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, color="#2563EB")
    axis.set(title="Churn Precision–Recall Curve", xlabel="Recall", ylabel="Precision")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=axis,
    )
    axis.set(title="Held-out Churn Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def train(output_dir: Path, minimum_recall: float, seed: int) -> None:
    if not 0 < minimum_recall <= 1:
        raise ValueError("--minimum-recall must be greater than zero and at most one.")

    features, target = load_data()
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=seed,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=seed,
    )

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }

    candidates = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessing", make_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        candidates[name] = {
            "pipeline": pipeline,
            "validation_average_precision": average_precision_score(
                y_validation, probability
            ),
            "validation_probability": probability,
        }

    selected_name = max(
        candidates,
        key=lambda name: candidates[name]["validation_average_precision"],
    )
    selected = candidates[selected_name]
    validation_probability = pd.Series(
        selected["validation_probability"], index=y_validation.index
    )
    threshold, validation_precision, validation_recall = select_threshold(
        y_validation,
        validation_probability,
        minimum_recall,
    )

    test_probability = pd.Series(
        selected["pipeline"].predict_proba(x_test)[:, 1],
        index=y_test.index,
        name="churn_probability",
    )
    test_prediction = (test_probability >= threshold).astype(int)

    metrics = evaluate(y_test, test_probability, test_prediction)
    metrics.update(
        {
            "selected_model": selected_name,
            "decision_threshold": threshold,
            "validation_precision_at_threshold": validation_precision,
            "validation_recall_at_threshold": validation_recall,
            "validation_average_precision": float(
                selected["validation_average_precision"]
            ),
            "churn_prevalence": float(target.mean()),
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "random_seed": seed,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame(
        {
            "actual": y_test,
            "churn_probability": test_probability,
            "predicted": test_prediction,
        }
    ).sort_index().to_csv(output_dir / "test_predictions.csv", index_label="row_index")
    joblib.dump(selected["pipeline"], output_dir / "churn_pipeline.joblib")
    save_plots(y_test, test_probability, test_prediction, output_dir)
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn classifiers.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--minimum-recall", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.output_dir, arguments.minimum_recall, arguments.seed)
