"""Ewaluacja detekcji: dopasowanie punktów, metryki i wizualizacja wyników."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.config import (
    CALIBRATION_PATH,
    FIGURES_DIR,
    GROUND_TRUTH_POINTS_PATH,
    IMAGES_DIR,
    MATCH_TOLERANCE_FRACTION,
    METRICS_PATH,
    PREDICTIONS_PATH,
    SUMMARY_PATH,
)
from src.generate import data_to_pixel


def points_for_image(df: pd.DataFrame, image_id: str, x_col: str, y_col: str) -> list:
    """Zwraca listę punktów (x, y) jednego obrazu z tabeli."""
    subset = df[df["image_id"] == image_id]
    return list(zip(subset[x_col], subset[y_col]))


def match_points(true_points: list, pred_points: list, x_range: float, y_range: float) -> list:
    """Dopasowuje prawdziwe punkty do wykrytych zachłannie po najbliższym sąsiedzie w tolerancji."""
    unmatched_pred = list(range(len(pred_points)))
    matches = []
    for true_idx, (tx, ty) in enumerate(true_points):
        if not unmatched_pred:
            continue
        best_pred_idx = min(
            unmatched_pred,
            key=lambda i: ((pred_points[i][0] - tx) / x_range) ** 2 + ((pred_points[i][1] - ty) / y_range) ** 2,
        )
        px, py = pred_points[best_pred_idx]
        err_x = abs(px - tx) / x_range
        err_y = abs(py - ty) / y_range
        if err_x <= MATCH_TOLERANCE_FRACTION and err_y <= MATCH_TOLERANCE_FRACTION:
            matches.append({"true_idx": true_idx, "pred_idx": best_pred_idx, "err_x": err_x, "err_y": err_y})
            unmatched_pred.remove(best_pred_idx)
    return matches


def f1_score(n_matched: int, n_true: int, n_pred: int) -> float:
    """Liczy F1 z liczby dopasowań, prawdziwych i wykrytych punktów."""
    if n_true == 0 and n_pred == 0:
        return np.nan
    if n_matched == 0:
        return 0.0
    precision = n_matched / n_pred
    recall = n_matched / n_true
    return 2 * precision * recall / (precision + recall)


def evaluate_image(true_points: list, pred_points: list, calibration: pd.Series) -> tuple:
    """Dopasowuje punkty jednego obrazu i liczy jego metryki."""
    x_range = calibration["x_max"] - calibration["x_min"]
    y_range = calibration["y_max"] - calibration["y_min"]
    matches = match_points(true_points, pred_points, x_range, y_range)

    n_true, n_pred, n_matched = len(true_points), len(pred_points), len(matches)
    metrics = {
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "precision": n_matched / n_pred if n_pred > 0 else np.nan,
        "recall": n_matched / n_true if n_true > 0 else np.nan,
        "f1": f1_score(n_matched, n_true, n_pred),
        "mae_x": np.mean([m["err_x"] for m in matches]) if matches else np.nan,
        "mae_y": np.mean([m["err_y"] for m in matches]) if matches else np.nan,
    }
    return metrics, matches


def summarize_split(metrics_df: pd.DataFrame, match_errors_df: pd.DataFrame, split: str) -> dict:
    """Liczy metryki zagregowane po wszystkich obrazach jednego podzbioru."""
    subset = metrics_df[metrics_df["split"] == split]
    n_true, n_pred, n_matched = subset["n_true"].sum(), subset["n_pred"].sum(), subset["n_matched"].sum()
    split_errors = match_errors_df[match_errors_df["split"] == split]
    return {
        "split": split,
        "n_images": len(subset),
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "precision": n_matched / n_pred if n_pred > 0 else np.nan,
        "recall": n_matched / n_true if n_true > 0 else np.nan,
        "f1": f1_score(n_matched, n_true, n_pred),
        "mae_x": split_errors["err_x"].mean() if len(split_errors) else np.nan,
        "mae_y": split_errors["err_y"].mean() if len(split_errors) else np.nan,
    }


def select_examples(metrics_df: pd.DataFrame, n_successes: int = 3) -> tuple:
    """Wybiera obrazy testowe z porażką detekcji i przykłady pełnego sukcesu."""
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    failures = test_metrics[
        (test_metrics["n_matched"] < test_metrics["n_true"]) | (test_metrics["n_pred"] > test_metrics["n_matched"])
    ]["image_id"].tolist()
    successes = test_metrics[
        (test_metrics["n_matched"] == test_metrics["n_true"]) & (test_metrics["n_pred"] == test_metrics["n_true"])
    ]["image_id"].head(n_successes).tolist()
    return failures, successes


def plot_detection(
    image_id: str, calibration: pd.Series, true_points: list, pred_points: list, matches: list, path: Path = None
) -> Figure:
    """Rysuje obraz z prawdziwymi i wykrytymi punktami, zaznaczając niedopasowane, i zapisuje go do pliku jeśli podano ścieżkę."""
    matched_true_idx = {m["true_idx"] for m in matches}
    matched_pred_idx = {m["pred_idx"] for m in matches}

    true_arr = np.array(true_points).reshape(-1, 2) if true_points else np.empty((0, 2))
    pred_arr = np.array(pred_points).reshape(-1, 2) if pred_points else np.empty((0, 2))
    true_col, true_row = data_to_pixel(true_arr[:, 0], true_arr[:, 1], calibration)
    pred_col, pred_row = data_to_pixel(pred_arr[:, 0], pred_arr[:, 1], calibration)

    image = plt.imread(IMAGES_DIR / calibration["split"] / f"{image_id}.png")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(image)

    for i in range(len(true_points)):
        color = "lime" if i in matched_true_idx else "black"
        ax.scatter(true_col[i], true_row[i], marker="o", facecolors="none", edgecolors=color, s=140, linewidths=2)
    for i in range(len(pred_points)):
        color = "lime" if i in matched_pred_idx else "red"
        ax.scatter(pred_col[i], pred_row[i], marker="+", color=color, s=140, linewidths=2)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color="lime", label="prawdziwy, dopasowany"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color="black", label="prawdziwy, pominięty"),
        Line2D([], [], marker="+", linestyle="", color="lime", label="wykryty, dopasowany"),
        Line2D([], [], marker="+", linestyle="", color="red", label="wykryty, błędny"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=1.0, facecolor="white")

    n_true, n_pred, n_matched = len(true_points), len(pred_points), len(matches)
    ax.set_title(f"{image_id}: {n_matched}/{n_true} dopasowań, {n_pred} wykrytych punktów")
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def main() -> None:
    """Liczy metryki i rysuje wizualizacje na podstawie zapisanych wyników detekcji."""
    calibration_df = pd.read_csv(CALIBRATION_PATH).set_index("image_id")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_POINTS_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    metrics_rows = []
    match_errors_rows = []
    matches_by_image = {}
    for image_id, calibration in calibration_df.iterrows():
        true_points = points_for_image(ground_truth_df, image_id, "x_true", "y_true")
        pred_points = points_for_image(predictions_df, image_id, "x_pred", "y_pred")
        metrics, matches = evaluate_image(true_points, pred_points, calibration)
        metrics_rows.append({"image_id": image_id, "split": calibration["split"], **metrics})
        matches_by_image[image_id] = matches
        for m in matches:
            match_errors_rows.append({"split": calibration["split"], "err_x": m["err_x"], "err_y": m["err_y"]})

    metrics_df = pd.DataFrame(metrics_rows)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Metryki per obraz zapisane w {METRICS_PATH}")

    match_errors_df = pd.DataFrame(match_errors_rows, columns=["split", "err_x", "err_y"])
    summary_df = pd.DataFrame([summarize_split(metrics_df, match_errors_df, split) for split in ["dev", "test"]])
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nPodsumowanie dev/test zapisane w {SUMMARY_PATH}")
    print(summary_df.round(4).to_string(index=False))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    failures, successes = select_examples(metrics_df)

    for image_id in failures + successes:
        calibration = calibration_df.loc[image_id]
        true_points = points_for_image(ground_truth_df, image_id, "x_true", "y_true")
        pred_points = points_for_image(predictions_df, image_id, "x_pred", "y_pred")
        kind = "failure" if image_id in failures else "success"
        fig = plot_detection(
            image_id, calibration, true_points, pred_points, matches_by_image[image_id],
            FIGURES_DIR / f"detection_{kind}_{image_id}.png",
        )
        plt.close(fig)
    print(f"\nWizualizacje ({len(failures) + len(successes)}) zapisane w {FIGURES_DIR}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
