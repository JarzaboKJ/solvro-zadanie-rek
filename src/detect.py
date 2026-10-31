"""Detekcja punktów na wykresie na podstawie nasycenia koloru markerów."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    CALIBRATION_PATH,
    IMAGES_DIR,
    MIN_BLOB_AREA_PX,
    PREDICTIONS_PATH,
    SATURATION_THRESHOLD,
)


def saturation_mask(image_path: Path) -> np.ndarray:
    """Wczytuje obraz i zwraca maskę pikseli o nasyceniu powyżej progu."""
    image = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return (saturation > SATURATION_THRESHOLD).astype(np.uint8) * 255


def blob_centroids(mask: np.ndarray) -> list:
    """Znajduje środki plam w masce, odrzucając te mniejsze niż minimalny rozmiar."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_BLOB_AREA_PX:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroids.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return centroids


def pixel_to_data(px_col: float, px_row: float, calibration: pd.Series) -> tuple:
    """Przelicza współrzędne pikselowe na wartości osi."""
    x = calibration["x_min"] + (px_col - calibration["px_left"]) / (
        calibration["px_right"] - calibration["px_left"]
    ) * (calibration["x_max"] - calibration["x_min"])
    y = calibration["y_min"] + (calibration["px_bottom"] - px_row) / (
        calibration["px_bottom"] - calibration["px_top"]
    ) * (calibration["y_max"] - calibration["y_min"])
    return x, y


def detect_image(image_id: str, calibration: pd.Series) -> list:
    """Wykrywa punkty na jednym obrazie i zwraca je jako wiersze predictions."""
    path = IMAGES_DIR / calibration["split"] / f"{image_id}.png"
    mask = saturation_mask(path)
    rows = []
    for px_col, px_row in blob_centroids(mask):
        x_pred, y_pred = pixel_to_data(px_col, px_row, calibration)
        rows.append({"image_id": image_id, "split": calibration["split"], "x_pred": x_pred, "y_pred": y_pred})
    return rows


def main() -> None:
    """Wykrywa punkty na wszystkich obrazach i zapisuje predictions.csv."""
    calibration_df = pd.read_csv(CALIBRATION_PATH).set_index("image_id")

    all_rows = []
    counts = []
    for image_id, calibration in calibration_df.iterrows():
        rows = detect_image(image_id, calibration)
        all_rows += rows
        counts.append({
            "image_id": image_id,
            "split": calibration["split"],
            "n_true": calibration["n_points"],
            "n_detected": len(rows),
        })

    predictions_df = pd.DataFrame(all_rows)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Zapisano {len(predictions_df)} wykrytych punktów do {PREDICTIONS_PATH}")

    counts_df = pd.DataFrame(counts)
    print("\nDev - wykryte punkty vs prawdziwe:")
    print(counts_df[counts_df["split"] == "dev"].drop(columns="split").to_string(index=False))


if __name__ == "__main__":
    main()
