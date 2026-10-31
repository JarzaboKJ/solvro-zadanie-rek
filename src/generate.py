"""Generowanie syntetycznych wykresów punktowych i ich ground truth."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import (
    CALIBRATION_PATH,
    DEV_SEEDS,
    DPI,
    FIGSIZE,
    FIGURES_DIR,
    GRID_PROBABILITY,
    GROUND_TRUTH_POINTS_PATH,
    IMAGES_DIR,
    MARKER_COLORS,
    MARKER_SIZE,
    N_POINTS_MAX,
    N_POINTS_MIN,
    TEST_SEEDS,
)


def sample_plot_params(rng: np.random.Generator) -> dict:
    """Losuje zakresy osi, liczbę punktów, kolor markerów i siatkę dla jednego wykresu."""
    x_min = rng.uniform(0, 50)
    x_max = x_min + rng.uniform(10, 100)
    y_min = rng.uniform(-20, 20)
    y_max = y_min + rng.uniform(5, 50)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "n_points": int(rng.integers(N_POINTS_MIN, N_POINTS_MAX + 1)),
        "color": str(rng.choice(MARKER_COLORS)),
        "grid": bool(rng.random() < GRID_PROBABILITY),
    }


def sample_points(rng: np.random.Generator, params: dict) -> np.ndarray:
    """Losuje prawdziwe punkty na wykresie w zakresach osi, z marginesem od krawędzi."""
    margin_x = 0.05 * (params["x_max"] - params["x_min"])
    margin_y = 0.05 * (params["y_max"] - params["y_min"])
    x = rng.uniform(params["x_min"] + margin_x, params["x_max"] - margin_x, params["n_points"])
    y = rng.uniform(params["y_min"] + margin_y, params["y_max"] - margin_y, params["n_points"])
    return np.column_stack([x, y])


def render_plot(points: np.ndarray, params: dict, path: Path) -> dict:
    """Rysuje wykres punktowy, zapisuje go jako PNG i zwraca kalibrację pikseli."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_axisbelow(True)
    ax.scatter(points[:, 0], points[:, 1], c=params["color"], s=MARKER_SIZE, edgecolors="none")
    ax.set_xlim(params["x_min"], params["x_max"])
    ax.set_ylim(params["y_min"], params["y_max"])
    if params["grid"]:
        ax.grid(True, color="0.8", linewidth=0.8)
    fig.tight_layout()

    # Trzeba narysować figurę, zanim get_window_extent() odda realne pozycje w pikselach.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    img_height_px = FIGSIZE[1] * DPI

    fig.savefig(path, dpi=DPI)
    plt.close(fig)

    return {
        "px_left": bbox.x0,
        "px_right": bbox.x1,
        # matplotlib liczy piksele od dołu, obraz od góry - stąd odwrócenie.
        "px_top": img_height_px - bbox.y1,
        "px_bottom": img_height_px - bbox.y0,
    }


def generate_split(seeds: list, split: str) -> tuple:
    """Generuje wszystkie obrazy jednego podzbioru i zwraca wiersze ground truth oraz kalibracji."""
    out_dir = IMAGES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    point_rows = []
    calibration_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        params = sample_plot_params(rng)
        points = sample_points(rng, params)
        image_id = f"{split}_{seed:03d}"
        calibration = render_plot(points, params, out_dir / f"{image_id}.png")

        for i, (x, y) in enumerate(points):
            point_rows.append({"image_id": image_id, "point_idx": i, "x_true": x, "y_true": y})

        calibration_rows.append({
            "image_id": image_id,
            "split": split,
            "seed": seed,
            "n_points": params["n_points"],
            "color": params["color"],
            "grid": params["grid"],
            "x_min": params["x_min"],
            "x_max": params["x_max"],
            "y_min": params["y_min"],
            "y_max": params["y_max"],
            **calibration,
        })
    return point_rows, calibration_rows


def data_to_pixel(x: np.ndarray, y: np.ndarray, calibration: dict) -> tuple:
    """Przelicza wartości osi na współrzędne pikselowe w układzie obrazu."""
    px_col = calibration["px_left"] + (x - calibration["x_min"]) / (calibration["x_max"] - calibration["x_min"]) * (
        calibration["px_right"] - calibration["px_left"]
    )
    px_row = calibration["px_bottom"] - (y - calibration["y_min"]) / (calibration["y_max"] - calibration["y_min"]) * (
        calibration["px_bottom"] - calibration["px_top"]
    )
    return px_col, px_row


def save_calibration_sanity_check(image_id: str, split: str, calibration: dict, points: pd.DataFrame) -> Figure:
    """Rysuje prawdziwe punkty przeliczone na piksele na zapisanym obrazie i zapisuje go do pliku."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    image = plt.imread(IMAGES_DIR / split / f"{image_id}.png")
    px_col, px_row = data_to_pixel(points["x_true"].to_numpy(), points["y_true"].to_numpy(), calibration)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.imshow(image)
    ax.scatter(px_col, px_row, facecolors="none", edgecolors="black", s=150, linewidths=1.5)
    ax.set_title(f"Sanity check kalibracji - {image_id}\n(czarne kółka = prawdziwe punkty przeliczone na piksele)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_sanity_check.png", dpi=150)
    return fig


def main() -> None:
    """Generuje dev i test, zapisuje obrazy i ground truth, robi sanity check kalibracji."""
    all_points = []
    all_calibration = []
    for seeds, split in [(DEV_SEEDS, "dev"), (TEST_SEEDS, "test")]:
        point_rows, calibration_rows = generate_split(seeds, split)
        all_points += point_rows
        all_calibration += calibration_rows
        print(f"{split}: {len(seeds)} obrazów zapisanych w {IMAGES_DIR / split}")

    GROUND_TRUTH_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    points_df = pd.DataFrame(all_points)
    calibration_df = pd.DataFrame(all_calibration)
    points_df.to_csv(GROUND_TRUTH_POINTS_PATH, index=False)
    calibration_df.to_csv(CALIBRATION_PATH, index=False)
    print(f"\nGround truth punktów: {GROUND_TRUTH_POINTS_PATH} ({len(points_df)} wierszy)")
    print(f"Kalibracja: {CALIBRATION_PATH} ({len(calibration_df)} wierszy)")

    example = next(row for row in all_calibration if row["grid"])
    example_points = points_df[points_df["image_id"] == example["image_id"]]
    fig = save_calibration_sanity_check(example["image_id"], example["split"], example, example_points)
    plt.close(fig)
    print(f"\nSanity check kalibracji zapisany w {FIGURES_DIR / 'calibration_sanity_check.png'}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
