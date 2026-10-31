"""Wspólne stałe: ścieżki, parametry generowania danych, tolerancja ewaluacji."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
GROUND_TRUTH_POINTS_PATH = DATA_DIR / "ground_truth_points.csv"
CALIBRATION_PATH = DATA_DIR / "calibration.csv"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

# Osobne zakresy seedów dla dev i test, żeby test nie powtarzał obrazów z dev.
DEV_SEEDS = list(range(0, 10))
TEST_SEEDS = list(range(100, 120))

FIGSIZE = (6, 4)  # cale
DPI = 100  # razem z FIGSIZE daje obraz 600x400 px

N_POINTS_MIN = 5
N_POINTS_MAX = 15
MARKER_COLORS = ["red", "blue", "green", "orange", "purple", "magenta"]
MARKER_SIZE = 80
GRID_PROBABILITY = 0.5

MATCH_TOLERANCE_FRACTION = 0.02  # 2% rozpiętości osi, osobno dla x i y

# Dobrane na dev: markery mają nasycenie 255, tło blisko 0, plamy mają ok. 100 px2.
SATURATION_THRESHOLD = 100
MIN_BLOB_AREA_PX = 50
