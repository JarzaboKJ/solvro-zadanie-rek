"""Uruchamia cały pipeline: generowanie danych, detekcja, ewaluacja."""
import matplotlib

matplotlib.use("Agg")

from src import detect, evaluate, generate


def main() -> None:
    print("=== Generowanie danych ===")
    generate.main()

    print("\n=== Detekcja ===")
    detect.main()

    print("\n=== Ewaluacja ===")
    evaluate.main()


if __name__ == "__main__":
    main()
