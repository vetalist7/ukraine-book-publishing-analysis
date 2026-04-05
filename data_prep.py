import pandas as pd
import numpy as np
from pathlib import Path


def build_dataset() -> pd.DataFrame:
    """Create dataset for Ukraine book publishing analysis (1991–2025)."""

    years = list(range(1991, 2026))

    data = {
        "year": years,
        "total_titles": [
            5855, 5451, 5136, 4710, 6109, 6094, 7036, 7065, 6282, 7583,
            10561, 12444, 13805, 14790, 15720, 17904, 17951, 24040,
            22491, 22826, 22128, 26036, 26323, 22108, 19927, 21330,
            21961, 22612, 24416, 20224, 21095, 8716, 15187, 14500, 15069
        ],
        "total_copies_mln": [
            136.4, 125.8, 89.2, 54.1, 48.2, 51.5, 55.3, 36.4, 21.6, 21.3,
            47.3, 51.1, 44.5, 52.8, 54.1, 54.2, 56.1, 56.0, 48.5, 45.1,
            46.5, 62.2, 69.5, 55.3, 36.4, 48.9, 45.1, 47.3, 61.0, 20.1,
            44.8, 11.6, 24.6, 34.0, 33.2
        ],
        "ukr_copies_mln": [
            45.5, 41.2, 35.1, 31.5, 28.1, 29.5, 30.1, 20.5, 11.2, 12.1,
            30.1, 32.1, 28.5, 33.1, 33.1, 32.5, 31.7, 28.5, 27.5, 26.5,
            34.5, 35.2, 30.1, 20.5, 30.2, 27.5, 30.1, 48.2, 48.2, 14.5,
            34.4, 7.5, 22.6, 31.4, 30.7
        ],
        "rus_copies_mln": [
            85.2, 79.5, 49.1, 19.5, 17.1, 19.2, 22.0, 13.5, 9.1, 8.2,
            14.5, 16.2, 14.1, 17.5, 18.2, 20.5, 21.5, 17.5, 15.0, 17.1,
            24.2, 31.4, 22.0, 13.5, 16.5, 15.0, 14.5, 10.5, 10.5, 4.5,
            1.9, 0.4, 0.29, 0.35, 0.25
        ]
    }

    df = pd.DataFrame(data)

    # --- Thematic data (from 2009) ---
    thematic_years = list(range(2009, 2026))

    edu_titles = [9850, 9600, 9400, 11500, 12100, 9800, 8500, 9200, 9100, 9400, 10500, 8900, 9100, 3800, 6800, 6400, 6800]
    fiction_titles = [3400, 3500, 3600, 4200, 4100, 3900, 3400, 3800, 3900, 4100, 4500, 3800, 4200, 1800, 2900, 3100, 3200]
    non_fiction_titles = [2500, 2600, 2700, 3100, 3200, 2900, 2400, 2800, 2900, 3100, 3500, 2900, 3100, 1300, 2400, 2500, 2600]

    thematic_df = pd.DataFrame({
        "year": thematic_years,
        "edu_titles": edu_titles,
        "fiction_titles": fiction_titles,
        "non_fiction_titles": non_fiction_titles
    })

    df = df.merge(thematic_df, on="year", how="left")

    # --- Feature engineering ---
    df["avg_print_run"] = (df["total_copies_mln"] * 1_000_000 / df["total_titles"]).round(0)

    df["ukr_share_pct"] = (df["ukr_copies_mln"] / df["total_copies_mln"] * 100).round(2)

    df["other_copies_mln"] = (
        df["total_copies_mln"]
        - df["ukr_copies_mln"]
        - df["rus_copies_mln"]
    )

    df["non_fiction_share_pct"] = (
        df["non_fiction_titles"] / df["total_titles"] * 100
    ).round(2)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Basic validation checks."""

    assert df["year"].is_unique, "Years must be unique"
    assert (df["total_copies_mln"] > 0).all(), "Total copies must be positive"

    diff = (
        df["ukr_copies_mln"]
        + df["rus_copies_mln"]
        + df["other_copies_mln"]
        - df["total_copies_mln"]
    ).abs().max()

    print(f"Max deviation in totals: {diff}")


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = build_dataset()
    validate_dataset(df)

    output_file = Path("data/processed/ukraine_books.csv")
    save_dataset(df, output_file)

    print("Dataset created successfully!")