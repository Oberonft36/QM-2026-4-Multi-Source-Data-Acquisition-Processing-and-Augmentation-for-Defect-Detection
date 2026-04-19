from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "meta" / "cleaned_annotations_stage1.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    df["bbox_w"] = df["xmax"] - df["xmin"]
    df["bbox_h"] = df["ymax"] - df["ymin"]
    df["bbox_area"] = df["bbox_w"] * df["bbox_h"]

    print("=== overall bbox size stats ===")
    print(df[["bbox_w", "bbox_h", "bbox_area"]].describe())

    print("\n=== count under thresholds ===")
    print(f"min(w,h) < 4  : {(df[['bbox_w','bbox_h']].min(axis=1) < 4).sum()}")
    print(f"min(w,h) < 8  : {(df[['bbox_w','bbox_h']].min(axis=1) < 8).sum()}")
    print(f"min(w,h) < 12 : {(df[['bbox_w','bbox_h']].min(axis=1) < 12).sum()}")
    print(f"bbox_area < 16 : {(df['bbox_area'] < 16).sum()}")
    print(f"bbox_area < 32 : {(df['bbox_area'] < 32).sum()}")
    print(f"bbox_area < 64 : {(df['bbox_area'] < 64).sum()}")

    print("\n=== per-class small box stats ===")
    temp = df.copy()
    temp["min_side"] = temp[["bbox_w", "bbox_h"]].min(axis=1)

    result = temp.groupby("class_name").agg(
        total_boxes=("class_name", "count"),
        min_side_lt_8=("min_side", lambda s: (s < 8).sum()),
        min_side_lt_12=("min_side", lambda s: (s < 12).sum()),
        area_lt_32=("bbox_area", lambda s: (s < 32).sum()),
        area_lt_64=("bbox_area", lambda s: (s < 64).sum()),
    )

    print(result.sort_values("total_boxes", ascending=False))


if __name__ == "__main__":
    main()