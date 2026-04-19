from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "raw_annotation_summary.csv"

META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV = META_DIR / "cleaned_annotations_stage1.csv"
REMOVED_INVALID_CSV = META_DIR / "removed_invalid_boxes.csv"
REMOVED_BAD_CLASS_CSV = META_DIR / "removed_bad_class_rows.csv"
RENAMED_CLASS_CSV = META_DIR / "renamed_class_rows.csv"
FLAGGED_LARGE_BOX_CSV = META_DIR / "flagged_large_boxes.csv"


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到原始 CSV: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    print("=== stage 1 cleaning: start ===")
    print(f"raw rows: {len(df)}")

    # ---------------------------
    # 1. 删除非法框
    # ---------------------------
    invalid_mask = (
        df["image_width"].isna() |
        df["image_height"].isna() |
        df["xmin"].isna() |
        df["ymin"].isna() |
        df["xmax"].isna() |
        df["ymax"].isna() |
        (df["xmin"] < 0) |
        (df["ymin"] < 0) |
        (df["xmax"] > df["image_width"]) |
        (df["ymax"] > df["image_height"]) |
        (df["xmin"] >= df["xmax"]) |
        (df["ymin"] >= df["ymax"])
    )

    removed_invalid = df[invalid_mask].copy()
    df = df[~invalid_mask].copy()

    removed_invalid.to_csv(REMOVED_INVALID_CSV, index=False, encoding="utf-8-sig")

    print(f"removed invalid boxes: {len(removed_invalid)}")
    print(f"rows after invalid removal: {len(df)}")

    # ---------------------------
    # 2. 删除明确错误类别 d
    # ---------------------------
    bad_class_mask = df["class_name"] == "d"
    removed_bad_class = df[bad_class_mask].copy()
    df = df[~bad_class_mask].copy()

    removed_bad_class.to_csv(REMOVED_BAD_CLASS_CSV, index=False, encoding="utf-8-sig")

    print(f"removed bad class rows: {len(removed_bad_class)}")
    print(f"rows after bad class removal: {len(df)}")

    # ---------------------------
    # 3. 统一类别命名
    # 10_yaozhed -> 10_yaozhe
    # ---------------------------
    rename_mask = df["class_name"] == "10_yaozhed"
    renamed_rows = df[rename_mask].copy()

    df.loc[rename_mask, "class_name"] = "10_yaozhe"

    renamed_rows.to_csv(RENAMED_CLASS_CSV, index=False, encoding="utf-8-sig")

    print(f"renamed class rows: {len(renamed_rows)}")

    # ---------------------------
    # 4. 重建 class_id
    # ---------------------------
    class_names = sorted(df["class_name"].dropna().unique().tolist())
    class_map = {name: idx for idx, name in enumerate(class_names)}
    df["class_id"] = df["class_name"].map(class_map)

    print("\n=== new class mapping ===")
    for name, idx in class_map.items():
        print(f"{idx}: {name}")

    # ---------------------------
    # 5. 过滤过小框
    # 条件：min(w, h) < 20
    # ---------------------------
    df["bbox_w"] = df["xmax"] - df["xmin"]
    df["bbox_h"] = df["ymax"] - df["ymin"]
    df["min_side"] = df[["bbox_w", "bbox_h"]].min(axis=1)

    small_box_mask = df["min_side"] < 20
    removed_small_boxes = df[small_box_mask].copy()
    df = df[~small_box_mask].copy()

    REMOVED_SMALL_BOX_CSV = META_DIR / "removed_small_boxes.csv"
    removed_small_boxes.to_csv(REMOVED_SMALL_BOX_CSV, index=False, encoding="utf-8-sig")

    print(f"removed small boxes (min_side < 20): {len(removed_small_boxes)}")
    print(f"rows after small box removal: {len(df)}")

    # ---------------------------
    # 6. 标记异常大框（先标记，不删除）
    # 条件：bbox_area / image_area > 0.5
    # ---------------------------
    df["bbox_w"] = df["xmax"] - df["xmin"]
    df["bbox_h"] = df["ymax"] - df["ymin"]
    df["bbox_area"] = df["bbox_w"] * df["bbox_h"]
    df["image_area"] = df["image_width"] * df["image_height"]
    df["bbox_area_ratio"] = df["bbox_area"] / df["image_area"]

    large_box_mask = df["bbox_area_ratio"] > 0.5
    flagged_large = df[large_box_mask].copy()
    flagged_large.to_csv(FLAGGED_LARGE_BOX_CSV, index=False, encoding="utf-8-sig")

    print(f"\nflagged large boxes (>50% image area): {len(flagged_large)}")


    # ---------------------------
    # 7. 保存阶段 1 清洗结果
    # ---------------------------
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")

    print("\n=== stage 1 cleaning: finished ===")
    print(f"saved cleaned csv: {CLEANED_CSV}")
    print(f"final rows: {len(df)}")
    print(f"final images: {df['image_path'].nunique()}")
    print(f"final classes: {df['class_name'].nunique()}")



if __name__ == "__main__":
    main()