from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = PROJECT_ROOT / "raw_annotation_summary.csv"
CLEAN_STAGE1_CSV = META_DIR / "cleaned_annotations_stage1.csv"
SQUARED_CSV = META_DIR / "squared_annotations.csv"
FINAL_CSV = META_DIR / "final_annotations_with_split.csv"
AUG_LOG_CSV = META_DIR / "augmentation_log.csv"

OUT_SPLIT_TABLE = META_DIR / "report_table_split.csv"
OUT_CLASS_TABLE = META_DIR / "report_table_class.csv"
OUT_CLEANING_TABLE = META_DIR / "report_table_cleaning_stages.csv"
OUT_IMAGE_GRAIN_TABLE = META_DIR / "report_table_image_granularity.csv"
OUT_AUG_TABLE = META_DIR / "report_table_augmentation.csv"


def main():
    raw_df = pd.read_csv(RAW_CSV)
    clean_df = pd.read_csv(CLEAN_STAGE1_CSV)
    squared_df = pd.read_csv(SQUARED_CSV)
    final_df = pd.read_csv(FINAL_CSV)
    aug_df = pd.read_csv(AUG_LOG_CSV)

    # 1. split 统计
    split_rows = []
    for split in ["train", "val", "test"]:
        sub = final_df[final_df["split"] == split].copy()
        n_images = sub["image_path"].nunique()
        n_boxes = len(sub)
        avg_boxes = n_boxes / n_images if n_images > 0 else 0
        split_rows.append({
            "split": split,
            "num_images": n_images,
            "num_boxes": n_boxes,
            "avg_boxes_per_image": round(avg_boxes, 4)
        })
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(OUT_SPLIT_TABLE, index=False, encoding="utf-8-sig")

    # 2. 类别统计（基于最终 split 前后的有效数据）
    class_table = final_df.groupby("class_name").agg(
        num_boxes=("class_name", "count"),
        num_images=("image_path", "nunique")
    ).reset_index()

    class_table["box_ratio"] = (class_table["num_boxes"] / class_table["num_boxes"].sum()).round(4)
    class_table = class_table.sort_values("num_boxes", ascending=False)
    class_table.to_csv(OUT_CLASS_TABLE, index=False, encoding="utf-8-sig")

    # 3. 清洗阶段统计
    raw_images = raw_df["image_path"].nunique()
    raw_boxes = len(raw_df)

    clean_images = clean_df["image_path"].nunique()
    clean_boxes = len(clean_df)

    squared_images = squared_df["image_path"].nunique()
    squared_boxes = len(squared_df)

    final_images = final_df["image_path"].nunique()
    final_boxes = len(final_df)

    aug_success = (aug_df["status"] == "success").sum()
    aug_fail = (aug_df["status"] != "success").sum()

    cleaning_rows = [
        {"stage": "raw_parse", "num_images": raw_images, "num_boxes": raw_boxes},
        {"stage": "clean_stage1", "num_images": clean_images, "num_boxes": clean_boxes},
        {"stage": "square_pad", "num_images": squared_images, "num_boxes": squared_boxes},
        {"stage": "final_split_total", "num_images": final_images, "num_boxes": final_boxes},
        {"stage": "train_aug_success", "num_images": aug_success, "num_boxes": aug_success},  # 每张生成1张增强图
        {"stage": "train_aug_failed", "num_images": aug_fail, "num_boxes": None},
    ]
    cleaning_table = pd.DataFrame(cleaning_rows)
    cleaning_table.to_csv(OUT_CLEANING_TABLE, index=False, encoding="utf-8-sig")

    # 4. 每图缺陷粒度统计（基于最终数据）
    img_box_count = final_df.groupby("image_path").size().reset_index(name="num_boxes_in_image")

    grain_rows = [
        {
            "bucket": "1 defect",
            "num_images": int((img_box_count["num_boxes_in_image"] == 1).sum())
        },
        {
            "bucket": "2-3 defects",
            "num_images": int(((img_box_count["num_boxes_in_image"] >= 2) & (img_box_count["num_boxes_in_image"] <= 3)).sum())
        },
        {
            "bucket": "4+ defects",
            "num_images": int((img_box_count["num_boxes_in_image"] >= 4).sum())
        },
    ]
    grain_table = pd.DataFrame(grain_rows)
    grain_table.to_csv(OUT_IMAGE_GRAIN_TABLE, index=False, encoding="utf-8-sig")

    # 5. 增强统计
    aug_table = aug_df.groupby("status").size().reset_index(name="count")
    aug_table.to_csv(OUT_AUG_TABLE, index=False, encoding="utf-8-sig")

    print("=== report tables generated ===")
    print(f"split table       : {OUT_SPLIT_TABLE}")
    print(f"class table       : {OUT_CLASS_TABLE}")
    print(f"cleaning table    : {OUT_CLEANING_TABLE}")
    print(f"image grain table : {OUT_IMAGE_GRAIN_TABLE}")
    print(f"augmentation table: {OUT_AUG_TABLE}")


if __name__ == "__main__":
    main()