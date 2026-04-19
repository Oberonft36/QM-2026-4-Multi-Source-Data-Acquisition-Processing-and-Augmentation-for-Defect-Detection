from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "final_annotations_with_split.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILENAME_CSV = OUT_DIR / "audit_shortcut_filename_patterns.csv"
OUT_IMAGE_SIZE_CSV = OUT_DIR / "audit_shortcut_image_sizes.csv"
OUT_SPLIT_SIZE_CSV = OUT_DIR / "audit_shortcut_split_size_summary.csv"


def extract_prefix(name: str):

    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    elif len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def extract_classlike_id(name: str):
    
    m = re.search(r"img_(\d+)_", name)
    if m:
        return m.group(1)
    return None


def main():
    df = pd.read_csv(FINAL_CSV)

    # 图像级去重
    img_df = df[["image_path", "split", "image_width", "image_height"]].drop_duplicates().copy()
    img_df["image_name"] = img_df["image_path"].apply(lambda p: Path(p).name)
    img_df["filename_prefix"] = img_df["image_name"].apply(extract_prefix)
    img_df["filename_group_id"] = img_df["image_name"].apply(extract_classlike_id)
    img_df["image_size_str"] = img_df["image_width"].astype(str) + "x" + img_df["image_height"].astype(str)

    # 1. 文件名前缀在各 split 的分布
    filename_table = img_df.groupby(["filename_prefix", "split"]).size().unstack(fill_value=0).reset_index()
    if "train" not in filename_table.columns:
        filename_table["train"] = 0
    if "val" not in filename_table.columns:
        filename_table["val"] = 0
    if "test" not in filename_table.columns:
        filename_table["test"] = 0

    filename_table["total"] = filename_table[["train", "val", "test"]].sum(axis=1)
    filename_table["num_splits_present"] = (
        (filename_table["train"] > 0).astype(int)
        + (filename_table["val"] > 0).astype(int)
        + (filename_table["test"] > 0).astype(int)
    )
    filename_table = filename_table.sort_values(["num_splits_present", "total"], ascending=[False, False])
    filename_table.to_csv(OUT_FILENAME_CSV, index=False, encoding="utf-8-sig")

    # 2. 图像尺寸分布
    size_table = img_df.groupby(["split", "image_width", "image_height"]).size().reset_index(name="num_images")
    size_table = size_table.sort_values(["split", "num_images"], ascending=[True, False])
    size_table.to_csv(OUT_IMAGE_SIZE_CSV, index=False, encoding="utf-8-sig")

    # 3. split 级尺寸汇总
    split_size_summary = img_df.groupby("split").agg(
        num_images=("image_path", "count"),
        num_unique_sizes=("image_size_str", "nunique"),
        min_width=("image_width", "min"),
        max_width=("image_width", "max"),
        min_height=("image_height", "min"),
        max_height=("image_height", "max"),
    ).reset_index()
    split_size_summary.to_csv(OUT_SPLIT_SIZE_CSV, index=False, encoding="utf-8-sig")

    print("=== shortcut audit finished ===")
    print(f"filename pattern table : {OUT_FILENAME_CSV}")
    print(f"image size table       : {OUT_IMAGE_SIZE_CSV}")
    print(f"split size summary     : {OUT_SPLIT_SIZE_CSV}")

    print("\n=== quick preview: split size summary ===")
    print(split_size_summary.to_string(index=False))

    print("\n=== quick preview: top filename prefixes ===")
    print(filename_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()