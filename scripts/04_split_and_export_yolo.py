from pathlib import Path
import random
import shutil

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "squared_annotations.csv"

IMG_ALL_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "all"
IMG_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
LBL_ROOT = PROJECT_ROOT / "data" / "processed" / "labels"
META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"

SPLIT_SUMMARY_CSV = META_DIR / "split_summary.csv"
FINAL_CSV = META_DIR / "final_annotations_with_split.csv"

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (IMG_ROOT / split).mkdir(parents=True, exist_ok=True)
        (LBL_ROOT / split).mkdir(parents=True, exist_ok=True)


def clear_split_dirs():
    for split in ["train", "val", "test"]:
        for p in (IMG_ROOT / split).glob("*"):
            if p.is_file():
                p.unlink()
        for p in (LBL_ROOT / split).glob("*"):
            if p.is_file():
                p.unlink()


def yolo_convert(xmin, ymin, xmax, ymax, img_w, img_h):
    bw = xmax - xmin
    bh = ymax - ymin
    xc = xmin + bw / 2
    yc = ymin + bh / 2

    return (
        xc / img_w,
        yc / img_h,
        bw / img_w,
        bh / img_h
    )


def assign_split(image_list):
    random.seed(RANDOM_SEED)
    images = image_list[:]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    split_map = {}
    for i, img in enumerate(images):
        if i < n_train:
            split_map[img] = "train"
        elif i < n_train + n_val:
            split_map[img] = "val"
        else:
            split_map[img] = "test"

    return split_map


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {INPUT_CSV}")

    ensure_dirs()
    clear_split_dirs()

    df = pd.read_csv(INPUT_CSV)

    # 只保留正方形化后合法框
    if "is_valid_squared" in df.columns:
        df = df[df["is_valid_squared"] == True].copy()

    unique_images = sorted(df["image_path"].unique().tolist())
    split_map = assign_split(unique_images)

    df["split"] = df["image_path"].map(split_map)

    # 复制图片 + 写 YOLO 标签
    for img_path_str in unique_images:
        img_path = Path(img_path_str)
        split = split_map[img_path_str]

        dst_img_path = IMG_ROOT / split / img_path.name
        shutil.copy2(img_path, dst_img_path)

        one_img_df = df[df["image_path"] == img_path_str].copy()

        label_lines = []
        img_w = int(one_img_df.iloc[0]["image_width"])
        img_h = int(one_img_df.iloc[0]["image_height"])

        for _, row in one_img_df.iterrows():
            class_id = int(row["class_id"])
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            xc, yc, bw, bh = yolo_convert(xmin, ymin, xmax, ymax, img_w, img_h)

            # 安全裁剪到 [0,1]
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        dst_label_path = LBL_ROOT / split / f"{img_path.stem}.txt"
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

    # 保存最终注释表
    df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    # split 统计
    summary = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].copy()
        n_images = sub["image_path"].nunique()
        n_boxes = len(sub)
        avg_boxes = n_boxes / n_images if n_images > 0 else 0

        summary.append({
            "split": split,
            "num_images": n_images,
            "num_boxes": n_boxes,
            "avg_boxes_per_image": round(avg_boxes, 4),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("=== split + yolo export finished ===")
    print(summary_df.to_string(index=False))
    print(f"\nfinal csv: {FINAL_CSV}")
    print(f"split summary: {SPLIT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()