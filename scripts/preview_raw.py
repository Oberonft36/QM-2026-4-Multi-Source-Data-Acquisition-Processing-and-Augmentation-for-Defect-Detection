from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "raw_annotation_summary.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "raw_preview"

# 抽样数量
NUM_SAMPLES = 20

# 固定随机种子，保证每次抽到同一批，可复现啦啦啦啦啦啦啦啦啦
RANDOM_SEED = 42


def draw_one_image(image_path: Path, ann_df: pd.DataFrame, save_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for _, row in ann_df.iterrows():
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        class_name = str(row["class_name"])

        # 画框
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        # 类别文字
        text_pos = (xmin, max(0, ymin - 15))
        draw.text(text_pos, class_name, fill="yellow")

    img.save(save_path)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 CSV: {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 只保留路径存在、bbox 合法的记录
    df = df[df["image_path"].notna()].copy()
    df = df[df["image_path"].astype(str).str.len() > 0].copy()
    df = df[df["is_valid_raw"] == True].copy()

    if df.empty:
        raise ValueError("没有可用于可视化的有效标注记录")

    unique_images = df["image_path"].unique().tolist()

    random.seed(RANDOM_SEED)
    sample_images = random.sample(unique_images, min(NUM_SAMPLES, len(unique_images)))

    saved = 0
    failed = []

    for img_path_str in sample_images:
        img_path = Path(img_path_str)

        try:
            one_img_df = df[df["image_path"] == img_path_str].copy()
            save_name = img_path.stem + "_preview.jpg"
            save_path = OUT_DIR / save_name

            draw_one_image(img_path, one_img_df, save_path)
            saved += 1

        except Exception as e:
            failed.append((img_path_str, str(e)))

    print("=== raw preview finished ===")
    print(f"sampled images : {len(sample_images)}")
    print(f"saved previews : {saved}")
    print(f"output dir     : {OUT_DIR}")

    if failed:
        print("\n=== failed images ===")
        for path, err in failed:
            print(f"{path} -> {err}")


if __name__ == "__main__":
    main()