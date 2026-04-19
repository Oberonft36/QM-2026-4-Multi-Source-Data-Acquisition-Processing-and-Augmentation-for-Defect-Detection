from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "final_annotations_with_split.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "audit_examples" / "suspect_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 8

# 重点审计类别
TARGET_CLASSES = [
    "8_yahen",
    "9_zhehen",
    "10_yaozhe",
    "6_siban",
]


def draw_image(img_path: Path, ann_df: pd.DataFrame, save_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for _, row in ann_df.iterrows():
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        cname = str(row["class_name"])

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
        draw.text((xmin, max(0, ymin - 16)), cname, fill="yellow")

    img.save(save_path)


def main():
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {FINAL_CSV}")

    random.seed(RANDOM_SEED)
    df = pd.read_csv(FINAL_CSV)

    for class_name in TARGET_CLASSES:
        sub = df[df["class_name"] == class_name].copy()
        image_list = sub["image_path"].dropna().unique().tolist()

        if not image_list:
            continue

        sampled = random.sample(image_list, min(SAMPLES_PER_CLASS, len(image_list)))

        for idx, img_path_str in enumerate(sampled):
            one = sub[sub["image_path"] == img_path_str].copy()
            img_path = Path(img_path_str)

            save_name = f"{class_name}_{idx:02d}_{img_path.stem}.jpg"
            save_path = OUT_DIR / save_name
            draw_image(img_path, one, save_path)

    print("=== suspect label audit figures saved ===")
    print(f"output dir: {OUT_DIR}")
    print(f"classes checked: {TARGET_CLASSES}")


if __name__ == "__main__":
    main()