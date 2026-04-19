from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
OUT_DIR = PROJECT_ROOT / "figures" / "audit_examples" / "leakage_pairs"

NEAR_CSV = META_DIR / "audit_near_duplicates_across_splits.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20  # 取最相似的前20对


def open_and_resize(img_path, target_h=512):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    new_w = int(w * target_h / h)
    return img.resize((new_w, target_h))


def main():
    if not NEAR_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {NEAR_CSV}")

    df = pd.read_csv(NEAR_CSV)
    if df.empty:
        print("near duplicate csv is empty")
        return

    df = df.sort_values(["hamming_distance", "split_a", "split_b"]).head(TOP_K)

    for idx, row in df.iterrows():
        img_a = open_and_resize(row["image_path_a"])
        img_b = open_and_resize(row["image_path_b"])

        gap = 20
        canvas_w = img_a.width + gap + img_b.width
        canvas_h = max(img_a.height, img_b.height) + 50

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
        canvas.paste(img_a, (0, 50))
        canvas.paste(img_b, (img_a.width + gap, 50))

        draw = ImageDraw.Draw(canvas)
        title = f"{row['split_a']}:{row['image_name_a']}  |  {row['split_b']}:{row['image_name_b']}  |  pHash dist={row['hamming_distance']}"
        draw.text((10, 15), title, fill=(255, 255, 0))

        save_path = OUT_DIR / f"pair_{idx:03d}_dist{int(row['hamming_distance'])}.jpg"
        canvas.save(save_path)

    print("=== leakage example figures saved ===")
    print(f"output dir: {OUT_DIR}")
    print(f"num pairs saved: {len(df)}")


if __name__ == "__main__":
    main()