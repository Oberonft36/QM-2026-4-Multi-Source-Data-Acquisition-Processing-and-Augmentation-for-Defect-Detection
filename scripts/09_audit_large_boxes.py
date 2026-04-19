from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
FLAGGED_CSV = META_DIR / "flagged_large_boxes.csv"

OUT_DIR = PROJECT_ROOT / "figures" / "audit_examples" / "large_boxes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 30  # 导出前30个证据样本


def draw_box(img_path: Path, row, save_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=4)

    text = f"{row['class_name']} | ratio={row['bbox_area_ratio']:.3f}"
    draw.text((xmin, max(0, ymin - 18)), text, fill="yellow")

    img.save(save_path)


def main():
    if not FLAGGED_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {FLAGGED_CSV}")

    df = pd.read_csv(FLAGGED_CSV)
    if df.empty:
        print("flagged_large_boxes.csv is empty")
        return

    df = df.sort_values("bbox_area_ratio", ascending=False).head(TOP_K)

    for i, row in df.iterrows():
        img_path = Path(row["image_path"])
        save_name = f"{i:03d}_{img_path.stem}_{row['class_name']}.jpg"
        save_path = OUT_DIR / save_name
        draw_box(img_path, row, save_path)

    print("=== large box audit figures saved ===")
    print(f"output dir: {OUT_DIR}")
    print(f"num figures saved: {len(df)}")


if __name__ == "__main__":
    main()