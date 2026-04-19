from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "cleaned_annotations_stage1.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "squared_annotations.csv"

OUT_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "all"
EXAMPLE_DIR = PROJECT_ROOT / "figures" / "square_examples"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_EXAMPLES = 12


def compute_padding(width: int, height: int):
    """
    把任意矩形 padding 成正方形。
    返回:
        side, pad_left, pad_top, pad_right, pad_bottom
    """
    side = max(width, height)

    pad_w = side - width
    pad_h = side - height

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    return side, pad_left, pad_top, pad_right, pad_bottom


def draw_boxes(img: Image.Image, ann_df: pd.DataFrame, color="red"):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for _, row in ann_df.iterrows():
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        class_name = str(row["class_name"])

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
        draw.text((xmin, max(0, ymin - 15)), class_name, fill="yellow")

    return img


def make_before_after_example(
    orig_img: Image.Image,
    orig_ann: pd.DataFrame,
    squared_img: Image.Image,
    squared_ann: pd.DataFrame,
    save_path: Path
):
    """
    左边原图+原框，右边正方形图+更新后框
    """
    left = draw_boxes(orig_img, orig_ann, color="red")
    right = draw_boxes(squared_img, squared_ann, color="lime")

    gap = 20
    canvas_w = left.width + gap + right.width
    canvas_h = max(left.height, right.height)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(save_path)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 去掉无路径记录
    df = df[df["image_path"].notna()].copy()
    df = df[df["image_path"].astype(str).str.len() > 0].copy()

    unique_images = df["image_path"].unique().tolist()
    random.seed(RANDOM_SEED)
    example_images = set(random.sample(unique_images, min(NUM_EXAMPLES, len(unique_images))))

    new_records = []
    failed = []

    for img_path_str in unique_images:
        img_path = Path(img_path_str)

        try:
            ann = df[df["image_path"] == img_path_str].copy()

            orig_img = Image.open(img_path).convert("RGB")
            width, height = orig_img.size

            side, pad_left, pad_top, pad_right, pad_bottom = compute_padding(width, height)

            squared_img = ImageOps.expand(
                orig_img,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=(0, 0, 0)
            )

            # 保存新图
            out_name = img_path.stem + ".jpg"
            out_path = OUT_IMG_DIR / out_name
            squared_img.save(out_path)

            # 更新标注
            ann["orig_image_width"] = width
            ann["orig_image_height"] = height
            ann["square_size"] = side

            ann["pad_left"] = pad_left
            ann["pad_top"] = pad_top
            ann["pad_right"] = pad_right
            ann["pad_bottom"] = pad_bottom

            ann["xmin"] = ann["xmin"] + pad_left
            ann["xmax"] = ann["xmax"] + pad_left
            ann["ymin"] = ann["ymin"] + pad_top
            ann["ymax"] = ann["ymax"] + pad_top

            ann["image_width"] = side
            ann["image_height"] = side
            ann["image_path"] = str(out_path)

            # 再做一次合法性检查
            valid_mask = (
                (ann["xmin"] >= 0) &
                (ann["ymin"] >= 0) &
                (ann["xmax"] <= ann["image_width"]) &
                (ann["ymax"] <= ann["image_height"]) &
                (ann["xmin"] < ann["xmax"]) &
                (ann["ymin"] < ann["ymax"])
            )
            ann["is_valid_squared"] = valid_mask

            new_records.append(ann)

            # 保存示例图
            if img_path_str in example_images:
                example_save = EXAMPLE_DIR / f"{img_path.stem}_before_after.jpg"
                make_before_after_example(
                    orig_img=orig_img,
                    orig_ann=df[df["image_path"] == img_path_str].copy(),
                    squared_img=squared_img,
                    squared_ann=ann.copy(),
                    save_path=example_save
                )

        except Exception as e:
            failed.append((img_path_str, str(e)))

    if not new_records:
        raise ValueError("没有生成任何正方形化结果")

    out_df = pd.concat(new_records, ignore_index=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=== square padding finished ===")
    print(f"input images      : {len(unique_images)}")
    print(f"output rows       : {len(out_df)}")
    print(f"output images dir : {OUT_IMG_DIR}")
    print(f"output csv        : {OUTPUT_CSV}")
    print(f"example dir       : {EXAMPLE_DIR}")
    print(f"invalid after pad : {(~out_df['is_valid_squared']).sum()}")

    if failed:
        print("\n=== failed images ===")
        for path, err in failed[:20]:
            print(f"{path} -> {err}")
        print(f"... total failed: {len(failed)}")


if __name__ == "__main__":
    main()