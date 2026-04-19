from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "final_annotations_with_split.csv"
FLAGGED_LARGE_CSV = PROJECT_ROOT / "data" / "processed" / "meta" / "flagged_large_boxes.csv"

OUT_DIR = PROJECT_ROOT / "figures" / "audit_examples" / "missing_label_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
OUT_CANDIDATE_CSV = OUT_META_DIR / "audit_missing_label_candidates.csv"


def draw_boxes(img_path: Path, ann_df: pd.DataFrame, save_path: Path, note: str):
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

    draw.text((10, 10), note, fill="cyan")
    img.save(save_path)


def main():
    final_df = pd.read_csv(FINAL_CSV)
    large_df = pd.read_csv(FLAGGED_LARGE_CSV) if FLAGGED_LARGE_CSV.exists() else pd.DataFrame()

    # 图像级统计
    img_stats = final_df.groupby(["image_path", "split"]).agg(
        num_boxes=("class_name", "count")
    ).reset_index()

    # 规则1：一张图里框很多（>=4），更容易漏标
    dense_imgs = img_stats[img_stats["num_boxes"] >= 4].copy()
    dense_imgs["risk_reason"] = "many_boxes_in_image"

    # 规则2：包含大框的图，粗标注与漏标风险更高
    if not large_df.empty:
        large_img_paths = large_df["image_path"].dropna().unique().tolist()
        large_imgs = img_stats[img_stats["image_path"].isin(large_img_paths)].copy()
        large_imgs["risk_reason"] = "contains_large_box"
    else:
        large_imgs = pd.DataFrame(columns=img_stats.columns.tolist() + ["risk_reason"])

    candidate_df = pd.concat([dense_imgs, large_imgs], ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset=["image_path", "split"]).copy()

    # 合并风险标签
    reason_map = candidate_df.groupby(["image_path", "split"])["risk_reason"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
    candidate_df = img_stats.merge(reason_map, on=["image_path", "split"], how="inner")

    candidate_df = candidate_df.sort_values(["num_boxes", "risk_reason"], ascending=[False, True])
    candidate_df.to_csv(OUT_CANDIDATE_CSV, index=False, encoding="utf-8-sig")

    # 导出前30张候选图供人工审查
    top_n = candidate_df.head(30).copy()

    for idx, row in top_n.iterrows():
        img_path = Path(row["image_path"])
        ann = final_df[final_df["image_path"] == row["image_path"]].copy()

        note = f"split={row['split']} | boxes={row['num_boxes']} | risk={row['risk_reason']}"
        save_path = OUT_DIR / f"{idx:03d}_{img_path.stem}.jpg"
        draw_boxes(img_path, ann, save_path, note)

    print("=== missing-label candidate audit finished ===")
    print(f"candidate csv : {OUT_CANDIDATE_CSV}")
    print(f"example dir   : {OUT_DIR}")
    print(f"num candidates: {len(candidate_df)}")
    print(f"top exported  : {len(top_n)}")


if __name__ == "__main__":
    main()