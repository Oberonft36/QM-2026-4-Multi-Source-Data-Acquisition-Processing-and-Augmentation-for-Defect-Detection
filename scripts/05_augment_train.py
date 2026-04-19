from pathlib import Path
import random
import shutil

import cv2
import pandas as pd
import albumentations as A

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
TRAIN_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"

AUG_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train_aug"
AUG_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train_aug"
EXAMPLE_DIR = PROJECT_ROOT / "figures" / "augmentation_examples"
META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"

AUG_LOG_CSV = META_DIR / "augmentation_log.csv"

AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
AUG_LBL_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_EXAMPLE_SAVE = 12

# 每张 train 图生成 1 张增强图
AUG_PER_IMAGE = 1

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.7
        ),
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-8, 8),
            translate_percent=(0.0, 0.03),
            shear=(-3, 3),
            p=0.8
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)


def load_yolo_labels(label_path: Path):
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        bboxes.append([xc, yc, w, h])
        class_labels.append(class_id)

    return bboxes, class_labels


def save_yolo_labels(label_path: Path, bboxes, class_labels):
    lines = []
    for cls, box in zip(class_labels, bboxes):
        xc, yc, w, h = box
        if not (0 < w <= 1 and 0 < h <= 1):
            continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1):
            continue
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def draw_yolo_boxes(img, bboxes, class_labels):
    out = img.copy()
    h, w = out.shape[:2]

    for cls, box in zip(class_labels, bboxes):
        xc, yc, bw, bh = box

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out, str(cls), (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

    return out


def make_example(original_img, original_boxes, original_labels,
                 aug_img, aug_boxes, aug_labels, save_path: Path):
    left = draw_yolo_boxes(original_img, original_boxes, original_labels)
    right = draw_yolo_boxes(aug_img, aug_boxes, aug_labels)

    gap = 20
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + gap

    canvas = 255 * (cv2.UMat(h, w, cv2.CV_8UC3).get() * 0)
    canvas[:] = (30, 30, 30)
    canvas[0:left.shape[0], 0:left.shape[1]] = left
    canvas[0:right.shape[0], left.shape[1] + gap:left.shape[1] + gap + right.shape[1]] = right

    cv2.imwrite(str(save_path), canvas)


def main():
    random.seed(RANDOM_SEED)

    img_paths = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
    example_candidates = set(random.sample(img_paths, min(NUM_EXAMPLE_SAVE, len(img_paths))))

    logs = []
    success = 0
    skipped = 0

    for img_path in img_paths:
        label_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            logs.append({
                "image_name": img_path.name,
                "status": "skip_image_read_failed",
                "num_boxes_before": None,
                "num_boxes_after": None,
            })
            continue

        bboxes, class_labels = load_yolo_labels(label_path)

        if len(bboxes) == 0:
            skipped += 1
            logs.append({
                "image_name": img_path.name,
                "status": "skip_empty_label",
                "num_boxes_before": 0,
                "num_boxes_after": 0,
            })
            continue

        for aug_idx in range(AUG_PER_IMAGE):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_img = augmented["image"]
                aug_boxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                # 增强后必须仍有至少 1 个有效框
                if len(aug_boxes) == 0:
                    skipped += 1
                    logs.append({
                        "image_name": img_path.name,
                        "status": "skip_no_boxes_after_aug",
                        "num_boxes_before": len(bboxes),
                        "num_boxes_after": 0,
                    })
                    continue

                out_img_name = f"{img_path.stem}_aug{aug_idx}.jpg"
                out_lbl_name = f"{img_path.stem}_aug{aug_idx}.txt"

                out_img_path = AUG_IMG_DIR / out_img_name
                out_lbl_path = AUG_LBL_DIR / out_lbl_name

                cv2.imwrite(str(out_img_path), aug_img)
                save_yolo_labels(out_lbl_path, aug_boxes, aug_labels)

                # 再检查标签文件是否为空
                if out_lbl_path.stat().st_size == 0:
                    if out_img_path.exists():
                        out_img_path.unlink()
                    if out_lbl_path.exists():
                        out_lbl_path.unlink()

                    skipped += 1
                    logs.append({
                        "image_name": img_path.name,
                        "status": "skip_empty_label_after_save",
                        "num_boxes_before": len(bboxes),
                        "num_boxes_after": 0,
                    })
                    continue

                success += 1
                logs.append({
                    "image_name": img_path.name,
                    "status": "success",
                    "num_boxes_before": len(bboxes),
                    "num_boxes_after": len(aug_boxes),
                })

                if img_path in example_candidates:
                    example_path = EXAMPLE_DIR / f"{img_path.stem}_aug_compare.jpg"
                    make_example(
                        original_img=image,
                        original_boxes=bboxes,
                        original_labels=class_labels,
                        aug_img=aug_img,
                        aug_boxes=aug_boxes,
                        aug_labels=aug_labels,
                        save_path=example_path
                    )

            except Exception as e:
                skipped += 1
                logs.append({
                    "image_name": img_path.name,
                    "status": f"error:{str(e)}",
                    "num_boxes_before": len(bboxes),
                    "num_boxes_after": None,
                })

    log_df = pd.DataFrame(logs)
    log_df.to_csv(AUG_LOG_CSV, index=False, encoding="utf-8-sig")

    print("=== augmentation finished ===")
    print(f"train images scanned : {len(img_paths)}")
    print(f"augmented success    : {success}")
    print(f"skipped / failed     : {skipped}")
    print(f"aug image dir        : {AUG_IMG_DIR}")
    print(f"aug label dir        : {AUG_LBL_DIR}")
    print(f"example dir          : {EXAMPLE_DIR}")
    print(f"log csv              : {AUG_LOG_CSV}")


if __name__ == "__main__":
    main()