from pathlib import Path
import hashlib
from itertools import product

import pandas as pd
from PIL import Image
import imagehash

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMG_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
META_DIR = PROJECT_ROOT / "data" / "processed" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXACT_CSV = META_DIR / "audit_exact_duplicates_across_splits.csv"
OUT_PHASH_CSV = META_DIR / "audit_near_duplicates_across_splits.csv"


def md5_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_split_images(split: str):
    return sorted((IMG_ROOT / split).glob("*.jpg"))


def build_exact_hash_table(image_paths, split_name):
    rows = []
    for p in image_paths:
        try:
            rows.append({
                "split": split_name,
                "image_name": p.name,
                "image_path": str(p),
                "md5": md5_file(p)
            })
        except Exception as e:
            rows.append({
                "split": split_name,
                "image_name": p.name,
                "image_path": str(p),
                "md5": f"ERROR:{e}"
            })
    return pd.DataFrame(rows)


def build_phash_table(image_paths, split_name):
    rows = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("L")
            ph = imagehash.phash(img)
            rows.append({
                "split": split_name,
                "image_name": p.name,
                "image_path": str(p),
                "phash": str(ph)
            })
        except Exception as e:
            rows.append({
                "split": split_name,
                "image_name": p.name,
                "image_path": str(p),
                "phash": f"ERROR:{e}"
            })
    return pd.DataFrame(rows)


def hamming_distance(hash1: str, hash2: str):
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)


def main():
    train_imgs = get_split_images("train")
    val_imgs = get_split_images("val")
    test_imgs = get_split_images("test")

    print("=== build exact hash tables ===")
    train_exact = build_exact_hash_table(train_imgs, "train")
    val_exact = build_exact_hash_table(val_imgs, "val")
    test_exact = build_exact_hash_table(test_imgs, "test")

    all_exact = pd.concat([train_exact, val_exact, test_exact], ignore_index=True)

    # 只看跨 split 的相同 md5
    exact_dup = all_exact.merge(all_exact, on="md5")
    exact_dup = exact_dup[exact_dup["image_path_x"] < exact_dup["image_path_y"]].copy()
    exact_dup = exact_dup[exact_dup["split_x"] != exact_dup["split_y"]].copy()

    exact_dup = exact_dup[
        ["split_x", "image_name_x", "image_path_x", "split_y", "image_name_y", "image_path_y", "md5"]
    ].rename(columns={
        "split_x": "split_a",
        "image_name_x": "image_name_a",
        "image_path_x": "image_path_a",
        "split_y": "split_b",
        "image_name_y": "image_name_b",
        "image_path_y": "image_path_b",
    })

    exact_dup.to_csv(OUT_EXACT_CSV, index=False, encoding="utf-8-sig")

    print(f"exact duplicates across splits: {len(exact_dup)}")
    print(f"saved: {OUT_EXACT_CSV}")

    print("\n=== build pHash tables ===")
    train_ph = build_phash_table(train_imgs, "train")
    val_ph = build_phash_table(val_imgs, "val")
    test_ph = build_phash_table(test_imgs, "test")

    split_tables = {
        "train": train_ph,
        "val": val_ph,
        "test": test_ph,
    }

    # 只比较 train-val, train-test, val-test
    pairs_to_check = [("train", "val"), ("train", "test"), ("val", "test")]
    near_rows = []

    PHASH_THRESHOLD = 4

    for split_a, split_b in pairs_to_check:
        df_a = split_tables[split_a]
        df_b = split_tables[split_b]

        records_a = df_a.to_dict("records")
        records_b = df_b.to_dict("records")

        for ra, rb in product(records_a, records_b):
            if str(ra["phash"]).startswith("ERROR") or str(rb["phash"]).startswith("ERROR"):
                continue

            dist = hamming_distance(ra["phash"], rb["phash"])
            if dist <= PHASH_THRESHOLD:
                near_rows.append({
                    "split_a": ra["split"],
                    "image_name_a": ra["image_name"],
                    "image_path_a": ra["image_path"],
                    "split_b": rb["split"],
                    "image_name_b": rb["image_name"],
                    "image_path_b": rb["image_path"],
                    "phash_a": ra["phash"],
                    "phash_b": rb["phash"],
                    "hamming_distance": dist
                })

    near_df = pd.DataFrame(near_rows)
    if len(near_df) > 0:
        near_df = near_df.sort_values(["hamming_distance", "split_a", "split_b"])

    near_df.to_csv(OUT_PHASH_CSV, index=False, encoding="utf-8-sig")

    print(f"near duplicates across splits (pHash <= 4): {len(near_df)}")
    print(f"saved: {OUT_PHASH_CSV}")


if __name__ == "__main__":
    main()