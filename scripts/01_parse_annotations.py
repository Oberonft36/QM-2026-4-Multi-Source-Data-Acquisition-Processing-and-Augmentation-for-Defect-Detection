from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

# 你解压后的数据主目录名，按实际情况改
# 比如 data/raw/archive 或 data/raw/gc10_det
DATASET_ROOT = RAW_ROOT / "archive"

LABEL_DIR = DATASET_ROOT / "lable"
OUT_CSV = PROJECT_ROOT / "raw_annotation_summary.csv"


def find_image_by_filename(dataset_root: Path, filename: str) -> Path | None:
    """
    在数据集目录下递归查找同名图像。
    只按文件名匹配，不依赖 XML 里的原始绝对路径。
    """
    matches = list(dataset_root.rglob(filename))
    if not matches:
        return None

    # 优先排除 lable 目录中的非图像误匹配
    image_matches = [p for p in matches if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    if image_matches:
        return image_matches[0]

    return matches[0]


def safe_int(node, tag: str, default=None):
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return int(float(child.text.strip()))
    except Exception:
        return default


def validate_box(xmin, ymin, xmax, ymax, width, height):
    if None in [xmin, ymin, xmax, ymax, width, height]:
        return False
    if xmin < 0 or ymin < 0:
        return False
    if xmax > width or ymax > height:
        return False
    if xmin >= xmax or ymin >= ymax:
        return False
    return True


def collect_xml_files(label_dir: Path):
    return sorted(label_dir.glob("*.xml"))


def build_class_mapping(class_names):
    class_names = sorted(set(class_names))
    return {name: idx for idx, name in enumerate(class_names)}


def parse_single_xml(xml_path: Path, dataset_root: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    width = safe_int(root.find("size"), "width") if root.find("size") is not None else None
    height = safe_int(root.find("size"), "height") if root.find("size") is not None else None

    if not filename:
        return []

    image_path = find_image_by_filename(dataset_root, filename)
    image_path_str = str(image_path) if image_path else ""

    objects = root.findall("object")
    records = []

    # 若 XML 没有 object，这里先不生成伪框记录
    # 原始阶段先只汇总真实框；后面再统计无标注图
    for obj in objects:
        class_name = obj.findtext("name", default="unknown").strip()

        bndbox = obj.find("bndbox")
        if bndbox is None:
            xmin = ymin = xmax = ymax = None
        else:
            xmin = safe_int(bndbox, "xmin")
            ymin = safe_int(bndbox, "ymin")
            xmax = safe_int(bndbox, "xmax")
            ymax = safe_int(bndbox, "ymax")

        is_valid_raw = validate_box(xmin, ymin, xmax, ymax, width, height)

        records.append({
            "xml_file": str(xml_path),
            "image_filename": filename,
            "image_path": image_path_str,
            "image_width": width,
            "image_height": height,
            "class_name": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "source_annotation": "voc_xml",
            "is_valid_raw": is_valid_raw,
        })

    return records


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT 不存在: {DATASET_ROOT}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"LABEL_DIR 不存在: {LABEL_DIR}")

    xml_files = collect_xml_files(LABEL_DIR)
    if not xml_files:
        raise FileNotFoundError(f"在 {LABEL_DIR} 下没有找到 XML 文件")

    all_records = []
    for xml_file in xml_files:
        all_records.extend(parse_single_xml(xml_file, DATASET_ROOT))

    if not all_records:
        raise ValueError("没有解析出任何标注记录，请检查 XML 结构")

    df = pd.DataFrame(all_records)

    class_map = build_class_mapping(df["class_name"].dropna().tolist())
    df["class_id"] = df["class_name"].map(class_map)

    # 调整字段顺序
    df = df[
        [
            "image_path",
            "image_width",
            "image_height",
            "class_name",
            "class_id",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "source_annotation",
            "is_valid_raw",
            "image_filename",
            "xml_file",
        ]
    ]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=== parse finished ===")
    print(f"xml files: {len(xml_files)}")
    print(f"records : {len(df)}")
    print(f"images  : {df['image_path'].nunique()}")
    print(f"classes : {df['class_name'].nunique()}")
    print(f"saved   : {OUT_CSV}")

    print("\n=== class mapping ===")
    for name, idx in class_map.items():
        print(f"{idx}: {name}")

    invalid_count = (~df["is_valid_raw"]).sum()
    missing_path_count = (df["image_path"].astype(str).str.len() == 0).sum()

    print("\n=== quick check ===")
    print(f"invalid boxes     : {invalid_count}")
    print(f"missing image path: {missing_path_count}")


if __name__ == "__main__":
    main()