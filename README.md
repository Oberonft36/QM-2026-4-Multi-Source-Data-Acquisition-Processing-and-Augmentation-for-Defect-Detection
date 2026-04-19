# README.md

├─.idea
├─data
│  └─processed
│      └─meta
│         # 处理后的元数据与统计结果目录
│         # 主要存放：
│         # - cleaned_annotations_stage1.csv：清洗后的标注表
│         # - squared_annotations.csv：正方形化后的标注表
│         # - final_annotations_with_split.csv：最终划分后的总标注表
│         # - split_summary.csv：train/val/test 划分统计
│         # - augmentation_log.csv：增强日志
│         # - 各类审计结果 csv（泄漏、大框、漏标候选、捷径特征等）
├─figures
│  ├─audit_examples
│  │  ├─large_boxes
│  │  │  # 可疑大框审计样例
│  │  │  # 用于展示 bbox 过大、覆盖背景过多、疑似粗标注的问题
│  │  │
│  │  ├─leakage_pairs
│  │  │  # 数据泄漏审计样例
│  │  │  # 存放 train/val/test 之间高相似图像对的可视化结果
│  │  │  # 用于说明是否存在潜在软泄漏风险
│  │  │
│  │  ├─missing_label_candidates
│  │  │  # 疑似漏标候选样例
│  │  │  # 由规则筛出的高风险图像，用于人工复核是否存在漏标
│  │  │
│  │——suspect_labels
│  │     # 疑似错标/弱标注样例
│  │     # 对尾部类别或视觉证据较弱类别进行抽样检查的可视化结果
│  │
│  ├─augmentation_examples
│  │  # 数据增强前后对比图
│  │  # 用于展示增强策略（翻转、亮度/对比度扰动、旋转、缩放等）
│  │  # 以及增强后 bbox 是否仍然正确
│  │
│  ├─raw_preview
│  │  # 原始数据抽样可视化结果
│  │  # 用于验证 XML 解析是否正确、原始 bbox 是否越界/偏移
│  │
│  |——square_examples
│     # 正方形化前后对比图
│     # 用于展示 padding 到正方形后，图像与 bbox 是否同步正确更新
│
scripts #核心复现代码
