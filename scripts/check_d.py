import pandas as pd

df = pd.read_csv(r"D:\project_root\raw_annotation_summary.csv")
print(df["class_name"].value_counts())