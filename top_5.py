import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ALL_LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
              "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
              "Nodule","Pleural_Thickening","Pneumonia","Pneumothorax",
              "No Finding"]

df = pd.read_csv("df_clean.csv")

# top-5 by positive counts
label_counts = df[ALL_LABELS].sum().sort_values(ascending=False)
top5 = label_counts.head(5).index.tolist()
print("Top 5:", top5)

# keep any row with >=1 of the top5 labels
df_top = df[df[top5].sum(axis=1) >= 1].copy()

# patient split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss.split(df_top, groups=df_top["Patient ID"]))
df_train = df_top.iloc[train_idx].copy()
df_temp  = df_top.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["Patient ID"]))
df_val  = df_temp.iloc[val_idx].copy()
df_test = df_temp.iloc[test_idx].copy()

# keep only what train.py needs
keep_cols = ["img_path", "Patient ID"] + top5
df_train = df_train[keep_cols]
df_val   = df_val[keep_cols]
df_test  = df_test[keep_cols]

df_train.to_csv("train_top5.csv", index=False)
df_val.to_csv("val_top5.csv", index=False)
df_test.to_csv("test_top5.csv", index=False)

pd.Series(top5).to_csv("top5_labels.txt", index=False, header=False)
print("Saved: train_top5.csv, val_top5.csv, test_top5.csv, top5_labels.txt")