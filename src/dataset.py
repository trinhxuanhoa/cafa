import torch
import pandas as pd

# ==== Load embeddings ==== code chạy trên kaggle
emb = torch.load("/kaggle/input/version2/cafa_project/data/embeddings/train_embeddings_t33.pt") 

# Convert key: sp|A0A0...|Name → A0A0...
emb2 = {}
for full_id, vec in emb.items():
    acc = full_id.split("|")[1]
    emb2[acc] = vec

print("Total embeddings:", len(emb2))

# ==== Load labels ====
df = pd.read_csv("/kaggle/input/cafa-project/cafa_project/data/raw/train_terms.tsv", sep="\t")

# List of all GO terms
all_terms = sorted(df["term"].unique())
term2idx = {t: i for i, t in enumerate(all_terms)}
num_labels = len(all_terms)

print("Total GO terms:", num_labels)

# Build dataset
X = []
Y = []

missing = 0

for acc, group in df.groupby("EntryID"):
    if acc not in emb2:
        missing += 1
        continue

    vec = emb2[acc]
    labels = torch.zeros(num_labels)

    for go in group["term"]:
        labels[term2idx[go]] = 1

    X.append(vec)
    Y.append(labels)

print("Missing:", missing)
print("Usable:", len(X))

X = torch.stack(X)
Y = torch.stack(Y)

torch.save({"X": X, "Y": Y, "term2idx": term2idx}, "train_dataset.pt")
print("Saved train_dataset.pt")
