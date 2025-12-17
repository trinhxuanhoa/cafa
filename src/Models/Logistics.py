import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

@dataclass
class Config:
    DATA_DIR: str = "/kaggle/input/hahaha/cafa6/"
    OUTPUT_SUBMISSION: str = "/kaggle/working/submission.tsv"
    TRAIN_TERMS: str = "train_terms.tsv"
    IA_TSV: str = "IA.tsv" [cite: 2]
    
    INPUT_PROCESSED: str = "processed_data.pkl" # Đọc file từ bước 1

    # ontology/labels
    MIN_TERM_FREQ: int = 10
    MAX_TERMS_PER_ASPECT: int = 1000
    ASPECTS: tuple = ("F", "P", "C") [cite: 3]

    # ia weighting
    IA_MIN_W: float = 0.0
    IA_MAX_W: float = 1.0
    IA_ALPHA: float = 1.0

    # train
    RANDOM_STATE: int = 42
    BATCH_SIZE: int = 4096
    EPOCHS_PER_ASPECT: int = 15
    LR: float = 1e-2           
    MOMENTUM: float = 0.9        
    WEIGHT_DECAY: float = 1e-4   

    # Filtering
    MIN_PRED_SCORE: float = 0.02
    TOP_K_PER_PROTEIN: int = 50

CFG = Config()

# Gán nhãn và tạo ia-weight
def load_train_terms(path):
    df = pd.read_csv(path, sep="\t", header=0)
    df.columns = ["EntryID", "Term", "Aspect"]
    df["EntryID"] = df["EntryID"].astype(str)
    return df[df["Aspect"].isin(["F", "P", "C"])].copy()

def select_terms_for_aspect(df_terms_aspect, cfg):
    term_counts = df_terms_aspect["Term"].value_counts()
    term_counts = term_counts[term_counts >= cfg.MIN_TERM_FREQ] # [cite: 23]
    if len(term_counts) > cfg.MAX_TERMS_PER_ASPECT:
        term_counts = term_counts.iloc[:cfg.MAX_TERMS_PER_ASPECT]
    return list(term_counts.index)

def build_label_matrix(entry_ids, df_terms_aspect, selected_terms): # [cite: 24]
    term2idx = {t: i for i, t in enumerate(selected_terms)}
    pid2idx = {pid: i for i, pid in enumerate(entry_ids)}
    Y = np.zeros((len(entry_ids), len(selected_terms)), dtype=np.int8)
    
    for _, row in df_terms_aspect.iterrows():
        pid, term = row["EntryID"], row["Term"]
        if pid in pid2idx and term in term2idx:
            Y[pid2idx[pid], term2idx[term]] = 1
    return Y

def load_ia_scores(path): # [cite: 25]
    df = pd.read_csv(path, sep="\t", header=None, names=["Term", "IA"])
    return dict(zip(df["Term"].astype(str), df["IA"].astype(float)))

def build_ia_weights_for_terms(selected_terms, ia_map, cfg):
    if not selected_terms: return np.array([], dtype=np.float32)
    ia_vals = np.array([ia_map.get(t, 0.0) for t in selected_terms], dtype=np.float32)
    ia_norm = (ia_vals - ia_vals.min()) / (ia_vals.max() - ia_vals.min() + 1e-8)
    return cfg.IA_MIN_W + cfg.IA_ALPHA * ia_norm * (cfg.IA_MAX_W - cfg.IA_MIN_W)

# MLP
class MultiLabelLogistic(nn.Module): # [cite: 26]
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)
    def forward(self, x):
        return self.linear(x)

def train_aspect_torch(aspect, X_train_tensor, Y_a, ia_weights, cfg, device):
    n_samples, n_terms = Y_a.shape
    print(f"[Torch] Training aspect {aspect}: {n_samples} samples, {n_terms} terms") # [cite: 27]

    dataset = TensorDataset(X_train_tensor, torch.from_numpy(Y_a.astype(np.float32)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - int(len(dataset)*0.1), int(len(dataset)*0.1)], 
                                    generator=torch.Generator().manual_seed(cfg.RANDOM_STATE))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False) # [cite: 28]

    model = MultiLabelLogistic(X_train_tensor.shape[1], n_terms).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(ia_weights).float().to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY) # [cite: 28]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True) # [cite: 29]

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(cfg.EPOCHS_PER_ASPECT):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step() # [cite: 31]
            total_loss += loss.item() * xb.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
        
        avg_val = val_loss / len(val_ds)
        print(f"Epoch {epoch+1}, val_loss={avg_val:.4f}") # [cite: 33]
        scheduler.step(avg_val)

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4: break # Early stopping [cite: 34]
    
    if best_state: model.load_state_dict(best_state)
    return model

def predict_aspect_torch(model, X_test_tensor, cfg, device): # [cite: 35]
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_test_tensor.shape[0], cfg.BATCH_SIZE):
            xb = X_test_tensor[i:i + cfg.BATCH_SIZE].to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy()) [cite: 36]
    return np.vstack(preds)

def write_submission(aspect, preds, terms, test_ids, cfg, out_path):
    buffer = []
    print(f"Writing Top-{cfg.TOP_K_PER_PROTEIN} for {aspect}...")
    for i, pid in enumerate(test_ids):
        scores = preds[i]
        # Filter logic
        valid_idx = np.where(scores >= cfg.MIN_PRED_SCORE)[0] if cfg.MIN_PRED_SCORE > 0 else np.arange(len(scores))
        if valid_idx.size == 0: valid_idx = np.array([np.argmax(scores)]) [cite: 37]
        
        # Sort & Select Top K
        top_k = valid_idx[np.argsort(-scores[valid_idx])[:cfg.TOP_K_PER_PROTEIN]] # [cite: 38]
        for j in top_k:
            buffer.append((pid, terms[j], float(scores[j])))
        
        if len(buffer) >= 1_000_000:
            pd.DataFrame(buffer).to_csv(out_path, sep="\t", index=False, mode="a", header=False)
            buffer = []
            
    if buffer:
        pd.DataFrame(buffer).to_csv(out_path, sep="\t", index=False, mode="a", header=False) # [cite: 39]

# main 2
if __name__ == "__main__":
    np.random.seed(CFG.RANDOM_STATE)
    torch.manual_seed(CFG.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load data
    print(f"=== Loading processed data from {CFG.INPUT_PROCESSED} ===")
    with open(CFG.INPUT_PROCESSED, "rb") as f:
        data = pickle.load(f)
    X_train_tensor = torch.from_numpy(data["X_train"]).float()
    X_test_tensor = torch.from_numpy(data["X_test"]).float()
    train_entry_ids = data["train_entry_ids"]
    test_entry_ids = data["test_entry_ids"]

    # tạo file submission 
    if os.path.exists(CFG.OUTPUT_SUBMISSION): os.remove(CFG.OUTPUT_SUBMISSION)
    with open(CFG.OUTPUT_SUBMISSION, "w") as f: f.write("EntryID\tTerm\tScore\n")

    # load metadata
    df_terms = load_train_terms(os.path.join(CFG.DATA_DIR, CFG.TRAIN_TERMS)) [cite: 43]
    ia_map = load_ia_scores(os.path.join(CFG.DATA_DIR, CFG.IA_TSV))

    # loop
    for aspect in CFG.ASPECTS: # [cite: 44]
        print(f"\n========== Aspect {aspect} ==========")
        df_a = df_terms[df_terms["Aspect"] == aspect]
        if df_a.empty: continue
        
        selected_terms = select_terms_for_aspect(df_a, CFG)
        Y_a = build_label_matrix(train_entry_ids, df_a, selected_terms)

        # bỏ cột âm
        col_pos = Y_a.sum(axis=0) > 0
        if not col_pos.any(): continue # [cite: 45]
        Y_a = Y_a[:, col_pos]
        selected_terms = [t for t, k in zip(selected_terms, col_pos) if k]

        # train
        ia_weights = build_ia_weights_for_terms(selected_terms, ia_map, CFG)
        model = train_aspect_torch(aspect, X_train_tensor.to(device), Y_a, ia_weights, CFG, device) [cite: 46]
        preds = predict_aspect_torch(model, X_test_tensor.to(device), CFG, device) [cite: 47]

        # viết submission
        write_submission(aspect, preds, selected_terms, test_entry_ids, CFG, CFG.OUTPUT_SUBMISSION)

    print(f"\n=== Done. File output: {CFG.OUTPUT_SUBMISSION} ===") # [cite: 49]
