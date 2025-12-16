import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def main():
    # =============================
    # Setup Device
    # =============================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =============================
    # 1. Load train dataset
    # =============================
    data = torch.load("train_dataset.pt", weights_only=False)
    X_train = data["X"]        
    Y_train = data["Y"]         
    term2idx = data["term2idx"]
    idx2term = {v: k for k, v in term2idx.items()}
    num_labels = len(term2idx)
    print("Train shape:", X_train.shape, Y_train.shape)

  
    X_train = X_train.to(device).float()
    Y_train = Y_train.to(device).float()
    X_train = X_train / X_train.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # =============================
    # 2. Load test embeddings
    # =============================
    emb_raw = torch.load(
        "test_embeddings_t33.pt",
        map_location=device,
        weights_only=False
    )
    test_ids = emb_raw["ids"]
    X_test = torch.tensor(emb_raw["features"], dtype=torch.float32, device=device)
    X_test = X_test / X_test.norm(dim=1, keepdim=True).clamp(min=1e-8)
    print("Test shape:", X_test.shape)

    # =============================
    # 3. kNN prediction
    # =============================
    print("Start kNN predicting...")
    rows = []
    batch_size = 256         
    k = 20                   
    threshold = 0.05          

    with torch.no_grad():
        for i in range(0, X_test.size(0), batch_size):
            xb = X_test[i:i+batch_size]                    # (B, D)
            batch_ids = test_ids[i:i+batch_size]

            # cosine similarity: (B, N_train)
            sims = torch.matmul(xb, X_train.T)

            # top-k hàng xóm theo similarity
            k_eff = min(k, sims.size(1))
            vals, idx = torch.topk(sims, k=k_eff, dim=1)   # vals: (B, k), idx: (B, k)

            # bỏ similarity âm
            vals = torch.clamp(vals, min=0.0)
            # chuẩn hóa trọng số
            weight_sum = vals.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = vals / weight_sum                    # (B, k)

            # lấy nhãn hàng xóm: (B, k, T)
            neigh_labels = Y_train[idx]                    # (B, k, num_labels)

            # weighted average theo trục k
            # (B, T) = (B,1,k) @ (B,k,T)
            weights_exp = weights.unsqueeze(1)             # (B, 1, k)
            scores = torch.bmm(weights_exp, neigh_labels).squeeze(1)  # (B, num_labels)

            scores_np = scores.cpu().numpy()

            for acc, p in zip(batch_ids, scores_np):
                hit = np.where(p > threshold)[0]
                for t_idx in hit:
                    rows.append([acc, idx2term[t_idx], float(p[t_idx])])

    # =============================
    # 4. Save submission
    # =============================
    df = pd.DataFrame(rows, columns=["EntryID", "term", "score"])
    df.to_csv("submission_knn.tsv", sep="\t", index=False)
    print("Saved submission_knn.tsv")

if __name__ == "__main__":
    main()
