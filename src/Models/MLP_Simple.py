import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from numpy.core.multiarray import _reconstruct

# ==========================================
# Cấu hình chung
# ==========================================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_model(input_dim, num_labels):
    """
    Định nghĩa kiến trúc model ở một chỗ duy nhất 
    để đảm bảo đồng bộ giữa lúc train và test.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.Linear(2048, num_labels)
    )
    return model

def main():
  
    torch.serialization.add_safe_globals([_reconstruct])
    
    device = get_device()
    print("Device:", device)

    # ==========================================
    # PHẦN 1: TRAINING
    # ==========================================
    print("\n=== START TRAINING ===")
       
    data = torch.load("train_dataset.pt", weights_only=False)
    X = data["X"]
    Y = data["Y"]
    term2idx = data["term2idx"]
    idx2term = {v: k for k, v in term2idx.items()}
    num_labels = len(term2idx)

    train_ds = TensorDataset(X, Y)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    # Khởi tạo model
    model = get_model(1280, num_labels).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 10

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dl):.4f}")

    # Lưu model đã train
    torch.save({"model": model.state_dict(), "term2idx": term2idx}, "model.pt")
    print("Saved model.pt")

    # ==========================================
    # PHẦN 2: PREDICTION 
    # ==========================================
    
    # Chuyển sang chế độ đánh giá
    model.eval()
    
   
    try:
        emb_raw = torch.load(test_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {test_path}")
        return

    ids = emb_raw["ids"]            # list length N
    features = emb_raw["features"]  # numpy array (N, 1280)

    print("IDs count:", len(ids))
    print("Features shape:", features.shape)

    # Convert numpy -> Tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    rows = []
    batch_size = 1024
    threshold = 0.05

    print("Start batch prediction...")
    
    # Batch prediction loop
    with torch.no_grad():
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i:i+batch_size]      # (B, 1280)
            batch_ids = ids[i:i+batch_size]              # list length B

            # Dự đoán
            preds = torch.sigmoid(model(batch)).cpu().numpy()   # (B, num_labels)

            # Lọc kết quả theo threshold
            for acc, p in zip(batch_ids, preds):
                hit = np.where(p > threshold)[0]
                for idx in hit:
                    rows.append([acc, idx2term[idx], float(p[idx])])

    print("Prediction done. Writing output...")

    # Save submission
    df = pd.DataFrame(rows, columns=["EntryID", "term", "score"])
    df.to_csv("submission.tsv", sep="\t", index=False)
    print("Saved submission.tsv")

if __name__ == "__main__":
    main()