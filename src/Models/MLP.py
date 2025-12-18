import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim=1280, hidden1=1024, hidden2=512, output_dim=0, dropout=0.3):
        # output_dim mặc định là 0, truyền vào khi khởi tạo
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden2, output_dim)
        
    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

def main():
    #chỉnh device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # load dataset
    data = torch.load("train_dataset.pt", weights_only=False) 
    X = data["X"]
    Y = data["Y"]
    term2idx = data["term2idx"]
    idx2term = {v: k for k, v in term2idx.items()}
    num_labels = len(term2idx)
    train_ds = TensorDataset(X, Y)   
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    model = MLP(output_dim=num_labels).to(device)

    # train
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 15

    print("Start training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dl):.4f}")
        
    torch.save({"model": model.state_dict(), "term2idx": term2idx}, "best_model.pt")
    print("Saved best_model.pt")

    #load embedding
    emb_raw = torch.load(
        "test_embeddings_t33.pt", 
        map_location=device,
        weights_only=False
    )
    test_ids = emb_raw["ids"]
    test_features = torch.tensor(emb_raw["features"], dtype=torch.float32).to(device)

    # batch prediction
    print("Start predicting...")
    model.eval()
    rows = []
    batch_size = 1024
    threshold = 0.05

    with torch.no_grad():
        for i in range(0, len(test_features), batch_size):
            batch = test_features[i:i+batch_size]
            batch_ids = test_ids[i:i+batch_size]
          
            preds = torch.sigmoid(model(batch)).cpu().numpy()
            
            for acc, p in zip(batch_ids, preds):
                hit = np.where(p > threshold)[0]
                for idx in hit:
                    rows.append([acc, idx2term[idx], float(p[idx])])

    # lưu sub
    df = pd.DataFrame(rows, columns=["EntryID", "term", "score"])
    df.to_csv("submission.tsv", sep="\t", index=False)
    print("Saved submission.tsv")

if __name__ == "__main__":
    main()
