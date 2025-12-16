import pandas as pd
import numpy as np

# ==========================================
# CẤU HÌNH TRỌNG SỐ\
# ==========================================
# Tổng trọng số nên bằng 1.0
W_LOGISTIC = 0.1
W_KNN = 0.4
W_MLP = 0.5

FILES = {
    "logistic": "submission_logistics.tsv",
    "knn": "submission_knn.tsv",         
    "mlp": "submission_mlp.tsv"         
}

OUTPUT_FILE = "submission_ensemble.tsv"

def load_submission(filepath, weight):
    print(f"Loading {filepath} with weight {weight}...")
  
    try:
        df = pd.read_csv(filepath, sep="\t")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None
    
    # Chuẩn hóa tên cột
    df.columns = ["EntryID", "Term", "Score"]
    
    # Nhân điểm số với trọng số
    df["Score"] = df["Score"] * weight
    return df

def main():
    dfs = []
    
    # 1. Đọc các file
    if W_LOGISTIC > 0:
        dfs.append(load_submission(FILES["logistic"], W_LOGISTIC))
    if W_KNN > 0:
        dfs.append(load_submission(FILES["knn"], W_KNN))
    if W_MLP > 0:
        dfs.append(load_submission(FILES["mlp"], W_MLP))
    
    dfs = [d for d in dfs if d is not None]
    
    if not dfs:
        print("Không có dữ liệu để ensemble!")
        return

    print("Merging data...")
    # Gộp tất cả lại thành 1 DataFrame lớn
    full_df = pd.concat(dfs)
    
    # Cộng gộp điểm số 
    # Nếu một ID-Term xuất hiện ở cả 3 model, điểm sẽ được cộng lại
    ensemble_df = full_df.groupby(["EntryID", "Term"], as_index=False)["Score"].sum()
    
    # 4. Sắp xếp và Lọc (Tùy chọn: Giữ lại Top K nếu file quá lớn)
    ensemble_df = ensemble_df.sort_values(by=["EntryID", "Score"], ascending=[True, False])
    
    # 5. Lưu kết quả
    print(f"Writing ensemble result to {OUTPUT_FILE}...")
    ensemble_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    main()