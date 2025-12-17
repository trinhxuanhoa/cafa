import pandas as pd
import numpy as np

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
    
    # chuẩn hóa tên cột
    df.columns = ["EntryID", "Term", "Score"]
    
    df["Score"] = df["Score"] * weight
    return df

def main():
    dfs = []
    
    #load file
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
    # merge lại
    full_df = pd.concat(dfs)
    
    ensemble_df = full_df.groupby(["EntryID", "Term"], as_index=False)["Score"].sum()
    
    ensemble_df = ensemble_df.sort_values(by=["EntryID", "Score"], ascending=[True, False])
    
    # lưu kq
    print(f"Writing ensemble result to {OUTPUT_FILE}...")
    ensemble_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
