import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import os
from tqdm import tqdm

# --- CẤU HÌNH CHO SERVER (HIGH PERFORMANCE) ---
# 1. Chọn Model tốt nhất có thể chạy ổn định (t33 là chuẩn vàng cho CAFA)
# Nếu server bạn là A100 (40GB/80GB VRAM), bạn có thể thử bản "esm2_t36_3B_UR50D" (3 tỷ tham số)
MODEL_NAME = "facebook/esm2_t33_650M_UR50D" 

FASTA_FILE = "data/raw/train_sequences.fasta"
SAVE_PATH = "data/embeddings/train_embeddings_t33.pt"

# Tăng Batch size tùy GPU:
# - GPU 16GB (T4, 3060): Để khoảng 8-16
# - GPU 24GB (3090, 4090): Để khoảng 32
# - GPU 40GB/80GB (A100): Để 64 hoặc cao hơn
BATCH_SIZE = 16 

MAX_LEN = 1024 
# ---------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print(f"🚀 Đang chạy trên thiết bị: {device}")
    
    if device.type == 'cpu':
        print("⚠️ CẢNH BÁO: Bạn đang chạy model lớn trên CPU. Sẽ RẤT CHẬM. Hãy đảm bảo server có GPU.")

    print(f"📥 Đang tải model 'xịn' {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # KỸ THUẬT QUAN TRỌNG: Chuyển sang FP16 (Half Precision)
    # Giúp chạy nhanh hơn và giảm VRAM
    if device.type == 'cuda':
        model = model.half()
        print("⚡ Đã bật chế độ FP16 (Half Precision) để tăng tốc.")
    
    model = model.to(device)
    model.eval()

    # Xử lý đa GPU (Nếu server có nhiều hơn 1 GPU)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Phát hiện {torch.cuda.device_count()} GPUs. Đang kích hoạt chạy song song (DataParallel).")
        model = torch.nn.DataParallel(model)

    # Đọc dữ liệu
    print("📖 Đang đọc file FASTA...")
    sequences = []
    ids = []
    # Lưu ý: Nếu RAM server yếu (<16GB), đoạn này có thể cần tối ưu đọc từng dòng
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    
    print(f"📊 Tổng số protein cần xử lý: {len(sequences)}")
    
    embeddings_dict = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Creating Embeddings"):
            batch_seqs = sequences[i : i + BATCH_SIZE]
            batch_ids = ids[i : i + BATCH_SIZE]

            inputs = tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_LEN
            ).to(device)

            # Lấy output từ model
            outputs = model(**inputs)
            
            # Lấy embedding (Mean Pooling)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mở rộng mask để tính toán đúng kích thước
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Tính tổng và chia trung bình
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask
            
            # Quan trọng: Chuyển về CPU và float32 để lưu trữ an toàn
            # (Giữ file .pt ở dạng float32 để tương thích tốt nhất với code train sau này)
            for j, seq_id in enumerate(batch_ids):
                embeddings_dict[seq_id] = batch_embeddings[j].float().cpu()

    print(f"💾 Đang lưu file kết quả (nặng khoảng 2-4GB) vào {SAVE_PATH}...")
    torch.save(embeddings_dict, SAVE_PATH)
    print("✅ Xong! Bạn đã có bộ embedding chất lượng cao nhất.")

if __name__ == "__main__":
    main()