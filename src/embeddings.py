import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import os
from tqdm import tqdm
MODEL_NAME = "facebook/esm2_t33_650M_UR50D" 

FASTA_FILE = "data/raw/train_sequences.fasta"
SAVE_PATH = "data/embeddings/train_embeddings_t33.pt"

BATCH_SIZE = 16 

MAX_LEN = 1024 
# ---------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print(f"Đang chạy trên thiết bị: {device}")
    

    print(f"Đang tải model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
       
    if device.type == 'cuda':
        model = model.half()
        print("⚡ Đã bật chế độ FP16 (Half Precision) để tăng tốc.")
    
    model = model.to(device)
    model.eval()

    # Xử lý đa GPU 
    if torch.cuda.device_count() > 1:
        print(f"Phát hiện {torch.cuda.device_count()} GPUs. Đang kích hoạt chạy song song.")
        model = torch.nn.DataParallel(model)

    # Đọc dữ liệu
    print("📖 Đang đọc file FASTA...")
    sequences = []
    ids = []
    # Lưu ý: Nếu RAM server yếu (<16GB), đoạn này có thể cần tối ưu đọc từng dòng
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    
    print(f"Tổng số protein cần xử lý: {len(sequences)}")
    
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
                    
            for j, seq_id in enumerate(batch_ids):
                embeddings_dict[seq_id] = batch_embeddings[j].float().cpu()

    print(f" Đang lưu file kết quả vào {SAVE_PATH}...")
    torch.save(embeddings_dict, SAVE_PATH)
    print("Xong")

if __name__ == "__main__":
    main()