import os
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

class Config:
  
    DATA_DIR: str = "/kaggle/input/hahaha/cafa6/"
    
    TRAIN_FASTA: str = "train_sequences.fasta"
    TRAIN_TAXON: str = "train_taxonomy.tsv" [cite: 2]
    TEST_FASTA: str = "testsuperset.fasta"

    # Sequence feature
    AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
    KMER_K: int = 1
    MAX_SEQ_LEN: int = 500
    N_POSITION_BINS: int = 20
    USE_POSITIONAL_1MERS: bool = True

    # Taxon
    MIN_TAXON_COUNT: int = 10
    
    OUTPUT_PROCESSED: str = "processed_data.pkl" # File trung gian

CFG = Config()

def read_train_fasta(path):
    seqs = {}
    with open(path, "r") as f:
        current_id = None
        buf = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(buf) [cite: 6]
                header = line[1:].strip()
                token = header.split()[0]
                if "|" in token:
                    parts = token.split("|")
                    token = parts[1] if (len(parts) >= 2 and parts[1]) else parts[-1] [cite: 7, 8]
                if ":" in token:
                    token = token.split(":")[-1]
                current_id = token
                buf = []
            else:
                buf.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(buf)
    return seqs

def read_test_fasta_with_taxon(path): # [cite: 10]
    seqs, taxon_map = {}, {}
    with open(path, "r") as f:
        current_id = None
        buf = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(buf) [cite: 11]
                parts = line[1:].strip().split()
                entry_id = parts[0]
                taxon_id = parts[1] if len(parts) > 1 else "UNKNOWN"
                current_id = entry_id
                taxon_map[entry_id] = str(taxon_id) [cite: 12]
                buf = []
            else:
                buf.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(buf)
    return sorted(seqs.keys()), seqs, taxon_map

def build_kmer_vocab(amino_acids, k): # [cite: 13]
    from itertools import product
    kmers = ["".join(p) for p in product(amino_acids, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def sequence_to_features(seq, cfg: Config, kmer2idx): # [cite: 14]
    aa2idx = {aa: i for i, aa in enumerate(cfg.AMINO_ACIDS)}
    num_kmers = len(kmer2idx)
    
    # Xử lý sequence feature vectors
    if cfg.USE_POSITIONAL_1MERS:
        pos_dim = cfg.N_POSITION_BINS * len(cfg.AMINO_ACIDS)
        feat_dim = num_kmers + pos_dim
    else:
        feat_dim = num_kmers

    if not seq:
        return np.zeros(feat_dim, dtype=np.float32)

    # Downsample sequence
    L_raw = len(seq)
    stride = math.ceil(L_raw / cfg.MAX_SEQ_LEN) if L_raw > cfg.MAX_SEQ_LEN else 1
    seq_ds = seq[::stride]
    L = len(seq_ds)
    k = cfg.KMER_K
    
    kmer_counts = np.zeros(num_kmers, dtype=np.float32)
    if L >= k:
        for i in range(L - k + 1):
            kmer = seq_ds[i:i + k]
            idx = kmer2idx.get(kmer)
            if idx is not None:
                kmer_counts[idx] += 1.0 [cite: 16]
        kmer_counts /= max(1, (L - k + 1))

    if cfg.USE_POSITIONAL_1MERS:
        n_bins = cfg.N_POSITION_BINS
        pos_counts = np.zeros((n_bins, len(cfg.AMINO_ACIDS)), dtype=np.float32)
        if L > 0:
            for i, ch in enumerate(seq_ds):
                if ch in aa2idx:
                    bin_idx = min(int(i / L * n_bins), n_bins - 1)
                    pos_counts[bin_idx, aa2idx[ch]] += 1.0 [cite: 18]
            # Normalize bins
            for b in range(n_bins):
                s = pos_counts[b].sum()
                if s > 0: pos_counts[b] /= s
        feats = np.concatenate([kmer_counts, pos_counts.reshape(-1)], axis=0) [cite: 19]
    else:
        feats = kmer_counts
    return feats

def build_sequence_feature_matrix(entry_ids, seq_dict, cfg, kmer2idx):
    X_list = []
    for pid in tqdm(entry_ids, desc="Building seq features"):
        X_list.append(sequence_to_features(seq_dict.get(pid, ""), cfg, kmer2idx))
    return np.stack(X_list, axis=0) [cite: 20]

# TAXON FEATURES
def load_train_taxonomy(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["EntryID", "TaxonID"])
    return dict(zip(df["EntryID"].astype(str), df["TaxonID"].astype(str)))

def build_taxon_encoder(train_entry_ids, train_taxon_map, cfg):
    counts = Counter([train_taxon_map.get(pid, "unknown") for pid in train_entry_ids])
    taxon2idx = {}
    idx = 0
    for tid, c in counts.items():
        if c >= cfg.MIN_TAXON_COUNT: [cite: 21]
            taxon2idx[tid] = idx
            idx += 1
    return taxon2idx, idx

def build_taxon_feature_matrix(entry_ids, taxon_map, taxon2idx, idx_other):
    rows, cols, data = [], [], []
    for i, pid in enumerate(entry_ids):
        tid = taxon_map.get(pid, "unknown")
        rows.append(i)
        cols.append(taxon2idx.get(tid, idx_other))
        data.append(1.0) [cite: 22]
    return csr_matrix((data, (rows, cols)), shape=(len(entry_ids), len(taxon2idx)+1), dtype=np.float32)
  
# main 1
if __name__ == "__main__":
    print("=== Loading FASTA (train) ===")
    train_seqs = read_train_fasta(os.path.join(CFG.DATA_DIR, CFG.TRAIN_FASTA))
    train_entry_ids = sorted(train_seqs.keys())
    
    print("=== Loading FASTA (test + taxon) ===")
    test_entry_ids, test_seqs, test_taxon_map = read_test_fasta_with_taxon(
        os.path.join(CFG.DATA_DIR, CFG.TEST_FASTA)
    ) [cite: 41]
    
    print("=== Loading Taxonomy (train) ===")
    train_taxon_map = load_train_taxonomy(os.path.join(CFG.DATA_DIR, CFG.TRAIN_TAXON))

    print("=== Building Taxon encoder ===")
    taxon2idx, idx_other = build_taxon_encoder(train_entry_ids, train_taxon_map, CFG)

    print("=== Building k-mer vocab ===")
    kmer2idx = build_kmer_vocab(CFG.AMINO_ACIDS, CFG.KMER_K)

    print("=== Building & Scaling sequence features ===")
    X_seq_train = build_sequence_feature_matrix(train_entry_ids, train_seqs, CFG, kmer2idx) [cite: 42]
    X_seq_test = build_sequence_feature_matrix(test_entry_ids, test_seqs, CFG, kmer2idx)
    
    scaler = StandardScaler()
    X_seq_train = scaler.fit_transform(X_seq_train).astype(np.float32)
    X_seq_test = scaler.transform(X_seq_test).astype(np.float32)

    print("=== Building taxon features ===")
    X_tax_train = build_taxon_feature_matrix(train_entry_ids, train_taxon_map, taxon2idx, idx_other)
    X_tax_test = build_taxon_feature_matrix(test_entry_ids, test_taxon_map, taxon2idx, idx_other)

    print("=== Combining features ===")
    X_train = np.hstack([X_seq_train, X_tax_train.toarray()]) [cite: 43]
    X_test = np.hstack([X_seq_test, X_tax_test.toarray()])

    # Lưu dữ liệu đã xử lý và xuất ra file
    print(f"=== Saving processed data to {CFG.OUTPUT_PROCESSED} ===")
    data_to_save = {
        "X_train": X_train,
        "X_test": X_test,
        "train_entry_ids": train_entry_ids,
        "test_entry_ids": test_entry_ids
    }
    with open(CFG.OUTPUT_PROCESSED, "wb") as f:
        pickle.dump(data_to_save, f)
    print("Done Preprocessing.")
