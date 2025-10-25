# -*- coding: utf-8 -*-
"""
Bước 8 — Silver Layer: Hybrid Text (BM25 + Embedding)
🎯 Mục tiêu: Tăng khả năng bắt các mô tả sai chính tả hoặc viết khác.
📥 Đầu vào:
    - bank_stmt.csv (để lấy text mô tả)
    - gl_entries.csv (để lấy text đối tác)
    - adjusted_pairs.csv (để biết các cặp ĐÃ KHỚP)
⚙️ Xử lý:
    - Tính điểm BM25 (keyword matching)
    - Tính điểm Embedding (semantic matching)
    - Kết hợp: f_text = 0.5 * BM25_norm + 0.5 * Embedding_norm
    - Lưu top-3 ứng viên.
📤 Đầu ra: silver_candidates.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# === CÀI ĐẶT CẦN THIẾT ===
# Cần cài đặt 2 thư viện này trước khi chạy:
# pip install rank_bm25
# pip install sentence-transformers

# === THAM SỐ ===
MODEL_NAME = 'intfloat/multilingual-e5-small'  # Hoặc 'BAAI/bge-m3'
TOP_K = 3                # Lưu top-3 ứng viên
HYBRID_WEIGHT_BM25 = 0.5 # Trọng số cho BM25
HYBRID_WEIGHT_EMBED = 0.5 # Trọng số cho Embedding

MIN_SCORE_THRESHOLD = 0.3 # Ngưỡng điểm tối thiểu để lưu kết quả

# === ĐƯỜNG DẪN ===
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"

# Input files
IN_BANK = data_dir / "bank_stmt.csv"
IN_GL = data_dir / "gl_entries.csv"
IN_ADJUSTED_PAIRS = output_dir / "adjusted_pairs.csv" # Đầu ra từ Bước 7

# Output file
OUT_SILVER = output_dir / "silver_candidates.csv"

def load_unmatched_data(bank_path, gl_path, adjusted_path):
    """
    Tải dữ liệu bank và GL, sau đó lọc ra những
    dòng chưa được khớp ở Bước 7.
    """
    bank_df = pd.read_csv(bank_path)
    gl_df = pd.read_csv(gl_path)
    adjusted_df = pd.read_csv(adjusted_path)

    # Lọc các cặp đã review hoặc matched trong adjusted_pairs
    reviewed_pairs = adjusted_df[
        adjusted_df['match_status'].isin(['Review', 'Matched'])
    ][['bank_ref', 'gl_doc']]

    # Lấy danh sách các ref và doc đã khớp
    matched_bank_refs = reviewed_pairs['bank_ref'].unique()
    matched_gl_docs = reviewed_pairs['gl_doc'].unique()

    # Lọc ra những dòng chưa khớp
    bank_df = bank_df.copy()
    gl_df = gl_df.copy()
    
    # Map ref để khớp với adjusted_pairs
    bank_df['ref'] = bank_df['ref'].astype(str)
    gl_df['doc_no'] = gl_df['doc_no'].astype(str)
    
    unmatched_bank = bank_df[~bank_df['ref'].isin(matched_bank_refs)].copy()
    unmatched_gl = gl_df[~gl_df['doc_no'].isin(matched_gl_docs)].copy()

    # Chuẩn bị text: bank dùng 'desc', GL dùng 'partner'
    # .fillna('') để xử lý các giá trị NaN/trống
    unmatched_bank['text'] = unmatched_bank['desc'].fillna('')
    unmatched_bank['desc'] = unmatched_bank['desc'].fillna('')  # Giữ lại desc gốc
    unmatched_gl['text'] = unmatched_gl['partner'].fillna('')
    unmatched_gl['partner'] = unmatched_gl['partner'].fillna('')  # Giữ lại partner gốc
    
    # Chỉ lấy các cột cần thiết
    unmatched_bank = unmatched_bank[['ref', 'text', 'desc']]
    unmatched_gl = unmatched_gl[['doc_no', 'text', 'partner']]

    return unmatched_bank, unmatched_gl

def preprocess_text(text):
    """Tiền xử lý text trước khi tính toán"""
    text = str(text).lower()
    # Có thể thêm các bước tiền xử lý khác như:
    # - Loại bỏ dấu câu
    # - Chuẩn hóa unicode
    # - Loại bỏ stopwords
    return text

def get_bm25_scores(queries, corpus):
    """Tính điểm BM25 với tiền xử lý cải tiến"""
    print("... Đang tính điểm BM25 ...")
    
    # Tiền xử lý
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    processed_queries = [preprocess_text(q) for q in queries]
    
    # Tokenize
    tokenized_corpus = [doc.split() for doc in processed_corpus]
    tokenized_queries = [q.split() for q in processed_queries]
    
    # Khởi tạo BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tính điểm
    bm25_scores = [bm25.get_scores(q) for q in tokenized_queries]
    scores = np.array(bm25_scores)
    
    # Chuẩn hóa về [0,1]
    max_scores = np.maximum(scores.max(axis=1, keepdims=True), 1e-6)
    normalized_scores = scores / max_scores
    
    return normalized_scores

def get_embedding_scores(model, queries, corpus):
    """Tính điểm embedding similarity với caching"""
    print(f"... Đang tính Embedding (sử dụng {MODEL_NAME}) ...")
    
    # Tiền xử lý
    processed_queries = [preprocess_text(q) for q in queries]
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    
    # Tính embeddings
    query_embeddings = model.encode(
        processed_queries,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32  # Tăng tốc độ với batching
    )
    
    corpus_embeddings = model.encode(
        processed_corpus,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    # Tính cosine similarity
    similarities = util.cos_sim(query_embeddings, corpus_embeddings)
    return similarities.cpu().numpy()

def get_ground_truth_pairs(adjusted_path):
    """Đọc các cặp đã khớp từ adjusted_pairs.csv để làm ground truth"""
    df = pd.read_csv(adjusted_path)
    # Lấy các cặp có status là Matched
    matched = df[df['match_status'] == 'Matched'][['bank_ref', 'gl_doc']]
    # Tạo dictionary để map bank_ref -> gl_doc
    truth_dict = dict(zip(matched['bank_ref'], matched['gl_doc']))
    return truth_dict

def calculate_recall_at_k(bank_ref, predictions, truth_dict, k=3):
    """Tính Recall@K cho một query dựa trên ground truth"""
    if bank_ref not in truth_dict:
        return 0.0  # Không có trong ground truth
    correct_gl = truth_dict[bank_ref]
    if correct_gl in predictions[:k]:
        return 1.0
    return 0.0

def main():
    print("\n=== BƯỚC 8: SILVER LAYER - HYBRID TEXT MATCHING ===")

    # 1. Kiểm tra và tải dữ liệu
    required_files = [IN_BANK, IN_GL, IN_ADJUSTED_PAIRS]
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Không tìm thấy file: {file_path}")
            return
            
    print("✓ Đang tải dữ liệu...")
    try:
        bank_queries_df, gl_corpus_df = load_unmatched_data(IN_BANK, IN_GL, IN_ADJUSTED_PAIRS)
        
        if bank_queries_df.empty or gl_corpus_df.empty:
            print("❌ Không có dữ liệu bank hoặc GL chưa khớp để xử lý.")
            return

        print(f"✓ Dữ liệu chưa khớp: {len(bank_queries_df):,} bank vs {len(gl_corpus_df):,} GL")
        
        queries_list = bank_queries_df['text'].tolist()
        corpus_list = gl_corpus_df['text'].tolist()

    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu: {str(e)}")
        return

    # 2. Tải mô hình embedding
    print(f"\n✓ Đang tải mô hình {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {str(e)}")
        print("Hint: Kiểm tra kết nối mạng hoặc cài đặt thư viện.")
        return

    # 3. Tính toán điểm BM25 và Embedding
    try:
        # Tính điểm BM25 (đã chuẩn hóa)
        bm25_scores = get_bm25_scores(queries_list, corpus_list)
        
        # Tính điểm Embedding (đã chuẩn hóa)
        embed_scores = get_embedding_scores(model, queries_list, corpus_list)
        
        # Kết hợp điểm với trọng số
        hybrid_scores = (HYBRID_WEIGHT_BM25 * bm25_scores + 
                        HYBRID_WEIGHT_EMBED * embed_scores)
                        
    except Exception as e:
        print(f"❌ Lỗi tính toán điểm: {str(e)}")
        return

    # 4. Tải ground truth và trích xuất Top-K ứng viên
    print("\n✓ Đang tải ground truth từ adjusted_pairs.csv...")
    truth_dict = get_ground_truth_pairs(IN_ADJUSTED_PAIRS)
    
    print(f"✓ Đang xử lý {len(queries_list):,} queries...")
    results = []
    recalls = []
    
    for query_idx, scores in enumerate(hybrid_scores):
        bank_ref = str(bank_queries_df['ref'].iloc[query_idx])
        bank_desc = str(bank_queries_df['desc'].iloc[query_idx])
        
        # Lấy top-K theo điểm
        top_k_indices = np.argsort(scores)[-TOP_K:][::-1]
        
        # Lấy danh sách GL docs cho query này
        gl_candidates = [str(gl_corpus_df['doc_no'].iloc[i]) for i in top_k_indices]
        
        # Tính Recall@K
        recall = calculate_recall_at_k(bank_ref, gl_candidates, truth_dict, TOP_K)
        recalls.append(recall)
        
        # Lưu kết quả cho từng ứng viên
        for rank, idx in enumerate(top_k_indices):
            score = scores[idx]
            gl_partner = str(gl_corpus_df['partner'].iloc[idx])
            
            if score >= MIN_SCORE_THRESHOLD:  # Chỉ lưu nếu vượt ngưỡng
                # Tính text similarity từ điểm BM25 và embedding
                text_similarity = score  # Đã là điểm kết hợp giữa BM25 và embedding
                
                # Xác định reason_flag
                reason_flag = []
                if score >= 0.8:
                    reason_flag.append("HIGH_SIMILARITY")
                if bm25_scores[query_idx][idx] >= 0.7:
                    reason_flag.append("STRONG_TEXT_MATCH")
                if embed_scores[query_idx][idx] >= 0.7:
                    reason_flag.append("STRONG_SEMANTIC_MATCH")
                if not reason_flag:
                    reason_flag.append("POTENTIAL_MATCH")
                    
                results.append({
                    'bank_ref': bank_ref,
                    'gl_doc': str(gl_corpus_df['doc_no'].iloc[idx]),
                    'partner': gl_partner,
                    'text_similarity': round(text_similarity, 4),
                    'reason_flag': "|".join(reason_flag),
                    'f_text': round(score, 4),
                    'Recall@3': recall,
                    'Rank': rank + 1,
                    'in_ground_truth': bank_ref in truth_dict
                })

    # 5. Tạo DataFrame kết quả
    if not results:
        print("⚠️ Không tìm thấy ứng viên nào vượt ngưỡng điểm.")
        return

    results_df = pd.DataFrame(results)
    
    # Sắp xếp theo bank_ref và điểm số
    results_df = results_df.sort_values(
        ['bank_ref', 'f_text'], 
        ascending=[True, False]
    )
    
    # 6. Lưu kết quả và in thống kê
    results_df.to_csv(OUT_SILVER, index=False)
    
    # Tính các thống kê
    ground_truth_count = sum(1 for ref in bank_queries_df['ref'] if ref in truth_dict)
    
    print("\n=== THỐNG KÊ ===")
    print(f"✓ Tổng số cặp ứng viên: {len(results_df):,}")
    print(f"✓ Số lượng bank refs: {results_df['bank_ref'].nunique():,}")
    print(f"✓ Điểm trung bình: {results_df['f_text'].mean():.4f}")
    print(f"✓ Recall@{TOP_K}: {np.mean(recalls):.2%}")
    print(f"✓ Trong ground truth: {sum(results_df['in_ground_truth']):,}/{ground_truth_count:,}")
    
    print("\n=== MẪU KẾT QUẢ ===")
    print(results_df[['bank_ref', 'gl_doc', 'partner', 'text_similarity', 'reason_flag', 'f_text', 'Rank']]
          .head(5)
          .to_string(index=False))

if __name__ == '__main__':
    main()