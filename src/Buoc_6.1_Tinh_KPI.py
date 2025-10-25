# -*- coding: utf-8 -*-
"""
Bước 6.1 - Tính toán KPI
🎯 Mục tiêu: Tính các chỉ số hiệu suất của hệ thống
📝 Các KPI cần tính:
    1. Auto-match rate: Tỷ lệ giao dịch được ghép tự động
    2. Precision: Độ chính xác của các cặp được ghép
    3. Recall: Tỷ lệ tìm thấy các cặp đúng
    4. Latency: Thời gian xử lý (được tính trong quá trình chạy)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# === THIẾT LẬP ĐƯỜNG DẪN ===
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")

# === THIẾT LẬP TÊN CÁC TỆP ĐẦU VÀO/RA ===
MATCHED_PAIRS = os.path.join(output_dir, 'matched_pairs.csv')
BANK_STMT = os.path.join(data_dir, 'bank_stmt.csv')
ANSWER_KEY = os.path.join(data_dir, 'answer_key_sample.csv')
KPI_REPORT = os.path.join(output_dir, 'kpi_report.csv')

def calculate_metrics_by_type(system_df, truth_df, transaction_type=None):
    """Tính toán các metrics cho từng loại giao dịch"""
    
    if transaction_type:
        system_matched = system_df[system_df['match_type'] == transaction_type]
        truth_matched = truth_df[truth_df['match_type'] == transaction_type]
    else:
        system_matched = system_df
        truth_matched = truth_df

    # Tính True Positives (các cặp ghép đúng)
    merged_df = pd.merge(system_matched, truth_matched, 
                        on=['bank_ref', 'gl_doc'], 
                        how='inner')
    true_positives = len(merged_df)
    
    # Tính False Positives (các cặp ghép sai)
    false_positives = len(system_matched) - true_positives
    
    # Tính False Negatives (các cặp bỏ sót)
    false_negatives = len(truth_matched) - true_positives
    
    # Tính Precision
    precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
    
    # Tính Recall
    recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
    
    # Tính F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'type': transaction_type if transaction_type else 'ALL',
        'total_pairs': len(system_matched),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    # --- KIỂM TRA SỰ TỒN TẠI CỦA CÁC TỆP ---
    required_files = [system_results_file, bank_statement_file, ground_truth_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ LỖI: Không tìm thấy tệp '{f}'!")
            print("   -> Vui lòng đảm bảo tệp này tồn tại trong cùng thư mục với mã.")
            return

    print("✅ Đã tìm thấy tất cả các tệp cần thiết.")
    
    try:
        # Đọc các tệp
        system_df = pd.read_csv(system_results_file)
        bank_df = pd.read_csv(bank_statement_file)
        truth_df = pd.read_csv(ground_truth_file)
        
        # --- 1. Tính Auto-match Rate ---
        total_bank_txns = len(bank_df)
        # Đảm bảo cột 'match_status' tồn tại
        if 'match_status' not in system_df.columns:
             print("❌ LỖI: Tệp 'matched_pairs.csv' phải có cột 'match_status'.")
             return
        auto_matched_count = len(system_df[system_df['match_status'] == 'Matched'])
        auto_match_rate = (auto_matched_count / total_bank_txns) * 100 if total_bank_txns > 0 else 0

        # --- 2. Tính Precision và Recall ---
        system_matched = system_df[system_df['match_status'] == 'Matched'][['bank_ref', 'gl_doc']].dropna()
        merged_df = pd.merge(system_matched, truth_df, on=['bank_ref', 'gl_doc'], how='inner')
        
        true_positives = len(merged_df)
        false_positives = len(system_matched) - true_positives
        false_negatives = len(truth_df) - true_positives
        
        precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        
        print("✅ Đã tính toán xong các chỉ số KPI.")

        # --- 3. Hiển thị kết quả ---
        kpi_summary = {
            'KPI': ['Auto-match rate', 'Precision', 'Recall', 'Latency'],
            'Value': [f"{auto_match_rate:.2f}%", f"{precision:.2f}%", f"{recall:.2f}%", "N/A"],
            'Description': [
                'Tỷ lệ giao dịch được hệ thống tự động đối soát thành công.',
                'Trong số các cặp được nối tự động, có bao nhiêu % là đúng.',
                'Trong tổng số các cặp đúng, hệ thống tìm thấy được bao nhiêu %.',
                'Thời gian xử lý (cần đo lường thủ công khi chạy quy trình).'
            ]
        }
        kpi_df = pd.DataFrame(kpi_summary)
        
        print("\n" + "="*50)
        print("              BẢNG TỔNG HỢP KPI              ")
        print("="*50)
        print(kpi_df.to_markdown(index=False, tablefmt="grid"))
        print("="*50)

    except Exception as e:
        print(f"😕 Đã xảy ra lỗi không mong muốn trong quá trình xử lý: {e}")

# --- Chạy hàm chính ---
if __name__ == "__main__":
    calculate_kpis()