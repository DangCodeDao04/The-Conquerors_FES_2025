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
root_dir = os.path.dirname(script_dir)  # Thư mục code_du_an/code_du_an
data_dir = os.path.join(root_dir, "data")  # data nằm trong code_du_an/code_du_an/data
output_dir = os.path.join(root_dir, "output")  # output nằm trong code_du_an/code_du_an/output

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

print("=== ĐƯỜNG DẪN ===")
print(f"Script dir: {script_dir}")
print(f"Root dir: {root_dir}")
print(f"Data dir: {data_dir}")
print(f"Output dir: {output_dir}")

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

def main():
    """Hàm chính để tính toán KPI"""
    print("\n=== BƯỚC 6.1: TÍNH TOÁN KPI ===")
    
    # 1. Kiểm tra files
    required_files = [MATCHED_PAIRS, BANK_STMT, ANSWER_KEY]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Không tìm thấy file: {f}")
            return

    try:
        # 2. Đọc dữ liệu
        print("✓ Đang đọc dữ liệu...")
        system_df = pd.read_csv(MATCHED_PAIRS)
        bank_df = pd.read_csv(BANK_STMT)
        truth_df = pd.read_csv(ANSWER_KEY)
        
        # 3. Chuẩn bị dữ liệu
        print("\nCấu trúc dữ liệu:")
        print("Các cột trong system_df:", system_df.columns.tolist())
        
        # Xử lý match_type
        if 'match_type' not in system_df.columns:
            # Phân loại giao dịch dựa vào định dạng bank_ref
            def determine_type(bank_ref):
                if isinstance(bank_ref, str):
                    if bank_ref.startswith('TXN-'):
                        return 'TUITION'
                    elif bank_ref.startswith('FEE-'):
                        return 'BANKFEE'
                    elif bank_ref.startswith('USDTXN-'):
                        return 'FX_USD_to_VND'
                    elif bank_ref.startswith('SRV-'):
                        return 'SERVICE'
                return 'UNKNOWN'
            
            system_df['match_type'] = system_df['bank_ref'].apply(determine_type)
            print("✓ Đã tạo cột match_type dựa vào định dạng bank_ref")
        else:
            system_df['match_type'] = system_df['match_type'].fillna('UNKNOWN')
            print("✓ Đã điền giá trị UNKNOWN cho match_type thiếu")
        
        # 4. Tính Auto-match rate tổng thể
        total_bank_txns = len(bank_df)
        auto_matched = len(system_df)
        auto_match_rate = (auto_matched / total_bank_txns) * 100

        # 5. Tính metrics cho từng loại giao dịch
        transaction_types = ['TUITION', 'BANKFEE', 'FX_USD_to_VND', 'SERVICE']
        metrics_results = []
        
        # Tính cho tất cả các giao dịch
        overall_metrics = calculate_metrics_by_type(system_df, truth_df)
        metrics_results.append(overall_metrics)
        
        # Tính cho từng loại giao dịch
        for tx_type in transaction_types:
            metrics = calculate_metrics_by_type(system_df, truth_df, tx_type)
            metrics_results.append(metrics)

        # 6. Tạo báo cáo
        print("\n=== BÁO CÁO KPI ===")
        print(f"\nAuto-match rate: {auto_match_rate:.2f}%")
        
        # Tạo DataFrame từ kết quả
        report_df = pd.DataFrame(metrics_results)
        
        # Định dạng các cột số
        for col in ['precision', 'recall', 'f1_score']:
            report_df[col] = report_df[col].round(2)
            
        # Hiển thị bảng kết quả
        print("\nChi tiết theo loại giao dịch:")
        print("="*80)
        print(report_df.to_string(index=False))
        print("="*80)
        
        # Lưu báo cáo
        report_df.to_csv(KPI_REPORT, index=False)
        print(f"\n✓ Đã lưu báo cáo KPI vào: {KPI_REPORT}")

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        raise e

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"\nLatency: {processing_time:.2f} seconds")