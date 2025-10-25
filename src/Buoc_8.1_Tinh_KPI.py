# -*- coding: utf-8 -*-
"""
Bước 8.1 - Phân tích kết quả ghép cặp từ bước 6
🎯 Mục tiêu: Hiển thị chi tiết và đánh giá các cặp ghép tự động
📝 Input:
    - output/matched_pairs.csv: File kết quả từ bước 6 (các cặp đã ghép)
    - data/answer_key_sample.csv: File đáp án để so sánh
� Output:
    - Danh sách chi tiết các cặp matched
    - Thống kê tổng hợp (số lượng, tỷ lệ)
    - Phân tích chất lượng ghép (KPIs)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# === THIẾT LẬP ĐƯỜNG DẪN ===
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"

# Đảm bảo các thư mục tồn tại
output_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Tạo thư mục output nếu chưa tồn tại
output_dir.mkdir(exist_ok=True)

# === THIẾT LẬP TÊN CÁC TỆP ===
MATCHED_FILE = output_dir / "matched_pairs.csv"
ANSWER_KEY = data_dir / "answer_key_sample.csv"
KPI_REPORT = output_dir / "kpi_report_8.1.csv"
KPI_DETAILS = output_dir / "kpi_details_8.1.csv"

def calculate_metrics(all_transactions, truth_pairs):
    """Tính toán các chỉ số Precision, Recall"""
    
    # Lọc các cặp đã matched
    system_pairs = all_transactions[all_transactions['status'] == 'matched']
    
    # Tính True Positives (ghép đúng)
    merged_df = pd.merge(system_pairs, truth_pairs, 
                        on=['bank_ref', 'gl_doc'], 
                        how='inner')
    true_positives = len(merged_df)
    
    # Tính False Positives (ghép sai)
    false_positives = len(system_pairs) - true_positives
    
    # Tính False Negatives (bỏ sót)
    false_negatives = len(truth_pairs) - true_positives
    
    # Tính Precision
    precision = (true_positives / len(system_pairs) * 100) if len(system_pairs) > 0 else 0
    
    # Tính Recall
    recall = (true_positives / len(truth_pairs) * 100) if len(truth_pairs) > 0 else 0
    
    # Tính F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def analyze_results(all_transactions, truth_df, total_transactions):
    """Phân tích chi tiết kết quả đối soát"""
    
    # 1. Tính Auto-match rate
    matched_pairs = all_transactions[all_transactions['status'] == 'matched']
    auto_matched = len(matched_pairs)
    auto_match_rate = (auto_matched / total_transactions) * 100 if total_transactions > 0 else 0
    
    # 2. Tính các metric cho từng nhóm
    matched_metrics = calculate_metrics(all_transactions, truth_df)
    
    # 3. Phân tích chi tiết
    details = []
    
    # Chi tiết các cặp ghép đúng
    true_matches = pd.merge(
        all_transactions[all_transactions['status'] == 'matched'],
        truth_df,
        on=['bank_ref', 'gl_doc'],
        how='inner'
    )
    for _, row in true_matches.iterrows():
        details.append({
            'bank_ref': row['bank_ref'],
            'gl_doc': row['gl_doc'],
            'status': 'TRUE_POSITIVE',
            'reason': 'Ghép đúng theo answer key'
        })
    
    # Chi tiết các cặp ghép sai
    matched_only = all_transactions[all_transactions['status'] == 'matched']
    false_matches = matched_only.merge(
        truth_df,
        on=['bank_ref', 'gl_doc'],
        how='left',
        indicator=True
    )
    false_matches = false_matches[false_matches['_merge'] == 'left_only']
    for _, row in false_matches.iterrows():
        details.append({
            'bank_ref': row['bank_ref'],
            'gl_doc': row['gl_doc'],
            'status': 'FALSE_POSITIVE',
            'reason': 'Ghép không khớp với answer key'
        })
    
    # Chi tiết các cặp bỏ sót
    missed_matches = truth_df.merge(
        matched_only,
        on=['bank_ref', 'gl_doc'],
        how='left',
        indicator=True
    )
    missed_matches = missed_matches[missed_matches['_merge'] == 'left_only']
    for _, row in missed_matches.iterrows():
        details.append({
            'bank_ref': row['bank_ref'],
            'gl_doc': row['gl_doc'],
            'status': 'FALSE_NEGATIVE',
            'reason': 'Cặp ghép đúng bị bỏ sót'
        })
    
    return {
        'auto_match_rate': auto_match_rate,
        'metrics': matched_metrics,
        'details': pd.DataFrame(details)
    }

def main():
    """Hàm chính để hiển thị và phân tích kết quả ghép cặp"""
    print("\n=== BƯỚC 8.1: PHÂN TÍCH KẾT QUẢ GHÉP CẶP ===")
    
    # 1. Kiểm tra file bắt buộc
    required_files = [MATCHED_FILE, ANSWER_KEY]
    for f in required_files:
        if not f.exists():
            print(f"❌ Không tìm thấy file bắt buộc: {f}")
            return
    
    try:
        # 2. Đọc dữ liệu
        print("\n📂 Đang đọc dữ liệu...")
        
        # Đọc file matched pairs và answer key
        matched_df = pd.read_csv(MATCHED_FILE)
        truth_df = pd.read_csv(ANSWER_KEY)

        # 3. Hiển thị danh sách các cặp matched
        print("\n📋 DANH SÁCH CÁC CẶP GHÉP THÀNH CÔNG:")
        print("=" * 100)
        print("   Bank Ref         GL Doc      Score     Status")
        print("-" * 100)
        
        matched_only = matched_df[matched_df['match_type'] == 'MATCH'].sort_values(by='match_score', ascending=False)
        for _, row in matched_only.iterrows():
            print(f"   {row['bank_ref']:<15} {row['gl_doc']:<10} {row['match_score']:>8.3f}   {row['match_type']}")
        
        # 4. Hiển thị thống kê tổng hợp
        print("\n📊 THỐNG KÊ TỔNG HỢP:")
        print("=" * 50)
        
        # Đọc các file nguồn
        bank_df = pd.read_csv(data_dir / "bank_stmt.csv")
        gl_df = pd.read_csv(data_dir / "gl_entries.csv")
        truth_df = pd.read_csv(data_dir / "answer_key_sample.csv")
        
        # Tính toán dựa trên answer key
        true_matches_count = len(truth_df)  # Số cặp match thực tế trong answer key
        bank_count = len(bank_df)
        gl_count = len(gl_df)
        
        total_pairs = len(matched_df)
        matched_count = len(matched_only)
        
        print(f"Tổng số giao dịch      : {bank_count:>5} (Bank: {bank_count}, GL: {gl_count})")
        print(f"Số cặp thực tế (answer): {true_matches_count:>5}")
        print(f"Số cặp ghép thành công  : {matched_count:>5}")
        print(f"Auto-match rate         : {matched_count/true_matches_count*100:>5.1f}%")
        print(f"Tỷ lệ ghép thành công   : {matched_count/total_pairs*100:>5.1f}%")
        
        # 5. Hiển thị phân bố điểm số
        print("\n📈 PHÂN BỐ ĐIỂM SỐ:")
        print("=" * 50)
        
        score_bins = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        score_counts = pd.cut(matched_only['match_score'], bins=score_bins).value_counts().sort_index()
        
        for interval, count in score_counts.items():
            print(f"Điểm {interval.left:.2f} - {interval.right:.2f}: {count:>3} cặp")
        
        # 6. Đánh giá chất lượng ghép
        print("\n🎯 ĐÁNH GIÁ CHẤT LƯỢNG GHÉP:")
        print("=" * 50)
        
        # So sánh với answer key
        true_matches = pd.merge(matched_only, truth_df, on=['bank_ref', 'gl_doc'], how='inner')
        precision = len(true_matches) / len(matched_only) * 100
        recall = len(true_matches) / len(truth_df) * 100
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision (độ chính xác) : {precision:>5.1f}%")
        print(f"Recall (độ phủ)          : {recall:>5.1f}%")
        print(f"F1-Score (điểm tổng hợp) : {f1_score:>5.1f}%")
        
        print("\n✨ Hoàn tất phân tích kết quả ghép cặp!")
        
        # 7. Lưu báo cáo chi tiết nếu cần
        detailed_report = {
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'Value': [precision, recall, f1_score],
            'Description': [
                f'Tỉ lệ ghép đúng ({len(true_matches)} / {len(matched_only)} cặp)',
                f'Tỉ lệ tìm thấy ({len(true_matches)} / {len(truth_df)} cặp)',
                'Điểm trung bình điều hòa'
            ]
        }
        
        # 8. Lưu báo cáo chi tiết
        pd.DataFrame(detailed_report).to_csv(KPI_REPORT, index=False)
        
        print(f"\n✓ Đã lưu báo cáo chi tiết vào: {KPI_REPORT}")

        # 9. Xuất 3 file riêng biệt cho MATCH, REVIEW, UNMATCHED
        print("\n📤 Đang xuất các file phân loại...")
        
        # File MATCH
        match_pairs = matched_df[matched_df['match_type'] == 'MATCH']
        match_file = output_dir / 'matched_pairs_MATCH.csv'
        match_pairs.to_csv(match_file, index=False)
        print(f"✅ Đã xuất {len(match_pairs)} cặp MATCH vào: {match_file}")
        
        # File REVIEW  
        review_pairs = matched_df[matched_df['match_type'] == 'REVIEW']
        review_file = output_dir / 'matched_pairs_REVIEW.csv'
        review_pairs.to_csv(review_file, index=False)
        print(f"✅ Đã xuất {len(review_pairs)} cặp REVIEW vào: {review_file}")
        
        # File UNMATCHED
        unmatched_pairs = matched_df[matched_df['match_type'] == 'UNMATCHED']
        unmatched_file = output_dir / 'matched_pairs_UNMATCHED.csv'
        unmatched_pairs.to_csv(unmatched_file, index=False)
        print(f"✅ Đã xuất {len(unmatched_pairs)} cặp UNMATCHED vào: {unmatched_file}")

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        raise e

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    
    # Tính và hiển thị latency
    processing_time = (end_time - start_time).total_seconds()
    print(f"\nLatency: {processing_time:.2f} seconds")