# -*- coding: utf-8 -*-
"""
Bước 4 — Sinh ứng viên ±biên độ ngày/số tiền (Cập nhật output)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === THAM SỐ ===
DAY_TOL = 10      # ±10 ngày
AMT_TOL_PCT = 0.05   # ±5%
AMT_TOL_ABS = 10_000  # ±10,000 VND

# === ĐƯỜNG DẪN ===
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

# Input/Output files
IN_BANK = OUTPUT_DIR / "bank_fx.csv"
IN_GL = DATA_DIR / "gl_entries.csv"
OUT_CAND = OUTPUT_DIR / "candidate_pairs_detailed.csv" # Đổi tên file output cho rõ nghĩa

def main():
    print("\n=== BƯỚC 4: SINH ỨNG VIÊN (OUTPUT CHI TIẾT) ===")
    
    # 1. Đọc dữ liệu
    try:
        bank_fx = pd.read_csv(IN_BANK)
        bank_stmt = pd.read_csv(DATA_DIR / "bank_stmt.csv")
        gl = pd.read_csv(IN_GL)
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file đầu vào. Hãy đảm bảo các file sau tồn tại:")
        print(f"   - {IN_BANK}")
        print(f"   - {DATA_DIR / 'bank_stmt.csv'}")
        print(f"   - {IN_GL}")
        return

    # Chuẩn hóa dữ liệu đầu vào
    try:
        # Merge cả currency từ bank_stmt
        bank = pd.merge(bank_fx, bank_stmt[['txn_date', 'ref', 'currency']], on='txn_date', how='left')
        bank['date'] = pd.to_datetime(bank['txn_date'])
        gl['date'] = pd.to_datetime(gl['post_date'])
        bank['amount'] = pd.to_numeric(bank['amount_vnd'], errors='coerce')
        gl['amount'] = pd.to_numeric(gl['amount'], errors='coerce')
        bank['bank_ref'] = bank['ref'].fillna('').astype(str)
        gl['gl_doc'] = gl['doc_no']
        
        # Standardize currency
        bank['currency'] = bank['currency'].str.upper()
        gl['currency'] = gl['currency'].str.upper()
        # Set default currency to VND if missing
        bank['currency'] = bank['currency'].fillna('VND')
        gl['currency'] = gl['currency'].fillna('VND')
        
        print(f"✓ Đã tải:\n   - Bank: {len(bank):,} dòng\n   - GL: {len(gl):,} dòng")
            
    except Exception as e:
        print(f"❌ Lỗi khi xử lý dữ liệu: {str(e)}")
        return
    
    # 2. Tìm kiếm ứng viên
    candidates = []
    
    for _, bank_row in bank.iterrows():
        mask_curr = gl['currency'] == bank_row['currency']
        mask_date = abs((gl['date'] - bank_row['date']).dt.days) <= DAY_TOL
        
        gl_matched = gl[mask_curr & mask_date].copy()
        
        if not gl_matched.empty:
            delta_amt = abs(gl_matched['amount'] - bank_row['amount'])
            tol_pct = AMT_TOL_PCT * abs(bank_row['amount'])
            tol_final = np.maximum(AMT_TOL_ABS, tol_pct)
            
            gl_matched = gl_matched[delta_amt <= tol_final]
            
            for _, gl_row in gl_matched.iterrows():
                days_diff = abs((gl_row['date'] - bank_row['date']).days)
                
                # CẬP NHẬT Ở ĐÂY: Thêm các cột chi tiết vào danh sách
                candidates.append({
                    'bank_ref': bank_row['bank_ref'],
                    'gl_doc': gl_row['gl_doc'],
                    'txn_date': bank_row['date'].strftime('%Y-%m-%d'),
                    'post_date': gl_row['date'].strftime('%Y-%m-%d'),
                    'amount_vnd(bank)': bank_row['amount'],
                    'amount_vnd(GL)': gl_row['amount'],
                    'Δdays': days_diff,
                    'Δamount': bank_row['amount'] - gl_row['amount'], # Chênh lệch số thực
                    'same_currency': True
                })
    
    # Chuyển list thành DataFrame
    if candidates:
        out = pd.DataFrame(candidates)
        # Sắp xếp theo các cột mới
        out = out.sort_values(by=['bank_ref', 'Δdays', 'Δamount'], ascending=[True, True, True])
        out = out.drop_duplicates()
    else:
        # CẬP NHẬT Ở ĐÂY: Tạo DataFrame rỗng với đúng cấu trúc cột mới
        out = pd.DataFrame(columns=[
            'bank_ref', 'gl_doc', 'txn_date', 'post_date', 
            'amount_vnd(bank)', 'amount_vnd(GL)', 'Δdays', 'Δamount', 'same_currency'
        ])
    
    # Ghi file kết quả
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CAND, index=False)
    
    print(f"\n✓ Đã tìm thấy {len(out):,} cặp ứng viên.")
    print(f"✓ Đã ghi file với các cột chi tiết: {OUT_CAND}")
    print("\nXem trước kết quả:")
    print(out.head().to_string())

if __name__ == '__main__':
    main()