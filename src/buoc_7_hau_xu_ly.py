# -*- coding: utf-8 -*-
"""
Bước 7 — Hậu xử lý lệch (Fee / FX / Partial / Duplicate)
🎯 Mục tiêu:
- Xử lý các trường hợp đặc biệt khiến giao dịch không khớp hoàn toàn
- Phát hiện và điều chỉnh phí
- Xử lý chênh lệch tỷ giá
- Gom nhóm thanh toán một phần
- Phát hiện giao dịch trùng lặp
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# (Các phần THAM SỐ và ĐƯỜNG DẪN giữ nguyên)
# === THAM SỐ ===
FEE_THRESHOLD = 20_000          # Ngưỡng phí tối đa
PARTIAL_THRESHOLD = 3           # Số giao dịch tối đa trong một nhóm partial
FX_THRESHOLD = 0.02            # Ngưỡng chênh lệch tỷ giá (2%)
DUPLICATE_TIME_WINDOW = 3       # Cửa sổ thời gian để kiểm tra trùng (ngày)

# === ĐƯỜNG DẪN ===
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"

# Input files
IN_MATCHES = output_dir / "matched_pairs.csv"
IN_BANK = data_dir / "bank_stmt.csv"
IN_GL = data_dir / "gl_entries.csv"
IN_BANK_FX = output_dir / "bank_fx.csv"  # File từ bước 3

# Output file
OUT_ADJUSTED = output_dir / "adjusted_pairs.csv"

def detect_fee_adjustments(matches_df, bank_df):
    """Phát hiện và điều chỉnh các khoản phí"""
    fee_adjustments = []
    bank_df = bank_df.sort_values('txn_date')
    
    # Tìm các giao dịch phí
    for _, row in matches_df.iterrows():
        bank_ref = row['bank_ref']
        bank_txns = bank_df[bank_df['ref'] == bank_ref]
        
        if bank_txns.empty:
            continue
            
        bank_txn = bank_txns.iloc[0]
        amount = abs(bank_txn['amount'])
        
        # Kiểm tra các điều kiện phí:
        # 1. Số tiền nhỏ hơn ngưỡng
        # 2. Có từ 'fee', 'phí', 'phi' trong mô tả
        desc = str(bank_txn.get('desc_clean', '')).lower()
        is_fee = (amount <= FEE_THRESHOLD) or \
                any(word in desc for word in ['fee', 'phí', 'phi', 'charge'])
        
        if is_fee:
            fee_adjustments.append({
                'bank_ref': bank_ref,
                'gl_doc': row['gl_doc'],
                'match_score': row['match_score'],
                'match_status': 'Matched',
                'mismatch_type': 'Fee Adjustment',
                'note': f'+{amount:,.0f} VND fee'
            })
    
    return pd.DataFrame(fee_adjustments)

def detect_fx_mismatches(matches_df, bank_fx_df):
    """Phát hiện các trường hợp chênh lệch tỷ giá"""
    fx_mismatches = []
    
    for _, row in matches_df.iterrows():
        bank_ref = row['bank_ref']
        fx_rows = bank_fx_df[bank_fx_df['ref_clean'] == bank_ref]
        
        if fx_rows.empty:
            continue
            
        fx_row = fx_rows.iloc[0]
        
        # Kiểm tra nếu là giao dịch USD và có flag_FX
        if str(bank_ref).startswith('USDTXN-') and fx_row.get('flag_FX', False):
            fx_mismatches.append({
                'bank_ref': bank_ref,
                'gl_doc': row['gl_doc'],
                'match_score': row['match_score'],
                'match_status': 'Matched',
                'mismatch_type': 'FX Mismatch',
                'note': f'FX rate variance: {fx_row.get("rate_used", 0):,.0f}'
            })
    
    return pd.DataFrame(fx_mismatches)

def detect_duplicates(matches_df, bank_df):
    """Phát hiện các giao dịch có khả năng trùng lặp"""
    duplicates = []
    bank_df['txn_date'] = pd.to_datetime(bank_df['txn_date'])
    
    for _, row in matches_df.iterrows():
        bank_ref = row['bank_ref']
        bank_txns = bank_df[bank_df['ref'] == bank_ref]
        
        if bank_txns.empty:
            continue
            
        txn = bank_txns.iloc[0]
        
        # Tìm các giao dịch trong cùng cửa sổ thời gian
        time_window = (
            bank_df['txn_date'].between(
                txn['txn_date'] - timedelta(days=DUPLICATE_TIME_WINDOW),
                txn['txn_date'] + timedelta(days=DUPLICATE_TIME_WINDOW)
            )
        )
        
        # Tìm giao dịch có cùng số tiền
        similar_amount = (
            (bank_df['amount'] - txn['amount']).abs() < 0.01
        )
        
        potential_dupes = bank_df[
            time_window & 
            similar_amount & 
            (bank_df['ref'] != bank_ref)  # Không tính chính nó
        ]
        
        if len(potential_dupes) > 0:
            duplicates.append({
                'bank_ref': bank_ref,
                'gl_doc': row['gl_doc'],
                'match_score': row['match_score'],
                'match_status': 'Review',
                'mismatch_type': 'Potential Duplicate',
                'note': f'Similar amount found within {DUPLICATE_TIME_WINDOW} days'
            })
    
    return pd.DataFrame(duplicates)

def detect_partial_payments(matches_df, bank_df, gl_df):
    """Phát hiện và gom nhóm các khoản thanh toán một phần"""
    partial_groups = []
    gl_groups = matches_df.groupby('gl_doc')
    
    for gl_doc, group in gl_groups:
        if 1 < len(group) <= PARTIAL_THRESHOLD:
            try:
                # Tính tổng các giao dịch ngân hàng
                bank_amounts = []
                bank_refs = []
                for _, row in group.iterrows():
                    bank_txns = bank_df[bank_df['ref'] == row['bank_ref']]
                    if not bank_txns.empty:
                        bank_amounts.append(abs(bank_txns.iloc[0]['amount']))
                        bank_refs.append(bank_txns.iloc[0]['ref'])
                
                bank_total = sum(bank_amounts)
                
                # Lấy số tiền GL
                gl_entries = gl_df[gl_df['doc_no'] == gl_doc]
                if gl_entries.empty:
                    continue
                    
                gl_amount = abs(gl_entries.iloc[0]['amount'])
                
                # So sánh tổng
                if abs(bank_total - gl_amount) / gl_amount < 0.01:
                    for bank_ref in bank_refs:
                        row_match = matches_df[matches_df['bank_ref'] == bank_ref].iloc[0]
                        partial_groups.append({
                            'bank_ref': bank_ref,
                            'gl_doc': gl_doc,
                            'match_score': row_match['match_score'],
                            'match_status': 'Matched',
                            'mismatch_type': 'Partial Payment',
                            'note': f'{len(bank_refs)} sub-txns combined'
                        })
            except (IndexError, ZeroDivisionError):
                continue
    
    return pd.DataFrame(partial_groups)

def main():
    print("\n=== BƯỚC 7: HẬU XỬ LÝ LỆCH (FEE/FX/PARTIAL/DUPLICATE) ===")
    
    # 1. Kiểm tra files
    required_files = [IN_MATCHES, IN_BANK, IN_GL, IN_BANK_FX]
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Không tìm thấy file: {file_path}")
            return
    
    # 2. Đọc dữ liệu
    try:
        # Đọc các file đầu vào
        matches_df = pd.read_csv(IN_MATCHES)
        bank_df = pd.read_csv(IN_BANK)
        gl_df = pd.read_csv(IN_GL)
        bank_fx_df = pd.read_csv(IN_BANK_FX)
        
        print(f"✓ Đã tải dữ liệu:")
        print(f"  - Cặp khớp: {len(matches_df):,} cặp")
        print(f"  - Bank statement: {len(bank_df):,} dòng")
        print(f"  - GL entries: {len(gl_df):,} dòng")
        print(f"  - Bank FX data: {len(bank_fx_df):,} dòng")
        
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu: {str(e)}")  
        return

    # 3. Phát hiện các trường hợp đặc biệt
    try:
        # === SỬA ĐỔI: Gán None cho mismatch và '-' cho note khi Matched ===
        def get_default_info(score):
            if score >= 0.8:
                # Matched: mismatch = None, note = '-'
                return 'Matched', np.nan, '-' 
            elif score >= 0.65:
                # Review: Lý do là điểm thấp
                return 'Review', 'Low Score', 'Review required due to low score'
            else:
                # Unmatched: Lý do là điểm thấp
                return 'Unmatched', 'Low Score', 'Unmatched due to very low score'
        
        matches_df['match_status'], matches_df['mismatch_type'], matches_df['note'] = zip(
            *matches_df.apply(lambda x: get_default_info(x['match_score']), axis=1)
        )

        fee_adj = detect_fee_adjustments(matches_df, bank_df)
        print(f"\n✓ Đã phát hiện {len(fee_adj):,} khoản phí cần điều chỉnh")
        
        partial = detect_partial_payments(matches_df, bank_df, gl_df)
        print(f"✓ Đã phát hiện {len(partial):,} khoản thanh toán một phần")
        
        # Gộp tất cả các điều chỉnh
        adjustments = pd.concat([
            matches_df[['bank_ref', 'gl_doc', 'match_score', 'match_status', 'mismatch_type', 'note']],
            fee_adj,
            partial,
        # Giữ lại bản ghi 'last' (tức là 'Fee'/'Partial') nếu có trùng lặp
        ]).drop_duplicates(subset=['bank_ref', 'gl_doc'], keep='last') 
        
        # Sắp xếp theo trạng thái và loại không khớp
        adjustments = adjustments.sort_values(
            by=['match_status', 'mismatch_type'],
            ascending=[True, True]
        )
        
        # Chọn và sắp xếp lại các cột theo yêu cầu
        adjustments = adjustments[[
            'bank_ref', 'gl_doc', 'match_status', 'mismatch_type', 'note'
        ]]
        
        # Ghi file kết quả
        adjustments.to_csv(OUT_ADJUSTED, index=False)
        print(f"\n✓ Đã ghi file kết quả: {OUT_ADJUSTED}")
        
        # In header
        print("\nbank_ref\tgl_doc\tmatch_status\tmismatch_type\tnote")
        
        # In 5 dòng đầu tiên với format được chỉ định
        for _, row in adjustments.head().iterrows():
            print(f"{row['bank_ref']}\t{row['gl_doc']}\t{row['match_status']}\t{row['mismatch_type']}\t{row['note']}")
        
        # In thống kê tổng hợp
        print(f"\n📊 Tổng số cặp khớp: {len(adjustments):,}")
        print(f"  - Matched: {len(adjustments[adjustments['match_status'] == 'Matched']):,}")
        print(f"  - Review: {len(adjustments[adjustments['match_status'] == 'Review']):,}")
        print(f"  - Unmatched: {len(adjustments[adjustments['match_status'] == 'Unmatched']):,}")
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý: {str(e)}")
        return

if __name__ == '__main__':
    main()