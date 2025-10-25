# -*- coding: utf-8 -*-
"""
Bước 5 — Chấm điểm (Scoring) tuyến tính (merge đầy đủ dữ liệu + fix missing)
"""
import pandas as pd
from thefuzz import fuzz
from pathlib import Path
from datetime import datetime

# === ĐƯỜNG DẪN ===
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    ROOT_DIR = Path('.').resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

# === HÀM HỖ TRỢ ===
def to_float_safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def calculate_f_date(delta_days, max_days=3):
    d = to_float_safe(delta_days, default=None)
    if d is None:
        return 0.0
    return max(0.0, 1.0 - (abs(d) / float(max_days)))

def calculate_f_amt(delta_amount, bank_amount):
    delta = to_float_safe(delta_amount, default=None)
    bank_amt = to_float_safe(bank_amount, default=None)
    if bank_amt is None or bank_amt == 0 or delta is None:
        return 0.0
    percent_diff = min(1.0, abs(delta) / abs(bank_amt))
    return max(0.0, 1.0 - percent_diff)

def calculate_f_text(bank_desc, gl_partner):
    a = '' if bank_desc is None else str(bank_desc)
    b = '' if gl_partner is None else str(gl_partner)
    if a.strip() == '' and b.strip() == '':
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0

def calculate_f_ref(bank_ref, gl_doc):
    a = '' if bank_ref is None else str(bank_ref)
    b = '' if gl_doc is None else str(gl_doc)
    if a.strip() == '' and b.strip() == '':
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0

def calculate_f_partner(bank_desc, gl_partner):
    a = '' if bank_desc is None else str(bank_desc)
    b = '' if gl_partner is None else str(gl_partner)
    if a.strip() == '' and b.strip() == '':
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0

# Tính delta ngày
def calculate_delta_days(bank_date, gl_date):
    try:
        b = pd.to_datetime(bank_date)
        g = pd.to_datetime(gl_date)
        return (b - g).days
    except:
        return 0

# Tính delta số tiền
def calculate_delta_amount(bank_amount, gl_amount):
    return to_float_safe(bank_amount) - to_float_safe(gl_amount)

def main():
    print("\n=== BƯỚC 5: CHẤM ĐIỂM TUYẾN TÍNH (Merge dữ liệu đầy đủ) ===")

    # --- 1. Load dữ liệu ---
    try:
        candidates_df = pd.read_csv(OUTPUT_DIR / "candidate_pairs_detailed.csv")
        bank_stmt_df = pd.read_csv(DATA_DIR / "bank_stmt.csv")
        gl_df = pd.read_csv(DATA_DIR / "gl_entries.csv")
        print("✓ Đã tải dữ liệu")
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file đầu vào. Chi tiết: {e}")
        return

    # --- 2. Merge dữ liệu đầy đủ ---
    df = pd.merge(candidates_df, bank_stmt_df, left_on='bank_ref', right_on='ref', how='left', suffixes=('', '_bank'))
    df = pd.merge(df, gl_df, left_on='gl_doc', right_on='doc_no', how='left', suffixes=('', '_gl'))

    # --- 3. Tạo cột delta nếu chưa có ---
    if 'Δdays' not in df.columns:
        df['Δdays'] = df.apply(lambda row: calculate_delta_days(row.get('date', None), row.get('doc_date', None)), axis=1)
    if 'Δamount' not in df.columns:
        df['Δamount'] = df.apply(lambda row: calculate_delta_amount(row.get('amount_vnd', None), row.get('amount_vnd_gl', None)), axis=1)
    if 'amount_vnd(bank)' not in df.columns:
        df['amount_vnd(bank)'] = df['amount_vnd'].fillna(0.0)

    # --- 4. Xử lý giá trị rỗng cho text ---
    for col in ['bank_ref', 'gl_doc', 'desc', 'partner']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)

    # --- 5. Tính các feature ---
    df['f_date'] = df.apply(lambda row: calculate_f_date(row['Δdays']), axis=1)
    df['f_amt'] = df.apply(lambda row: calculate_f_amt(row['Δamount'], row['amount_vnd(bank)']), axis=1)
    df['f_text'] = df.apply(lambda row: calculate_f_text(row['desc'], row['partner']), axis=1)
    df['f_ref'] = df.apply(lambda row: calculate_f_ref(row['bank_ref'], row['gl_doc']), axis=1)
    df['f_partner'] = df.apply(lambda row: calculate_f_partner(row['desc'], row['partner']), axis=1)

    print("✓ Đã tính xong các feature")

    # --- 6. Tính Score ---
    weights = {'f_amt': 0.35, 'f_date': 0.20, 'f_text': 0.25, 'f_ref': 0.15, 'f_partner': 0.05}
    df['Score'] = (
        df['f_amt']*weights['f_amt'] +
        df['f_date']*weights['f_date'] +
        df['f_text']*weights['f_text'] +
        df['f_ref']*weights['f_ref'] +
        df['f_partner']*weights['f_partner']
    )

    # --- 7. Xuất kết quả ---
    output_cols = ['bank_ref', 'gl_doc', 'f_amt', 'f_date', 'f_text', 'f_ref', 'f_partner', 'Score']
    final_df = df[output_cols].copy()
    for col in ['f_amt', 'f_date', 'f_text', 'f_ref', 'f_partner', 'Score']:
        final_df[col] = final_df[col].round(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "scored_candidates_full.csv"
    final_df.sort_values('Score', ascending=False).to_csv(output_path, index=False)

    print(f"🎉 Đã xuất file: {output_path}")
    print(final_df.head().to_string())

if __name__ == '__main__':
    main()
