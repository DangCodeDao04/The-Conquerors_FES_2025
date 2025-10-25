# -*- coding: utf-8 -*-
"""
buoc_6_gan_cap_toi_uu.py (Phiên bản đã xóa các cột f_*)

Hàm thực hiện bước 6: ghép cặp tối ưu (Hungarian) giữa `bank_ref` và `gl_doc`.
Script này chỉ yêu cầu 3 cột đầu vào: 'bank_ref', 'gl_doc', 'Score'.

Hướng dẫn ngắn (PowerShell):
python src/buoc_6_gan_cap_toi_uu.py --input output/scored_candidates_full.csv
"""

from pathlib import Path
import argparse
import traceback
import sys

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_INPUT = 'scored_candidates.csv'
DEFAULT_OUTPUT = 'matched_pairs.csv'


def find_input_file(name: str, script_dir: Path) -> Path:
    """Tìm tệp input ở một số vị trí hợp lý trong workspace."""
    candidates = [
        Path(name),
        script_dir / name,
        script_dir.parent / 'output' / name,
        Path.cwd() / name,
        Path.cwd() / 'output' / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def generate_explanation(bank_ref: str, gl_doc: str, score: float, match_type: str) -> str:
    """Tạo explanation dựa trên rule-based matching patterns."""
    if pd.isna(bank_ref) or pd.isna(gl_doc):
        return 'No matching partner found'
    
    bank_str = str(bank_ref)
    gl_str = str(gl_doc)
    
    # Rule 1: Service transactions
    if bank_str.startswith('SRV-') and gl_str.startswith('SRVRCPT-'):
        if match_type == 'MATCH':
            return f'Service transaction match: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        else:
            return f'Service transaction low confidence: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
    
    # Rule 2: Regular transactions  
    elif bank_str.startswith('TXN-') and gl_str.startswith('RCPT-'):
        if match_type == 'MATCH':
            return f'Regular transaction match: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        else:
            return f'Regular transaction low confidence: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
    
    # Rule 3: FX transactions
    elif bank_str.startswith('TXN-') and gl_str.startswith('RCPTFX-'):
        if match_type == 'MATCH':
            return f'Foreign exchange transaction match: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        else:
            return f'Foreign exchange transaction low confidence: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
    
    # Rule 4: Fee transactions
    elif bank_str.startswith('FEE-') and gl_str.startswith('BANKFEE-'):
        if match_type == 'MATCH':
            return f'Bank fee transaction match: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        else:
            return f'Bank fee transaction low confidence: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
    
    # Default case
    else:
        if match_type == 'MATCH':
            return f'Hungarian algorithm match: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        elif match_type == 'REVIEW':
            return f'Requires manual review: {bank_str} ↔ {gl_str} (Score: {score:.3f})'
        else:
            return f'No suitable match found: {bank_str} ↔ {gl_str} (Score: {score:.3f})'


def match_candidates(input_path: Path, output_dir: Path, output_name: str,
                     threshold_matched: float = 0.75, threshold_review: float = 0.65) -> Path:
    """Đọc tệp input, chạy Hungarian và ghi kết quả ra CSV."""
    # Đọc dữ liệu và chỉ lấy các cột cần thiết
    df_all = pd.read_csv(input_path)
    print(f"✅ Đã đọc tệp: {input_path} (dòng: {len(df_all)})")

    # Hỗ trợ cả cột 'Score' hoặc 'S'
    if 'Score' in df_all.columns:
        score_col = 'Score'
    elif 'S' in df_all.columns:
        score_col = 'S'
    else:
        raise KeyError("Tệp input phải có cột điểm tên 'Score' hoặc 'S'.")

    if not {'bank_ref', 'gl_doc', score_col}.issubset(set(df_all.columns)):
        raise KeyError("Tệp input phải có các cột: 'bank_ref', 'gl_doc' và cột điểm.")

    df = df_all[['bank_ref', 'gl_doc', score_col]].rename(columns={score_col: 'Score'})

    # Pivot thành ma trận điểm (hàng: bank_ref, cột: gl_doc)
    score_matrix = df.pivot_table(index='bank_ref', columns='gl_doc', values='Score', aggfunc='max').fillna(0)
    score_matrix = score_matrix.astype(float)

    # Đảm bảo ma trận vuông bằng cách padding các hàng/cols với 0
    rows = score_matrix.index.tolist()
    cols = score_matrix.columns.tolist()
    n_rows = len(rows)
    n_cols = len(cols)
    n = max(n_rows, n_cols)
    if n == 0:
        raise ValueError('Không có dữ liệu để ghép.')

    # Tạo ma trận vuông với chỉ số mở rộng
    full_index = list(rows) + [f'__pad_row_{i}' for i in range(n - n_rows)]
    full_columns = list(cols) + [f'__pad_col_{i}' for i in range(n - n_cols)]
    square = pd.DataFrame(0.0, index=full_index, columns=full_columns)
    # copy existing scores
    square.loc[rows, cols] = score_matrix.values
    score_matrix = square

    # Ma trận chi phí = 1 - điểm (càng nhỏ càng ưu tiên)
    cost_matrix = 1.0 - score_matrix

    # Chạy thuật toán Hungarian
    row_ind, col_ind = linear_sum_assignment(cost_matrix.values)
    print(f"✅ Thuật toán Hungarian thực thi xong. Số cặp gán: {len(row_ind)}")

    matched_results = []
    assigned_bank_refs = set()
    assigned_gl_docs = set()

    for r, c in zip(row_ind, col_ind):
        bank_ref = score_matrix.index[r]
        gl_doc = score_matrix.columns[c]
        score = float(score_matrix.iat[r, c])

        # Nếu ghép với một pad row/col thì coi như không có partner thực tế
        is_pad_pair = (str(bank_ref).startswith('__pad_row_') or str(gl_doc).startswith('__pad_col_'))

        # Ghi lại assigned nếu không phải pad
        if not str(bank_ref).startswith('__pad_row_'):
            assigned_bank_refs.add(bank_ref)
        if not str(gl_doc).startswith('__pad_col_'):
            assigned_gl_docs.add(gl_doc)

        # Nếu là ghép giữa 2 pad thì bỏ qua
        if is_pad_pair:
            continue

        # Bỏ qua các cặp không có điểm
        if score == 0:
            match_type = 'UNMATCHED'
            explanation = 'No matching score available'
        else:
            if score >= threshold_matched:
                match_type = 'MATCH'
            elif score >= threshold_review:
                match_type = 'REVIEW'
            else:
                match_type = 'UNMATCHED'

        # Tạo explanation dựa trên rule-based matching
        explanation = generate_explanation(bank_ref, gl_doc, score, match_type)

        matched_results.append({
            'bank_ref': bank_ref if not str(bank_ref).startswith('__pad_row_') else None,
            'gl_doc': gl_doc if not str(gl_doc).startswith('__pad_col_') else None,
            'match_score': score if score != 0 else np.nan,
            'match_type': match_type,
            'explanation': explanation,
        })

    # Thêm các mục chưa được ghép vào kết quả
    all_bank_refs = set(df['bank_ref'].unique())
    all_gl_docs = set(df['gl_doc'].unique())

    for b in sorted(all_bank_refs - assigned_bank_refs):
        matched_results.append({'bank_ref': b, 'gl_doc': None, 'match_score': np.nan, 'match_type': 'UNMATCHED', 'explanation': f'No matching GL document found for {b}'})

    for g in sorted(all_gl_docs - assigned_gl_docs):
        matched_results.append({'bank_ref': None, 'gl_doc': g, 'match_score': np.nan, 'match_type': 'UNMATCHED', 'explanation': f'No matching bank transaction found for {g}'})

    # Tạo DataFrame và sắp xếp kết quả
    out_df = pd.DataFrame(matched_results)
    # Chuẩn hoá cột
    if 'match_status' in out_df.columns and 'match_type' not in out_df.columns:
        out_df = out_df.rename(columns={'match_status': 'match_type'})
    # Sắp xếp: MATCH, REVIEW, UNMATCHED
    type_order = {'MATCH': 0, 'REVIEW': 1, 'UNMATCHED': 2}
    out_df['__sort_order'] = out_df['match_type'].map(type_order).fillna(99)
    out_df = out_df.sort_values(by=['__sort_order', 'match_score'], ascending=[True, False]).drop(columns=['__sort_order'])

    # Lưu file và in thống kê
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    # Thêm cột `matched_type` theo yêu cầu (MATCH -> MATCHED)
    out_df['matched_type'] = out_df['match_type'].map({'MATCH': 'MATCHED', 'REVIEW': 'REVIEW', 'UNMATCHED': 'UNMATCHED'})

    # Viết với các cột theo yêu cầu (thêm matched_type và explanation)
    out_df = out_df[['bank_ref', 'gl_doc', 'match_score', 'match_type', 'matched_type', 'explanation']]
    out_df.to_csv(out_path, index=False)

    print(f"\n🎉 Hoàn tất! Kết quả lưu tại: {out_path}")
    print('\n📊 Thống kê kết quả:')
    print(out_df['match_type'].value_counts())
    return out_path


def main(argv=None):
    """Hàm chính để xử lý tham số dòng lệnh và chạy logic ghép cặp."""
    parser = argparse.ArgumentParser(description='Bước 6: Ghép cặp tối ưu giữa bank_ref và gl_doc (tiếng Việt)')
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT, help='Đường dẫn tới scored_candidates_full.csv')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT, help='Tên tệp csv đầu ra')
    parser.add_argument('--outdir', default=str(Path(__file__).resolve().parent.parent / 'output'), help='Thư mục lưu kết quả')
    parser.add_argument('--matched', type=float, default=0.75, help='Ngưỡng để gán Matched (default 0.75)')
    parser.add_argument('--review', type=float, default=0.60, help='Ngưỡng để gán Review (default 0.60)')
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    input_path = find_input_file(args.input, script_dir)
    
    if input_path is None:
        print(f"❌ Không tìm thấy tệp input '{args.input}'. Hãy đặt tệp trong thư mục project/output hoặc thư mục hiện hành.")
        sys.exit(2)

    try:
        match_candidates(input_path, Path(args.outdir), args.output, args.matched, args.review)
    except Exception as e:
        print(f"😕 Lỗi trong quá trình xử lý: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()