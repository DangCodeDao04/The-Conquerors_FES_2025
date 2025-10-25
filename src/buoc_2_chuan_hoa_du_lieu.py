# -*- coding: utf-8 -*-
"""
BƯỚC 2: CHUẨN HÓA MÔ TẢ & TRÍCH "REF"
Làm sạch phần mô tả text để hệ thống dễ so sánh
"""

import pandas as pd
import re
import os
import warnings

warnings.filterwarnings('ignore')

# Tạo thư mục output nếu chưa tồn tại
thư_mục_output = 'code_du_an/output'
os.makedirs(thư_mục_output, exist_ok=True)

print("=" * 80)
print("BƯỚC 2: CHUẨN HÓA MÔ TẢ & TRÍCH 'REF'")
print("=" * 80)

# Tải dữ liệu
print("\n✓ Đang tải dữ liệu...")
bảng_ngân_hàng = pd.read_csv('code_du_an/data/bank_stmt.csv')

print(f"✓ Tải thành công: {len(bảng_ngân_hàng)} dòng")
print(f"✓ Cột: {bảng_ngân_hàng.columns.tolist()}")

# ============================================================================
# HÀM CHUẨN HÓA MÔ TẢ
# ============================================================================

def chuẩn_hóa_mô_tả(text):
    """Chuyển chữ thường, bỏ ký tự đặc biệt"""
    if pd.isna(text):
        return ""
    
    # Chuyển chữ thường
    text = str(text).lower()
    
    # Bỏ ký tự đặc biệt, giữ lại chữ cái, số, khoảng trắng
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def trích_loại_hậu_tố(text):
    """Trích loại hậu tố: CTY, TNHH, LTD, JSC"""
    if pd.isna(text):
        return "-"
    
    text = str(text).upper()
    
    # Danh sách hậu tố
    hậu_tố_list = ['CTY', 'TNHH', 'LTD', 'JSC', 'CO', 'INC']
    
    for hậu_tố in hậu_tố_list:
        if hậu_tố in text:
            return hậu_tố
    
    return "-"


def trích_mã_ref(text):
    """Trích mã ref đầy đủ (ví dụ: TXN-1SWJC, USDTXN-HAFE, FEE-RJRN)"""
    if pd.isna(text):
        return "-"
    
    text = str(text).upper().strip()
    
    # Pattern phổ biến:
    patterns = [
        # TXN-XXXXX, USDTXN-XXXX
        r'(?:USD)?TXN-[A-Z0-9]+',
        # FEE-XXXX
        r'FEE-[A-Z0-9]+',
        # SRV-XXXXX
        r'SRV-[A-Z0-9]+',
        # PEND-XXXXX
        r'PEND-[A-Z0-9]+'
    ]
    
    # Thử từng pattern
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
            
    # Nếu có trong cột ref, trả về nguyên giá trị
    if text.startswith(('TXN-', 'FEE-', 'SRV-', 'PEND-', 'USDTXN-')):
        return text
    
    return "-"


def chuẩn_hóa_tiền_tệ(text):
    """Chuẩn hóa tiền tệ"""
    if pd.isna(text):
        return "VND"
    
    text = str(text).upper().strip()
    
    if text in ['USD', 'VND', 'EUR', 'GBP', 'JPY']:
        return text
    
    return "VND"


# ============================================================================
# BƯỚC 1: CHUẨN HÓA DỮ LIỆU
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: CHUẨN HÓA DỮ LIỆU")
print("=" * 80)

# Tạo các cột mới
bảng_ngân_hàng['desc_clean'] = bảng_ngân_hàng['desc'].apply(chuẩn_hóa_mô_tả)
bảng_ngân_hàng['ref_clean'] = bảng_ngân_hàng['ref'].apply(trích_mã_ref)
bảng_ngân_hàng['loại_hậu_tố'] = bảng_ngân_hàng['desc'].apply(trích_loại_hậu_tố)  # Sửa tên cột
bảng_ngân_hàng['currency_std'] = bảng_ngân_hàng['currency'].apply(chuẩn_hóa_tiền_tệ)

print("✓ Đã chuẩn hóa mô tả (desc_clean)")
print("✓ Đã trích mã ref (ref_clean)")
print("✓ Đã trích loại hậu tố (loại_hậu_tố)")
print("✓ Đã chuẩn hóa tiền tệ (currency_std)")

# ============================================================================
# BƯỚC 2: TẠO FILE OUTPUT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: TẠO FILE OUTPUT")
print("=" * 80)

# Chọn các cột cần thiết
cột_output = ['txn_date', 'amount', 'desc_clean', 'ref_clean', 'loại_hậu_tố', 'currency_std']
bảng_output = bảng_ngân_hàng[cột_output].copy()

# Đổi tên cột
bảng_output.columns = ['txn_date', 'amount', 'desc_clean', 'ref_clean', 'loại_hậu_tố', 'currency']

# Lưu file
bảng_output.to_csv(f'{thư_mục_output}/bank_clean.csv', index=False)
print(f"✓ Đã lưu: {thư_mục_output}/bank_clean.csv")

# ============================================================================
# BƯỚC 3: HIỂN THỊ MẪU DỮ LIỆU
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: MẪU DỮ LIỆU ĐẦU TIÊN")
print("=" * 80)

print("\n📊 Dữ liệu gốc (5 dòng đầu):")
print(bảng_ngân_hàng[['txn_date', 'amount', 'desc', 'ref', 'currency']].head().to_string(index=False))

print("\n📊 Dữ liệu đã chuẩn hóa (5 dòng đầu):")
print(bảng_output.head().to_string(index=False))

# ============================================================================
# BƯỚC 4: THỐNG KÊ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: THỐNG KÊ")
print("=" * 80)

print(f"\n📊 Tổng số giao dịch: {len(bảng_output)}")
print(f"📊 Số cột: {len(bảng_output.columns)}")

print(f"\n📊 Phân bố mã ref:")
for ref, count in bảng_output['ref_clean'].value_counts().items():
    print(f"  {ref}: {count}")

print(f"\n📊 Phân bố loại hậu tố:")
for hậu_tố, count in bảng_output['loại_hậu_tố'].value_counts().items():
    print(f"  {hậu_tố}: {count}")

print(f"\n📊 Phân bố tiền tệ:")
for tiền_tệ, count in bảng_output['currency'].value_counts().items():
    print(f"  {tiền_tệ}: {count}")

print("\n" + "=" * 80)
print("✓ BƯỚC 2 HOÀN TẤT!")
print("=" * 80)
print(f"\nFile output: {thư_mục_output}/bank_clean.csv")

