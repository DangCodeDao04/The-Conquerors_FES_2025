# -*- coding: utf-8 -*-
"""
BƯỚC 3: CHÍNH SÁCH FX (USD→VND)
Đảm bảo các giao dịch USD được quy đổi chính xác sang VND
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

# Xác định đường dẫn tuyệt đối
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
thư_mục_output = os.path.join(project_dir, 'output')
thư_mục_data = os.path.join(project_dir, 'data')

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(thư_mục_output, exist_ok=True)

print("=" * 80)
print("BƯỚC 3: CHÍNH SÁCH FX (USD→VND)")
print("=" * 80)

# Tải dữ liệu
print("\n✓ Đang tải dữ liệu...")
try:
    # Đọc dữ liệu từ file test (tạm thời)
    bảng_clean = pd.read_csv(os.path.join(thư_mục_data, 'test_input.csv'))
    
    # Tạo bảng tỷ giá mẫu nếu chưa có
    if not os.path.exists(os.path.join(thư_mục_data, 'fx_rates.csv')):
        fx_data = {
            'date': pd.date_range(start='2025-02-01', end='2025-02-28'),
            'USD_VND': [24500] * 28  # Giá USD cố định cho test
        }
        bảng_tỷ_giá = pd.DataFrame(fx_data)
        bảng_tỷ_giá.to_csv(os.path.join(thư_mục_data, 'fx_rates.csv'), index=False)
    else:
        bảng_tỷ_giá = pd.read_csv(os.path.join(thư_mục_data, 'fx_rates.csv'))
        
except FileNotFoundError as e:
    print(f"❌ Lỗi: Không tìm thấy file dữ liệu. Vui lòng kiểm tra đường dẫn: {e}")
    print(f"Thư mục hiện tại: {os.getcwd()}")
    print(f"Đang tìm ở: {thư_mục_output}, {thư_mục_data}")
    exit()

# Kiểm tra và đảm bảo các cột cần thiết tồn tại
required_columns = ['txn_date', 'amount', 'desc_clean', 'ref_clean']
missing_columns = [col for col in required_columns if col not in bảng_clean.columns]
if missing_columns:
    print(f"❌ Lỗi: Thiếu các cột sau trong dữ liệu giao dịch: {', '.join(missing_columns)}")
    exit()

print(f"✓ Đã đọc {len(bảng_clean)} dòng từ file giao dịch")
print(f"✓ Đã đọc {len(bảng_tỷ_giá)} dòng từ fx_rates.csv")

# ============================================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU TỶ GIÁ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: CHUẨN BỊ DỮ LIỆU TỶ GIÁ")
print("=" * 80)

# Chuyển đổi sang datetime, loại bỏ múi giờ và chỉ lấy phần DATE
bảng_tỷ_giá['date'] = pd.to_datetime(bảng_tỷ_giá['date'], errors='coerce')
if bảng_tỷ_giá['date'].dt.tz is not None:
    bảng_tỷ_giá['date'] = bảng_tỷ_giá['date'].dt.tz_localize(None)
bảng_tỷ_giá['date'] = bảng_tỷ_giá['date'].dt.normalize()

bảng_clean['txn_date'] = pd.to_datetime(bảng_clean['txn_date'], errors='coerce')
if bảng_clean['txn_date'].dt.tz is not None:
    bảng_clean['txn_date'] = bảng_clean['txn_date'].dt.tz_localize(None)
bảng_clean['txn_date'] = bảng_clean['txn_date'].dt.normalize()

# Loại bỏ các dòng bị lỗi chuyển đổi ngày
bảng_tỷ_giá.dropna(subset=['date'], inplace=True)
bảng_clean.dropna(subset=['txn_date'], inplace=True)

# Sắp xếp theo ngày (cần thiết cho logic tìm kiếm sau này)
bảng_tỷ_giá = bảng_tỷ_giá.sort_values('date').reset_index(drop=True)

if len(bảng_tỷ_giá) > 0:
    print(f"✓ Khoảng tỷ giá: {bảng_tỷ_giá['date'].min().date()} → {bảng_tỷ_giá['date'].max().date()}")
    print(f"✓ Tỷ giá min: {bảng_tỷ_giá['USD_VND'].min():,.0f}")
    print(f"✓ Tỷ giá max: {bảng_tỷ_giá['USD_VND'].max():,.0f}")
else:
    print("❌ Lỗi: Bảng tỷ giá rỗng sau khi làm sạch. Không thể thực hiện quy đổi FX.")
    exit()

# ============================================================================
# HÀM TÌM TỶ GIÁ GẦN NHẤT (ĐÃ SỬA ĐỔI LOGIC)
# ============================================================================

def tìm_tỷ_giá_gần_nhất(ngày_giao_dịch, bảng_tỷ_giá):
    """
    Tìm tỷ giá có ngày gần nhất tuyệt đối với ngày giao dịch.
    - Ưu tiên ngày chính xác.
    - Nếu không có, tính khoảng cách tuyệt đối và chọn ngày gần nhất.
    """
    
    # Tìm ngày chính xác
    ngày_chính_xác = bảng_tỷ_giá[bảng_tỷ_giá['date'] == ngày_giao_dịch]
    if not ngày_chính_xác.empty:
        return ngày_chính_xác['USD_VND'].iloc[0]
    
    # Tính khoảng cách tuyệt đối (timedelta)
    bảng_tỷ_giá['delta'] = abs(bảng_tỷ_giá['date'] - ngày_giao_dịch)
    
    # Tìm dòng có khoảng cách nhỏ nhất
    gần_nhất = bảng_tỷ_giá.loc[bảng_tỷ_giá['delta'].idxmin()]
    
    # Cảnh báo nếu khoảng cách quá lớn (ví dụ: > 30 ngày, tùy theo chính sách)
    # if gần_nhất['delta'].days > 30:
    #     print(f"⚠️ Cảnh báo: Tỷ giá cho ngày {ngày_giao_dịch.date()} quá xa ({gần_nhất['delta'].days} ngày).")
    
    if gần_nhất['delta'].days >= 0: # Đảm bảo tìm được ngày nào đó
        return gần_nhất['USD_VND']
    
    # Nếu không tìm thấy
    print(f"⚠️ Cảnh báo: Không tìm thấy tỷ giá nào cho ngày {ngày_giao_dịch.date()}. Trả về 0.0.")
    return 0.0

# ============================================================================
# BƯỚC 2: QUY ĐỔI USD → VND
# (Phần này giữ nguyên logic quy đổi và flag)
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: QUY ĐỔI USD → VND")
print("=" * 80)

# Tạo bảng output
bảng_fx = bảng_clean.copy()

# Khởi tạo các cột mới
bảng_fx['amount_usd'] = 0.0
bảng_fx['rate_used'] = 0.0
bảng_fx['amount_vnd'] = 0.0
bảng_fx['flag_FX'] = False
bảng_fx['currency'] = 'VND'  # Mặc định là VND

# Tính tỷ giá trung bình chung trước vòng lặp
tỷ_giá_trung_bình = bảng_tỷ_giá['USD_VND'].mean()

# Xử lý từng dòng
for idx, row in bảng_fx.iterrows():
    # Kiểm tra ref_clean để xác định giao dịch USD
    is_usd = str(row['ref_clean']).startswith('USDTXN-')
    
    if is_usd:
        amount_usd = row['amount']
        
        # Kiểm tra số tiền hợp lệ
        if pd.isna(amount_usd) or amount_usd == 0:
            print(f"⚠️ Cảnh báo: Giao dịch USD có số tiền không hợp lệ: {row['txn_date']} - {row['ref_clean']}")
            continue
        
        # Tìm tỷ giá gần nhất
        rate_used = tìm_tỷ_giá_gần_nhất(row['txn_date'], bảng_tỷ_giá)
        
        # Kiểm tra tỷ giá hợp lệ
        if rate_used == 0:
            print(f"⚠️ Cảnh báo: Không tìm thấy tỷ giá cho giao dịch: {row['txn_date']} - {row['ref_clean']}")
            continue
        
        # Quy đổi sang VND và làm tròn
        amount_vnd = round(amount_usd * rate_used, 0)
        
        # Tính chênh lệch so với tỷ giá trung bình
        if rate_used > 0 and tỷ_giá_trung_bình > 0:
            chênh_lệch_phần_trăm = abs(rate_used - tỷ_giá_trung_bình) / tỷ_giá_trung_bình * 100
            flag_fx = chênh_lệch_phần_trăm > 0.6
        else:
            flag_fx = False
        
        # Gán giá trị
        bảng_fx.at[idx, 'amount_usd'] = amount_usd
        bảng_fx.at[idx, 'rate_used'] = rate_used
        bảng_fx.at[idx, 'amount_vnd'] = amount_vnd
        bảng_fx.at[idx, 'flag_FX'] = flag_fx
        bảng_fx.at[idx, 'currency'] = 'USD'  # Đánh dấu là giao dịch USD
    else:
        # Giao dịch VND giữ nguyên số tiền
        bảng_fx.at[idx, 'amount_usd'] = 0.0
        bảng_fx.at[idx, 'rate_used'] = '—'  # Dùng dấu gạch ngang cho đẹp
        bảng_fx.at[idx, 'amount_vnd'] = row['amount']
        bảng_fx.at[idx, 'flag_FX'] = False

print("✓ Đã quy đổi USD → VND")

# ============================================================================
# BƯỚC 3: TẠO FILE OUTPUT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: TẠO FILE OUTPUT")
print("=" * 80)

# Chọn và sắp xếp các cột theo yêu cầu
cột_output = [
    'txn_date',      # Ngày giao dịch
    'amount',        # Số tiền gốc
    'desc_clean',    # Mô tả đã làm sạch
    'ref_clean',     # Mã tham chiếu
    'rate_used',     # Tỷ giá sử dụng
    'amount_vnd',    # Số tiền quy đổi VND
    'flag_FX'        # Cờ đánh dấu chênh lệch tỷ giá
]
bảng_output = bảng_fx[cột_output].copy()

# Format lại một số cột
bảng_output['txn_date'] = pd.to_datetime(bảng_output['txn_date']).dt.strftime('%Y-%m-%d')
bảng_output['amount'] = bảng_output['amount'].round(2)
bảng_output['amount_vnd'] = bảng_output['amount_vnd'].round(0)

# Lưu file
output_path = os.path.join(thư_mục_output, 'bank_fx.csv')
bảng_output.to_csv(output_path, index=False)
print(f"✓ Đã lưu: {output_path}")

# Hiển thị mẫu kết quả
print("\nMẫu dữ liệu đầu ra:")
print(bảng_output.head().to_string(index=False))

# ============================================================================
# BƯỚC 4: HIỂN THỊ MẪU DỮ LIỆU
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: MẪU DỮ LIỆU")
print("=" * 80)

# Lọc chỉ giao dịch USD
bảng_usd = bảng_output[bảng_output['currency'] == 'USD'].head(10)

print("\n📊 Giao dịch USD (10 dòng đầu):")
print(bảng_usd[['txn_date', 'amount_usd', 'rate_used', 'amount_vnd', 'flag_FX']].to_string(index=False))

# ============================================================================
# BƯỚC 5: THỐNG KÊ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: THỐNG KÊ")
print("=" * 80)

# Thống kê giao dịch USD
giao_dịch_usd = bảng_output[bảng_output['currency'] == 'USD']
giao_dịch_vnd = bảng_output[bảng_output['currency'] == 'VND']

print(f"\n📊 Tổng giao dịch: {len(bảng_output)}")
print(f"  - USD: {len(giao_dịch_usd)}")
print(f"  - VND: {len(giao_dịch_vnd)}")

if len(giao_dịch_usd) > 0:
    print(f"\n📊 Giao dịch USD:")
    print(f"  - Tổng USD: {giao_dịch_usd['amount_usd'].sum():,.0f}")
    print(f"  - Tổng VND: {giao_dịch_usd['amount_vnd'].sum():,.0f}")
    print(f"  - Tỷ giá trung bình: {giao_dịch_usd['rate_used'].mean():,.0f}")
    print(f"  - Tỷ giá min: {giao_dịch_usd['rate_used'].min():,.0f}")
    print(f"  - Tỷ giá max: {giao_dịch_usd['rate_used'].max():,.0f}")

# Thống kê flag FX
flag_count = bảng_output['flag_FX'].sum()
print(f"\n📊 Giao dịch có chênh lệch tỷ giá >0.6%: {flag_count}")

if flag_count > 0:
    print("\n⚠️ Danh sách giao dịch cần kiểm tra:")
    bảng_flag = bảng_output[bảng_output['flag_FX'] == True][['txn_date', 'amount_usd', 'rate_used', 'amount_vnd']]
    print(bảng_flag.to_string(index=False))

print("\n" + "=" * 80)
print("✓ BƯỚC 3 HOÀN TẤT!")
print("=" * 80)
print(f"\nFile output: {thư_mục_output}/bank_fx.csv")