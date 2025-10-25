# -*- coding: utf-8 -*-
"""
PHÂN TÍCH DỮ LIỆU VÀ KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY)
Tác giả: Data Analysis Team
Ngày: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# Cấu hình font để hiển thị tiếng Việt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
sns.set_style("whitegrid")

# Tạo thư mục output nếu chưa tồn tại
thư_mục_output = 'code_du_an/output'
os.makedirs(thư_mục_output, exist_ok=True)

print("=" * 80)
print("BƯỚC 1: TẢI DỮ LIỆU")
print("=" * 80)

# Tải dữ liệu từ các file CSV
bảng_ngân_hàng = pd.read_csv('code_du_an/data/bank_stmt.csv')
bảng_sổ_cái = pd.read_csv('code_du_an/data/gl_entries.csv')
bảng_tỷ_giá = pd.read_csv('code_du_an/data/fx_rates.csv')

print(f"✓ bank_stmt.csv: {len(bảng_ngân_hàng)} dòng")
print(f"✓ gl_entries.csv: {len(bảng_sổ_cái)} dòng")
print(f"✓ fx_rates.csv: {len(bảng_tỷ_giá)} dòng")

# In cột của từng file
print(f"\nCột trong bank_stmt.csv: {bảng_ngân_hàng.columns.tolist()}")
print(f"Cột trong gl_entries.csv: {bảng_sổ_cái.columns.tolist()}")
print(f"Cột trong fx_rates.csv: {bảng_tỷ_giá.columns.tolist()}")

print("\n" + "=" * 80)
print("BƯỚC 2: KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY)")
print("=" * 80)

# Tạo báo cáo DQ
danh_sách_dq = []

# Phân tích bank_stmt.csv - tất cả các cột
for cột in bảng_ngân_hàng.columns:
    số_null = bảng_ngân_hàng[cột].isnull().sum()
    phần_trăm_null = (số_null / len(bảng_ngân_hàng)) * 100

    # Lấy Min/Max cho cột date và amount
    if cột in ['txn_date', 'amount']:
        if bảng_ngân_hàng[cột].dtype in ['int64', 'float64']:
            giá_trị_min = bảng_ngân_hàng[cột].min()
            giá_trị_max = bảng_ngân_hàng[cột].max()
        else:
            giá_trị_min = bảng_ngân_hàng[cột].min()
            giá_trị_max = bảng_ngân_hàng[cột].max()
    else:
        giá_trị_min = "-"
        giá_trị_max = "-"

    ghi_chú = "OK" if phần_trăm_null == 0 else f"Thiếu {phần_trăm_null:.1f}%"

    danh_sách_dq.append({
        'File': 'bank_stmt.csv',
        'Column': cột,
        'Null%': f"{phần_trăm_null:.1f}%",
        'Min': giá_trị_min,
        'Max': giá_trị_max,
        'Notes': ghi_chú
    })

# Phân tích gl_entries.csv - tất cả các cột
for cột in bảng_sổ_cái.columns:
    số_null = bảng_sổ_cái[cột].isnull().sum()

    # Đếm cả các giá trị "-" là thiếu dữ liệu (đặc biệt cho cột partner)
    if cột == 'partner':
        số_thiếu = (bảng_sổ_cái[cột] == '-').sum() + số_null
    else:
        số_thiếu = số_null

    phần_trăm_null = (số_thiếu / len(bảng_sổ_cái)) * 100

    # Lấy Min/Max cho cột date và amount
    if cột in ['post_date', 'amount']:
        if bảng_sổ_cái[cột].dtype in ['int64', 'float64']:
            giá_trị_min = bảng_sổ_cái[cột].min()
            giá_trị_max = bảng_sổ_cái[cột].max()
        else:
            giá_trị_min = bảng_sổ_cái[cột].min()
            giá_trị_max = bảng_sổ_cái[cột].max()
    else:
        giá_trị_min = "-"
        giá_trị_max = "-"

    ghi_chú = "OK" if phần_trăm_null == 0 else f"Thiếu {phần_trăm_null:.1f}%"
    if cột == 'partner' and phần_trăm_null > 0:
        ghi_chú = "Thiếu tên khách hàng"

    danh_sách_dq.append({
        'File': 'gl_entries.csv',
        'Column': cột,
        'Null%': f"{phần_trăm_null:.1f}%",
        'Min': giá_trị_min,
        'Max': giá_trị_max,
        'Notes': ghi_chú
    })

# Phân tích fx_rates.csv - tất cả các cột
for cột in bảng_tỷ_giá.columns:
    số_null = bảng_tỷ_giá[cột].isnull().sum()
    phần_trăm_null = (số_null / len(bảng_tỷ_giá)) * 100

    # Lấy Min/Max cho cột date và USD_VND
    if cột in ['date', 'USD_VND']:
        if bảng_tỷ_giá[cột].dtype in ['int64', 'float64']:
            giá_trị_min = bảng_tỷ_giá[cột].min()
            giá_trị_max = bảng_tỷ_giá[cột].max()
        else:
            giá_trị_min = bảng_tỷ_giá[cột].min()
            giá_trị_max = bảng_tỷ_giá[cột].max()
    else:
        giá_trị_min = "-"
        giá_trị_max = "-"

    ghi_chú = "OK" if phần_trăm_null == 0 else f"Thiếu {phần_trăm_null:.1f}%"

    danh_sách_dq.append({
        'File': 'fx_rates.csv',
        'Column': cột,
        'Null%': f"{phần_trăm_null:.1f}%",
        'Min': giá_trị_min,
        'Max': giá_trị_max,
        'Notes': ghi_chú
    })

# Tạo DataFrame báo cáo DQ
báo_cáo_dq = pd.DataFrame(danh_sách_dq)

# Lưu báo cáo DQ (không cần lọc vì đã chỉ lấy các cột quan trọng)
báo_cáo_dq.to_csv(f'{thư_mục_output}/DQ_report.csv', index=False)
print("\n✓ Báo cáo DQ đã lưu: output/DQ_report.csv")
print("\nTóm tắt Chất Lượng Dữ Liệu:")
print(báo_cáo_dq.to_string(index=False))

print("\n" + "=" * 80)
print("BƯỚC 3: CHUẨN BỊ DỮ LIỆU CHO BIỂU ĐỒ")
print("=" * 80)

# Chuyển đổi cột ngày tháng
bảng_ngân_hàng['txn_date'] = pd.to_datetime(bảng_ngân_hàng['txn_date'])
bảng_sổ_cái['post_date'] = pd.to_datetime(bảng_sổ_cái['post_date'])

# Phân loại loại giao dịch
bảng_ngân_hàng['loại_giao_dịch'] = bảng_ngân_hàng['desc'].apply(
    lambda x: 'Học phí AOF' if 'AOF TUITION' in x 
    else 'Phí dịch vụ' if 'SERVICE FEE' in x
    else 'Phí ngân hàng' if 'BANK FEE' in x
    else 'Chờ xử lý' if 'PENDING' in x
    else 'Thanh toán USD'
)

print("✓ Dữ liệu đã được chuẩn bị")

print("\n" + "=" * 80)
print("BƯỚC 4: VẼ CÁC BIỂU ĐỒ")
print("=" * 80)

# ============================================================================
# BIỂU ĐỒ 1: HISTOGRAM - PHÂN BỐ SỐ TIỀN
# ============================================================================
hình_1, trục_1 = plt.subplots(figsize=(12, 6))
số_tiền = bảng_ngân_hàng[bảng_ngân_hàng['amount'] > 0]['amount']
trục_1.hist(số_tiền, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
trục_1.set_xlabel('Số tiền (VND/USD)', fontsize=12, fontweight='bold')
trục_1.set_ylabel('Tần suất', fontsize=12, fontweight='bold')
trục_1.set_title('Histogram - Phan Bo So Tien Giao Dich', fontsize=14, fontweight='bold')
trục_1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{thư_mục_output}/amount_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Histogram đã lưu: output/amount_histogram.png")
plt.close()

# ============================================================================
# BIỂU ĐỒ 2: TIMELINE ZIGZAG - GIAO DỊCH THEO THỜI GIAN
# ============================================================================
hình_2, trục_2 = plt.subplots(figsize=(14, 6))
số_tiền_hàng_ngày = bảng_ngân_hàng.groupby('txn_date')['amount'].sum().sort_index()
trục_2.plot(số_tiền_hàng_ngày.index, số_tiền_hàng_ngày.values, marker='o', 
            linewidth=2.5, markersize=7, color='#e74c3c', label='Tong so tien hang ngay')
trục_2.fill_between(số_tiền_hàng_ngày.index, số_tiền_hàng_ngày.values, 
                     alpha=0.2, color='#e74c3c')
trục_2.set_xlabel('Ngay giao dich', fontsize=12, fontweight='bold')
trục_2.set_ylabel('So tien (VND/USD)', fontsize=12, fontweight='bold')
trục_2.set_title('Zigzag - Duong xu huong giao dich theo thoi gian', fontsize=14, fontweight='bold')
trục_2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f'{thư_mục_output}/txn_timeline.png', dpi=300, bbox_inches='tight')
print("✓ Zigzag Timeline đã lưu: output/txn_timeline.png")
plt.close()

# ============================================================================
# BIỂU ĐỒ 3: BIỂU ĐỒ TRÒN - PHÂN BỐ LOẠI GIAO DỊCH
# ============================================================================
hình_3, trục_3 = plt.subplots(figsize=(10, 8))
phân_bố_loại = bảng_ngân_hàng['loại_giao_dịch'].value_counts()
màu_sắc = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
trục_3.pie(phân_bố_loại.values, labels=phân_bố_loại.index, autopct='%1.1f%%',
           colors=màu_sắc, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
trục_3.set_title('Bieu do Tron - Phan bo loai giao dich', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{thư_mục_output}/txn_type_pie_chart.png', dpi=300, bbox_inches='tight')
print("✓ Biểu đồ Tròn (Loại giao dịch) đã lưu: output/txn_type_pie_chart.png")
plt.close()

# ============================================================================
# BIỂU ĐỒ 4: BIỂU ĐỒ TRÒN - PHÂN BỐ TIỀN TỆ
# ============================================================================
hình_4, trục_4 = plt.subplots(figsize=(10, 8))
phân_bố_tiền_tệ = bảng_ngân_hàng['currency'].value_counts()
màu_tiền_tệ = ['#45B7D1', '#FFA07A']
trục_4.pie(phân_bố_tiền_tệ.values, labels=phân_bố_tiền_tệ.index, autopct='%1.1f%%',
           colors=màu_tiền_tệ, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
trục_4.set_title('Bieu do Tron - Phan bo tien te', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{thư_mục_output}/currency_pie_chart.png', dpi=300, bbox_inches='tight')
print("✓ Biểu đồ Tròn (Tiền tệ) đã lưu: output/currency_pie_chart.png")
plt.close()

# ============================================================================
# BIỂU ĐỒ 5: ZIGZAG TÍCH LŨY - SỐ TIỀN TÍCH LŨY THEO THỜI GIAN
# ============================================================================
hình_5, trục_5 = plt.subplots(figsize=(14, 6))
bảng_sắp_xếp = bảng_ngân_hàng.sort_values('txn_date')
số_tiền_tích_lũy = bảng_sắp_xếp['amount'].cumsum()
trục_5.plot(bảng_sắp_xếp['txn_date'], số_tiền_tích_lũy, marker='o', linewidth=2.5,
            markersize=6, color='#27ae60', label='So tien tich luy')
trục_5.fill_between(bảng_sắp_xếp['txn_date'], số_tiền_tích_lũy, alpha=0.2, color='#27ae60')
trục_5.set_xlabel('Ngay giao dich', fontsize=12, fontweight='bold')
trục_5.set_ylabel('So tien tich luy (VND/USD)', fontsize=12, fontweight='bold')
trục_5.set_title('Zigzag - So tien tich luy theo thoi gian', fontsize=14, fontweight='bold')
trục_5.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f'{thư_mục_output}/cumulative_zigzag.png', dpi=300, bbox_inches='tight')
print("✓ Zigzag Tích lũy đã lưu: output/cumulative_zigzag.png")
plt.close()

print("\n" + "=" * 80)
print("BƯỚC 5: THỐNG KÊ TÓM TẮT")
print("=" * 80)

print(f"\nThong ke Bang Ngan Hang:")
print(f"  Tong so giao dich: {len(bảng_ngân_hàng)}")
print(f"  Khoang thoi gian: {bảng_ngân_hàng['txn_date'].min().date()} den {bảng_ngân_hàng['txn_date'].max().date()}")
print(f"  Tong so tien: {bảng_ngân_hàng['amount'].sum():,.0f}")
print(f"  Trung binh: {bảng_ngân_hàng['amount'].mean():,.0f}")
print(f"  Toi thieu: {bảng_ngân_hàng['amount'].min():,.0f}")
print(f"  Toi da: {bảng_ngân_hàng['amount'].max():,.0f}")

print(f"\nPhan bo loai giao dich:")
for loại, số_lượng in bảng_ngân_hàng['loại_giao_dịch'].value_counts().items():
    phần_trăm = (số_lượng / len(bảng_ngân_hàng)) * 100
    print(f"  {loại}: {số_lượng} ({phần_trăm:.1f}%)")

print(f"\nPhan bo tien te:")
for tiền_tệ, số_lượng in bảng_ngân_hàng['currency'].value_counts().items():
    phần_trăm = (số_lượng / len(bảng_ngân_hàng)) * 100
    print(f"  {tiền_tệ}: {số_lượng} ({phần_trăm:.1f}%)")

print("\n" + "=" * 80)
print("✓ PHAN TICH HOAN TAT!")
print("=" * 80)
print(f"\nCac file output da duoc luu trong: {thư_mục_output}/")
print("  - DQ_report.csv")
print("  - amount_histogram.png")
print("  - txn_timeline.png (Zigzag)")
print("  - txn_type_pie_chart.png (Bieu do Tron)")
print("  - currency_pie_chart.png (Bieu do Tron)")
print("  - cumulative_zigzag.png (Zigzag)")

