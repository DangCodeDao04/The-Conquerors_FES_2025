Hệ thống Đối chiếu Tự động Bank-GL (Bank-GL Auto-Reconciliation System)

Đây là tài liệu mô tả kỹ thuật cho pipeline tự động đối chiếu sao kê ngân hàng (bank_stmt.csv) và sổ cái kế toán (gl_entries.csv).

Luồng xử lý Pipeline (Pipeline Flow)

Hệ thống được xây dựng qua một chuỗi các bước xử lý (dựa trên các file Python):

phan_tich_du_lieu.py (Bước 1: DQ):

Tải dữ liệu thô (bank_stmt.csv, gl_entries.csv).

Thực hiện phân tích chất lượng dữ liệu (Data Quality), kiểm tra null, min/max, và tạo các biểu đồ phân tích (histogram, timeline).


buoc_2_chuan_hoa_du_lieu.py (Bước 2: Chuẩn hóa):

Text Cleaning: Chuẩn hóa mô tả (description) về chữ thường, loại bỏ các hậu tố doanh nghiệp (ví dụ: "cty", "tnhh", "jsc"...) và các ký tự đặc biệt.

Ref Extraction: Trích xuất mã tham chiếu (ref_clean) từ mô tả để tăng cường khả năng khớp.


buoc_3_chinh_sach_fx.py (Bước 3: Quy đổi FX):

Tải file fx_rates.csv.

Đối với các giao dịch bank không phải VND (ví dụ: USD), hệ thống sử dụng phương thức pandas.merge_asof để tìm tỷ giá gần nhất (nearest) theo ngày.

Tạo cột amount_vnd chuẩn hóa cho tất cả giao dịch.


buoc_4_sinh_vien_ung.py (Bước 4: Sinh Ứng viên):

Đây là bước lọc hiệu suất cao để giảm không gian tìm kiếm.

Một cặp Bank-GL chỉ được coi là "ứng viên" nếu thỏa mãn đồng thời hai điều kiện:

Biên độ ngày: DAY_TOL = 10 (lệch tối đa ±10 ngày).

Biên độ số tiền: Lệch tối đa AMT_TOL_PCT = 0.05 (±5%) hoặc AMT_TOL_ABS = 10_000 (±10,000 VND), lấy theo ngưỡng lớn hơn.

Đầu ra là file candidate_pairs_detailed.csv chứa các cặp tiềm năng.


buoc_5_cham_diem_v2.py (Bước 5: Chấm điểm Tuyến tính):

Tính 5 tín hiệu thành phần (f_amt, f_date, f_text, f_ref, f_partner) cho mỗi cặp ứng viên.

Áp dụng công thức trọng số tuyến tính để tính điểm Score tổng hợp.

Đầu ra là file scored_candidates_full.csv.


buoc_6_gan_cap_toi_uu.py (Bước 6: Gán cặp Tối ưu):

Sử dụng thuật toán Hungarian (scipy.optimize.linear_sum_assignment) để tìm ra tổ hợp gán cặp 1-1 tốt nhất, tối đa hóa tổng điểm.

Áp dụng ngưỡng để phân loại:

Score >= 0.45 $\to$ Matched

0.40 <= Score < 0.45 $\to$ Review

Score < 0.40 $\to$ Unmatched

Đầu ra là file matched_pairs.csv.


buoc_7_hau_xu_ly.py (Bước 7: Hậu xử lý):

Xử lý các trường hợp đặc biệt như Lệch Phí (FEE), Lệch Tỷ giá (FX), Thanh toán một phần (PARTIAL), và Trùng lặp (DUPLICATE).


buoc_8_silver.py (Bước 8: Silver Layer):

(Nâng cao) Sử dụng mô hình hybrid BM25 (từ khóa) và SentenceTransformer (ngữ nghĩa) để cải thiện điểm f_text cho các mô tả phức tạp.


Buoc_6.1_Tinh_KPI_new.py (Bước 6.1/8.1: Tính KPI):

So sánh kết quả matched_pairs.csv với answer_key_sample.csv để tính các KPI cuối cùng.

Các Tham số Cốt lõi (Key Parameters)

Đây là các quy tắc nghiệp vụ và tham số kỹ thuật chính được định nghĩa trong mã nguồn.

1. Quy đổi Ngoại tệ (FX Conversion)

File: buoc_3_chinh_sach_fx.py

Logic: Tìm tỷ giá trong fx_rates.csv có ngày gần nhất (nearest) với ngày giao dịch ngân hàng (txn_date).

Công thức: amount_vnd = amount_usd * rate_used

2. Ngưỡng lọc Ứng viên (Candidate Filtering)

File: buoc_4_sinh_vien_ung.py

Biên độ Ngày: Tối đa ±10 ngày (DAY_TOL = 10).

Biên độ Số tiền: max(Số tiền * 5%, 10,000 VND).

3. Công thức Chấm điểm (Scoring Formula)

File: buoc_5_cham_diem_v2.py

Công thức:

Score = (f_amt     * 0.35) +
        (f_date    * 0.20) +
        (f_text    * 0.25) +
        (f_ref     * 0.15) +
        (f_partner * 0.05)


Lưu ý: Tín hiệu f_date được tính dựa trên độ lệch so với max_days=3, nghĩa là một giao dịch lệch 4 ngày (nhưng vẫn qua vòng lọc 10 ngày) sẽ nhận 0 điểm f_date.

4. Ngưỡng Gán cặp (Matching Thresholds)

File: buoc_6_gan_cap_toi_uu.py

Matched: Score >= 0.45

Review: Score >= 0.40 (và < 0.45)

Unmatched: Score < 0.40

Cách thực thi (How to Run)

Chạy các file Python theo thứ tự từ Bước 2 đến Bước 8. Các file đầu ra của bước trước là đầu vào của bước sau, được lưu trữ trong thư mục /output.

python phan_tich_du_lieu.py (Tùy chọn, để xem DQ)

python buoc_2_chuan_hoa_du_lieu.py

python buoc_3_chinh_sach_fx.py

python buoc_4_sinh_vien_ung.py

python buoc_5_cham_diem_v2.py

python buoc_6_gan_cap_toi_uu.py

python Buoc_6.1_Tinh_KPI_new.py (Để xem KPI của mô hình Bronze)

(Tùy chọn) python buoc_7_hau_xu_ly.py

(Tùy chọn) python buoc_8_silver.py

Chỉ số đo lường (KPIs)

Hệ thống được đánh giá dựa trên các KPI chính (tính toán trong Buoc_6.1_Tinh_KPI_new.py):

Auto-match Rate: Tỷ lệ giao dịch được hệ thống tự động gán là Matched.

Precision (Độ chính xác): Trong số các cặp Matched, bao nhiêu cặp là đúng (so với answer_key).

Recall (Độ phủ): Hệ thống tìm thấy được bao nhiêu % trong tổng số các cặp đúng.

F1-Score: Trung bình điều hòa của Precision và Recall.