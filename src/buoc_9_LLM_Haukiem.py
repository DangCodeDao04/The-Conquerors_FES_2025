# -*- coding: utf-8 -*-
"""
Bước 9 — Gold Layer: LLM hậu kiểm (Sử dụng Qwen)
🎯 Mục tiêu: LLM giúp viết lại mô tả ngắn gọn, gắn nhãn giao dịch chính xác.
📥 Đầu vào:
    - silver_candidates.csv (từ Bước 8)
    - bank_stmt.csv, gl_entries.csv (để tra cứu thông tin gốc)
⚙️ Xử lý:
    - Dùng mô hình Qwen2.5-7B-Instruct để phân tích
    - Chuẩn hóa loại giao dịch (TUITION, FEE, ADJUSTMENT)
    - Tạo explanation dạng JSON
📤 Đầu ra: gold_layer_output.json
"""

import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# === CÀI ĐẶT CẦN THIẾT ===
# Chạy lệnh này trong terminal:
# pip install google-generativeai pandas tqdm

# === THAM SỐ ===
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"  # Hoặc đường dẫn đến model local
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Số lượng cặp xử lý mỗi lần
MAX_NEW_TOKENS = 512  # Độ dài tối đa của câu trả lời

# === ĐƯỜNG DẪN ===
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"
output_dir = root_dir / "output"

# Input files
IN_SILVER = output_dir / "silver_candidates.csv"
IN_BANK = data_dir / "bank_stmt.csv"
IN_GL = data_dir / "gl_entries.csv"



# Output file
OUT_GOLD = output_dir / "gold_layer_output.json"

class TransactionValidator:
    def __init__(self, model_name=MODEL_NAME):
        """Khởi tạo model và tokenizer"""
        print(f"✓ Đang tải model {model_name}...")
        try:
            # Thử tải từ cache local trước
            cache_dir = root_dir / ".cache" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            print("1. Đang tải tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
                use_fast_tokenizer=True,
                padding_side='left'
            )
            
            print("2. Đang tải model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
                torch_dtype=torch.float16,  # Dùng FP16 để giảm bộ nhớ
                low_cpu_mem_usage=True
            )
            
            print("3. Chuyển model sang chế độ eval...")
            self.model.eval()
            
        except Exception as e:
            print(f"\n❌ Lỗi tải model: {str(e)}")
            print("\nGợi ý khắc phục:")
            print("1. Kiểm tra kết nối internet")
            print("2. Thử tải lại nếu bị gián đoạn")
            print("3. Kiểm tra dung lượng ổ đĩa và bộ nhớ")
            print("4. Nếu có sẵn model local, đặt đường dẫn vào MODEL_NAME")
            raise e

    def prepare_prompt(self, bank_info, gl_info):
        """Chuẩn bị prompt cho LLM"""
        # Format amounts safely
        try:
            bank_amount = f"{float(bank_info.get('amount', 0)):,.0f}"
            gl_amount = f"{float(gl_info.get('amount', 0)):,.0f}"
        except:
            bank_amount = "0"
            gl_amount = "0"

        prompt = f"""Analyze these transactions and determine if they match:

Bank Transaction:
- Reference: {str(bank_info.get('ref', ''))}
- Amount: {bank_amount} VND
- Description: {str(bank_info.get('desc', ''))}
- Date: {str(bank_info.get('date', ''))}

GL Entry:
- Document: {str(gl_info.get('doc_no', ''))}
- Amount: {gl_amount} VND
- Partner: {str(gl_info.get('partner', ''))}
- Date: {str(gl_info.get('date', ''))}

Classify as:
- TUITION: Student tuition payments
- FEE: Other service fees and charges 
- ADJUSTMENT: Balance corrections and adjustments

Return only a valid JSON object with these exact fields:
- bank_ref: The bank transaction reference
- gl_doc: The GL document number  
- match_type: One of "TUITION", "FEE", or "ADJUSTMENT"
- explanation: Brief reason why they match

Response must be valid JSON only, no other text."""

        return prompt
        
    @torch.inference_mode()
    def analyze_transaction(self, bank_info, gl_info):
        """Analyze a transaction pair using LLM"""
        try:
            # Prepare prompt
            prompt = self.prepare_prompt(bank_info, gl_info)
            
            # Tokenize với streaming
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)
            
            # Generate với streaming
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                num_beams=1,
                length_penalty=1.0
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Extract JSON more carefully
            try:
                # Tìm và chuẩn hóa phần JSON
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    print(f"⚠️ Không tìm thấy JSON trong response: {response[:100]}...")
                    return None
                    
                json_str = response[start_idx:end_idx]
                # Đảm bảo JSON hợp lệ
                result = json.loads(json_str)
                
                # Kiểm tra các trường bắt buộc
                required_fields = ['bank_ref', 'gl_doc', 'match_type', 'explanation']
                if not all(field in result for field in required_fields):
                    missing = [f for f in required_fields if f not in result]
                    print(f"⚠️ Thiếu các trường JSON: {missing}")
                    return None
                    
                # Kiểm tra match_type hợp lệ
                valid_types = ['TUITION', 'FEE', 'ADJUSTMENT']
                if result['match_type'] not in valid_types:
                    print(f"⚠️ match_type không hợp lệ: {result['match_type']}")
                    return None
                    
                return result
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Lỗi JSON không hợp lệ: {str(e)}")
                print(f"Response: {response[:100]}...")
                return None
                
        except Exception as e:
            print(f"⚠️ Lỗi phân tích giao dịch: {str(e)}")
            return None

def main():
    print("\n=== BƯỚC 9: GOLD LAYER - LLM HẬU KIỂM ===")

    # 1. Kiểm tra files
    required_files = [IN_SILVER, IN_BANK, IN_GL]
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Không tìm thấy file: {file_path}")
            return
            
    # 2. Đọc dữ liệu
    try:
        print("✓ Đang đọc dữ liệu...")
        silver_df = pd.read_csv(IN_SILVER)
        bank_df = pd.read_csv(IN_BANK)
        gl_df = pd.read_csv(IN_GL)
        
        # Chuẩn bị dữ liệu
        bank_df['ref'] = bank_df['ref'].astype(str)
        gl_df['doc_no'] = gl_df['doc_no'].astype(str)
        
        print(f"✓ Đã đọc:")
        print(f"  - {len(silver_df):,} cặp từ silver_candidates.csv")
        print(f"  - {len(bank_df):,} giao dịch từ bank_stmt.csv")
        print(f"  - {len(gl_df):,} bút toán từ gl_entries.csv")
        
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu: {str(e)}")
        return
        
    # 3. Khởi tạo validator
    try:
        validator = TransactionValidator()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {str(e)}")
        return
        
    # 4. Phân tích giao dịch
    print("\n✓ Bắt đầu phân tích giao dịch...")
    gold_results = []
    
    # Chỉ lấy các cặp có điểm cao nhất cho mỗi bank_ref
    top_matches = silver_df.loc[silver_df.groupby('bank_ref')['f_text'].idxmax()]
    
    for _, row in tqdm(top_matches.iterrows(), total=len(top_matches)):
        try:
            bank_ref = str(row['bank_ref'])
            gl_doc = str(row['gl_doc'])
            
            # Lấy thông tin chi tiết
            bank_info = bank_df[bank_df['ref'] == bank_ref].iloc[0].to_dict()
            gl_info = gl_df[gl_df['doc_no'] == gl_doc].iloc[0].to_dict()
            
            # Phân tích bằng LLM
            analysis = validator.analyze_transaction(bank_info, gl_info)
            
            if analysis and all(k in analysis for k in ['bank_ref', 'gl_doc', 'match_type', 'explanation']):
                gold_results.append(analysis)
                
        except Exception as e:
            print(f"⚠️ Lỗi xử lý cặp {bank_ref}-{gl_doc}: {str(e)}")
            continue
            
    # 5. Lưu kết quả
    if not gold_results:
        print("\n❌ Không có kết quả phân tích nào để lưu.")
        return
        
    try:
        # Lưu file JSON
        with open(OUT_GOLD, 'w', encoding='utf-8') as f:
            json.dump(gold_results, f, ensure_ascii=False, indent=2)
            
        print(f"\n✓ Đã lưu {len(gold_results):,} kết quả vào {OUT_GOLD}")
        
        # Thống kê phân loại
        results_df = pd.DataFrame(gold_results)
        print("\n=== THỐNG KÊ PHÂN LOẠI ===")
        print(results_df['match_type'].value_counts())
        
        # In mẫu kết quả
        print("\n=== MẪU KẾT QUẢ ===")
        print(json.dumps(gold_results[0], ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ Lỗi lưu kết quả: {str(e)}")
        return

if __name__ == '__main__':
    main()