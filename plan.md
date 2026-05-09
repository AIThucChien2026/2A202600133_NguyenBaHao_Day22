# Plan: Day 22 — DPO/ORPO Alignment Lab (Track 3)

Dựa vào yêu cầu và tiến trình của Lab 22 (`README.md`), dưới đây là kế hoạch thực hiện End-to-End được chia thành các Phase rõ ràng:

## Phase 1: Chuẩn bị & Setup Môi trường
- **Mục tiêu:** Cài đặt các công cụ và cấu hình môi trường dựa vào phần cứng hiện có.
- **Chi tiết:**
  - Lựa chọn Compute Tier (`T4` cho GPU miễn phí/nhỏ, hoặc `BigGPU` nếu có máy chủ mạnh).
  - Khởi chạy mã cài đặt (chạy lệnh bash `setup-laptop.sh`, tiếp đó xem xét chạy `make smoke` để kiểm tra khả năng tương thích máy của bạn).

## Phase 2: Khởi tạo mô hình SFT Mini (`01_sft_mini.py`)
- **Mục tiêu:** Fine-tune Base model nhanh bằng phương pháp SFT (1k VN Alpaca).
- **Chi tiết:**
  - Build Checkpoint sử dụng Unsloth + LoRA (r=16, 1 epoch).
  - Xác nhận rằng loss giảm dần liên tục để chắc chắn adapter được train đúng.
- **Tiêu chuẩn hoàn thành (Deliverables):** File Adapter lưu trong `adapters/sft-mini/` và biểu đồ loss (`01_sft_loss.png`).

## Phase 3: Chuẩn bị Dữ liệu Preference (`02_preference_data.py`)
- **Mục tiêu:** Xử lý và chuẩn hoá preference data cho DPO.
- **Chi tiết:**
  - Tải framework data `argilla/ultrafeedback-binarized-preferences-cleaned`.
  - Kết xuất cấu trúc dataframe bao gồm cụ thể các trường: `prompt`, `chosen` và `rejected`.
- **Tiêu chuẩn hoàn thành (Deliverables):** Sinh ra dữ liệu lưu ở folder `data/pref/train.parquet` với 3 example in ra để kiểm tra.

## Phase 4: Tiến Hành Huấn Luyện DPO (`03_dpo_train.py`)
- **Mục tiêu:** Dùng PPO/DPO để Align cho model từ SFT Model và 1 Frozen Reference Model.
- **Chi tiết:**
  - Sử dụng TRL `DPOTrainer(beta=0.1, lr=5e-7)` cho việc training.
  - Theo dõi reward curves qua mỗi step.
- **Tiêu chuẩn hoàn thành (Deliverables):** Adapter lưu ở `adapters/dpo/` kèm biểu đồ reward curves gap mở rộng `03_dpo_reward_curves.png`.

## Phase 5: Đánh giá Helpfulness qua Side-by-Side (`04_compare_and_eval.py`)
- **Mục tiêu:** So sánh đáp án từ model SFT-only và model SFT+DPO.
- **Chi tiết:**
  - Sử dụng chung 8 câu prompt test.
  - Sử dụng Auto-judge (LLM-as-a-judge như GPT-4o, Claude) thủ công hoặc code tự động để đếm win/loss/tie tỷ lệ.
- **Tiêu chuẩn hoàn thành (Deliverables):** Bảng so sánh 2 model gồm 8 ví dụ và report đếm `04_side_by_side_table.png`.

## Phase 6: Hợp nhất (Merge) & Xuất GGUF deployment (`05_merge_deploy_gguf.py`)
- **Mục tiêu:** Xuất mô hình về định dạng nhẹ để đem chạy thực tế.
- **Chi tiết:**
  - Hợp nhất adapter vào base model sử dụng `merge_and_unload()`.
  - Lượng hoá và xuất mô hình file GGUF `Q4_K_M`.
  - Thực hiện unit smoke test với llama-cpp-python.
- **Tiêu chuẩn hoàn thành (Deliverables):** Lấy file test `<5 GB` tại `gguf/lab22-dpo-Q4_K_M.gguf` và file test inference `06_gguf_smoke.png`.

## Phase 7: Định Lượng Benchmark Cuối Cùng (`06_benchmark.py`)
- **Mục tiêu:** Kiểm đo khắt khe bằng bài test benchmark toàn diện.
- **Chi tiết:**
  - Chạy mô hình test bằng IFEval, GSM8K, MMLU, và AlpacaEval-lite trên cả model SFT vs DPO.
  - Xuất biểu đồ so sánh chi tiết.
- **Tiêu chuẩn hoàn thành (Deliverables):** Output thông tin đánh giá lưu vào `data/eval/benchmark_results.json`, có biểu đồ biểu diễn 4 bar `07-benchmark-comparison.png`.

## Phase 8: Nghiệm thu (Reflection) và Nộp Bài 
- **Mục tiêu:** Hoàn thiện sản phẩm cuối khóa.
- **Chi tiết:**
  - Chạy code kiểm tra `make verify` trước khi nộp.
  - Điền template cá nhân (6 sections) tại file `submission/REFLECTION.md`.
  - Submit 6+ ảnh chụp màn hình yêu cầu. Trả repo link bài tập lên hệ thống.
