# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Throughput (tin/s) | p50 (ms/tin) | p95 (ms/tin) | Chi phí (USD/1k tin) |
|---|---|---|---|---|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 1.000 | 0.763 | 0.866 | 0.889 | 6.506 | 17,749.0 | 0.045 | 0.062 | 0.000 |
| **2. BoW + MLP (deep learning)** | 0.988 | 0.871 | 0.926 | 0.947 | 4.186 | 68,399.5 | 0.176 | 0.215 | 0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.917 | 0.946 | 0.931 | 1.000 | 2.581 | 166.7 | 27.546 | 58.375 | 0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.943 | 0.882 | 0.911 | 0.947 | 3.629 | 1,356.7 | 6.396 | 7.441 | 0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.930 | 0.860 | 0.894 | 0.947 | 3.579 | 1,518.8 | 6.500 | 9.951 | 0.000 |
| **4a. LLM nhỏ zero-shot** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.384 | 677.862 | 1,050.499 | 0.025 |
| **4c. LLM nhỏ few-shot** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.548 | 619.590 | 939.679 | 0.064 |

## 3. Dự đoán và thực tế

1. **Thí nghiệm 3.A — `emb_search` (Tìm kiếm theo ngữ nghĩa)**
   - **Bạn đoán gì:** Đoán câu truy vấn `"you have been selected to receive a complimentary holiday, reply now to claim"` sẽ ra kết quả gần nhất là `ham` do câu không hề chứa các từ khóa spam kinh điển trong từ điển như *free*, *win*, *prize*.
   - **Ra gì:** Toàn bộ 6 kết quả có độ tương đồng cosine cao nhất (0.531 – 0.563) đều mang nhãn `spam`.
   - **Vì sao lệch:** Mô hình Sentence Transformer (`all-MiniLM-L6-v2`) ánh xạ câu vào không gian vector dày đặc (dense embedding) biểu diễn ngữ nghĩa thay vì chỉ so khớp chuỗi ký tự. Mô hình hiểu rằng cụm *"complimentary holiday"* đồng nghĩa với *"free holiday"* và mang cùng cấu trúc, ngữ cảnh chiêu dụ nhận giải thưởng với các tin spam trong tập train.

2. **Thí nghiệm 2.A — `dl_curve` (MLP so với Logistic Regression)**
   - **Bạn đoán gì:** MLP sẽ có F1 kém hơn Logistic Regression vì cho rằng trên tập dữ liệu nhỏ với biểu diễn Bag-of-Words thưa, mạng nơ-ron có nhiều tham số sẽ nhanh chóng bị overfit nặng nề.
   - **Ra gì:** MLP đạt F1 test = 0.926, vượt trội rõ rệt so với BoW + LogReg (0.866); đặc biệt Recall tăng vọt từ 0.763 lên 0.871 trong khi Precision vẫn duy trì ở mức 0.988.
   - **Vì sao lệch:** Lớp ẩn 64 nơ-ron kết hợp hàm phi tuyến ReLU đã tự động học được mối tương quan giữa các tổ hợp từ xuất hiện đồng thời (word co-occurrence/n-gram ngầm định, ví dụ từ ngữ hối thúc kết hợp ký hiệu tiền tệ), giúp giảm mạnh số ca bỏ sót (False Negative) mà mô hình tuyến tính đơn giản không nắm bắt được.

3. **Thí nghiệm 5 — `tradeoff` (Đường Pareto đánh đổi chất lượng và độ trễ)**
   - **Bạn đoán gì:** Phương án 3b (Encoder + LogReg) sẽ nằm trên đường Pareto vì là điểm cân bằng tối ưu giữa khả năng hiểu ngữ nghĩa và tốc độ suy luận nhanh của bộ phân loại tuyến tính.
   - **Ra gì:** Đường Pareto thực tế chỉ gồm: `1. BoW + LogReg`, `2. BoW + MLP` và `3a. Encoder + kNN`. Phương án 3b rơi vào nhóm bị lấn át (dominated).
   - **Vì sao lệch:** Mô hình BoW + MLP có độ trễ cực nhỏ (p95 chỉ 0.22 ms) nhưng đạt F1 trên tập SUB tương đương 3b (0.947), đồng thời F1 trên toàn bộ tập test (0.926) cao hơn 3b (0.911). Chi phí forward-pass qua 6 tầng Transformer đẩy p95 của 3b lên 7.44 ms mà không mang lại ưu thế vượt trội về F1 trên tập SMS này, khiến 3b bị BoW + MLP lấn át hoàn toàn.

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** Các mô hình dựa trên Bag-of-Words (BoW + LogReg và BoW + MLP) tụt điểm nghiêm trọng nhất, tập trung ở các nhóm `obfuscation` (né từ khóa), `natural_spam` (spam văn phong tự nhiên) và `ham_looks_spam` (tin nhắn bạn bè chứa từ khóa giật gân). Trong khi đó, LLM giữ độ chính xác cao ở văn phong tự nhiên nhưng có nguy cơ bị vượt mặt ở nhóm `prompt_injection`.
- **Case 1 (Nhóm Obfuscation — Spam):**
  - *Nội dung:* `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"` (Nhãn thực tế: `spam`).
  - *Cơ chế sai:* Kỹ thuật Leetspeak (thay chữ bằng số như `0` cho `o`, `1` cho `i`, `4` cho `a`) biến toàn bộ từ khóa quan trọng thành các token lạ chưa từng có trong từ điển tập train (Out-Of-Vocabulary). Vector BoW bị gán giá trị 0 ở tất cả vị trí từ khóa nhạy cảm. Do đó, điểm logit không thể vượt qua ngưỡng quyết định khiến mô hình đoán nhầm thành `ham`.
- **Case 2 (Nhóm Ham looks spam — Ham):**
  - *Nội dung:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thực tế: `ham`).
  - *Cơ chế sai:* Tin nhắn bạn bè nhưng chứa đồng thời 3 từ có trọng số dương rất lớn kéo mạnh về nhãn spam: `won`, `call`, `congrats`. Mô hình BoW chỉ cộng dồn độc lập các trọng số $w \cdot x + b$ mà không phân tích được ngữ cảnh đời thường (*"lol"*, *"beer"*), khiến tổng xác suất vượt ngưỡng 0.5 và bị chặn nhầm thành `spam` (False Positive).

## 5. Thử thách cá nhân

- **Đề theo SEED 5359:** Tối ưu hóa ngưỡng quyết định (Decision Threshold Tuning) kết hợp điều chỉnh trọng số phạt mất cân bằng lớp (Class Weight Balancing) cho mô hình BoW + Logistic Regression nhằm giải quyết bài toán lệch lớp (~12.7% spam), cải thiện Recall mà không làm suy giảm mạnh Precision.
- **Cách làm:**
  1. Thay vì cố định ngưỡng mặc định $threshold = 0.5$, thực hiện quét ngưỡng phân loại từ 0.1 đến 0.9 (bước nhảy 0.02) trên tập validation để chọn điểm cắt tối ưu F1-score.
  2. Bổ sung hệ số phạt mất cân bằng lớp vào gradient hàm mất mát Binary Cross-Entropy trong hàm `train_logreg`: nhân trọng số dương $w_{pos} = \frac{N_{ham}}{N_{spam}} \approx 6.9$ đối với các ca spam bị phân loại sai hoặc đoán thiếu.
- **Kết quả:**
  - Ngưỡng mặc định 0.5: Test Precision = 1.000, Recall = 0.763, F1 = 0.866 (bỏ sót 22 tin spam trên tổng số 93 tin spam ở tập test).
  - Sau khi áp dụng trọng số lớp và điều chỉnh về ngưỡng tối ưu $threshold = 0.28$: Test Precision đạt 0.953, Recall tăng vọt lên 0.882, điểm F1-score trên tập test tăng từ **0.866 lên 0.916** (tăng 5.0% F1, bắt thêm được 11 tin spam bị lọt lưới).
- **Giải thích:** Dữ liệu có tỷ lệ chênh lệch lớn khiến hàm mục tiêu thông thường ưu tiên tối ưu hóa lớp đa số (ham), dẫn đến hiện tượng mô hình quá dè dặt khi gán nhãn spam. Việc tăng trọng số phạt lớp thiểu số và hạ ngưỡng phân loại giúp nâng cao độ nhạy cảm trước các đặc trưng rủi ro yếu, khắc phục hạn chế thiếu sót thông tin của biểu diễn BoW nhị phân.

## 6. Phương án cho sản phẩm thật

Với yêu cầu độ trễ p95 < 50 ms/tin cho quy mô 10 triệu SMS/ngày:

- **Phương án lựa chọn:** Triển khai **Mô hình BoW + MLP (Deep Learning)** chạy độc lập on-premise, hoặc thiết kế kiến trúc phễu 2 tầng (Cascading Pipeline): Tầng 1 sử dụng BoW + MLP xử lý 100% lưu lượng để phân loại tức thì các tin nhắn rõ ràng; Tầng 2 chỉ chuyển khoảng 3–5% lượng tin nhắn rơi vào vùng phân vân không chắc chắn ($0.35 \le p \le 0.65$) sang mô hình **Encoder (MiniLM) + LogReg** để rà soát ngữ nghĩa chuyên sâu.
- **Dẫn chứng số liệu thực nghiệm:**
  - *Về tốc độ và thông lượng:* BoW + MLP đạt độ trễ p95 chỉ **0.22 ms/tin**, nhanh gấp hơn 220 lần so với trần SLA 50 ms. Throughput xử lý lô đạt **68,399.5 tin/s**, có thể xử lý toàn bộ lưu lượng 10 triệu tin nhắn trong chưa đầy 2.5 phút tính toán trên CPU/GPU cơ bản.
  - *Về chất lượng:* F1 test đạt **0.926** (Precision 0.988, Recall 0.871), vượt xa BoW + LogReg (0.866) và nhỉnh hơn cả Encoder + LogReg (0.911).
  - *So sánh với LLM API:* LLM nhỏ (zero-shot) có độ trễ p95 lên đến **1,050.50 ms** (vi phạm SLA gấp 21 lần), chỉ xử lý được 1.38 tin/s khi gọi tuần tự và tiêu tốn khoảng **$250 USD/ngày** (~$7,500 USD/tháng cho 10 triệu tin). LLM hoàn toàn bất khả thi cho gateway viễn thông thời gian thực.
- **Đánh đổi chấp nhận:**
  - Chấp nhận bỏ lọt một số ít trường hợp lừa đảo mạo danh tinh vi dạng zero-shot đòi hỏi khả năng lập luận ngôn ngữ tự nhiên bậc cao của LLM.
  - Đổi lại hệ thống đạt độ ổn định 100%, bảo mật dữ liệu tuyệt đối (không gửi SMS của khách hàng qua API bên thứ ba), không phát sinh chi phí biến đổi theo token và đảm bảo phản hồi tức thì dưới 1 ms.

## 7. AI log và điều đã học

- **AI Log:**
  - *Lần 1:* Yêu cầu AI hướng dẫn cấu hình OpenAI-compatible cho mô hình suy luận `gemini-3.5-flash-lite`. AI đề xuất đặt `max_tokens=3`. Nhận diện model reasoning cần không gian token để tư duy trước khi xuất kết quả nên nếu đặt 3 sẽ phát sinh lỗi 400 hoặc cắt cụt phản hồi. Đã sửa lại thành `max_tokens=512` kết hợp `reasoning_effort="low"`.
  - *Lần 2:* Nhờ AI viết hàm trích xuất nhãn `parse_label`. AI viết `return 1 if "spam" in response.lower() else 0`. Nhận thấy đây là bẫy logic nghiêm trọng vì các chuỗi như *"not spam"* hoặc giải thích *"this is not spam"* sẽ bị gắn nhãn 1 (False Positive). Đã sửa lại logic kiểm tra chuỗi phủ định và *"ham"* trước, sau đó mới xét chuỗi *"spam"*.
  - *Lần 3:* Nhờ AI xử lý hàm `llm_classify_batch` khi bị giới hạn tần suất (Rate Limit 503). Đã can thiệp thêm cơ chế `retry_with_backoff` kết hợp khoảng nghỉ `time.sleep(4.1)` giữa các tin nhắn để chạy an toàn trong hạn mức Free Tier của Google AI Studio.

Không phải mô hình AI càng lớn hay càng hiện đại thì càng tối ưu cho sản phẩm thực tế; độ trễ hệ thống, thông lượng và chi phí vận hành mới là yếu tố quyết định bài toán kỹ thuật. Biểu diễn đặc trưng (representation) đóng vai trò then chốt: việc chuyển từ túi từ rời rạc sang không gian vector ngữ nghĩa (dense embedding) giải quyết triệt để rào cản đồng nghĩa và ngữ cảnh mà không cần tiêu tốn tài nguyên cho mô hình sinh ngôn ngữ. Ngoài ra, độ chính xác cao trên tập dữ liệu cùng phân phối chưa phản ánh đầy đủ năng lực của mô hình; việc kiểm thử qua các trường hợp tấn công biên (adversarial/obfuscation) là yêu cầu bắt buộc trước khi đưa hệ sinh thái AI vào môi trường sản xuất.
