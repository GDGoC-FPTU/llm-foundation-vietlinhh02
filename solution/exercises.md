# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời trở nên sáng tạo hơn nhưng kém nhất quán và có thể lạc đề. Ở 0.0, mô hình luôn cho ra cùng một câu trả lời . Ở 1.5, câu trả lời có thể bị lặp từ, ngữ pháp bất thường, hoặc chứa thông tin không chính xác do quá ngẫu nhiên.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature khoảng 0.3–0.5. Chatbot hỗ trợ khách hàng cần câu trả lời chính xác, nhất quán và đáng tin cậy — không cần sáng tạo. Temperature thấp giúp giảm hallucination và đảm bảo người dùng nhận được cùng một câu trả lời cho cùng một câu hỏi, tạo trải nghiệm ổn định.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Giả sử mỗi call ~250 input tokens + 100 output tokens. Tổng/ngày: 10.000 × 3 = 30.000 calls → 7.5M input tokens + 3M output tokens.
> - GPT-4o: (7.5M × $5 + 3M × $20) / 1M = $37.50 + $60 = **$97.50/ngày**
> - GPT-4o-mini: (7.5M × $0.15 + 3M × $0.60) / 1M = $1.125 + $1.80 = **$2.925/ngày**
> → GPT-4o đắt hơn khoảng **33 lần**.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> - **GPT-4o xứng đáng:** Khi cần phân tích phức tạp, suy luận đa bước, hoặc viết code — ví dụ trợ lý y tế đưa ra chẩn đoán sơ bộ từ triệu chứng. Sai sót ở đây có hậu quả nghiêm trọng, nên cần model mạnh nhất.
> - **GPT-4o-mini phù hợp hơn:** Các tác vụ đơn giản, lặp lại như phân loại email (spam/not spam), trả lời FAQ, hoặc tóm tắt ngắn. Chi phí thấp cho phép phục vụ hàng triệu request mà chất lượng vẫn đủ tốt.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng thời gian thực nơi người dùng đang chờ phản hồi — như chatbot, trợ lý AI, hoặc công cụ viết văn bản — vì nó tạo trải nghiệm mượt mà, giảm cảm giác chờ đợi khi token được hiển thị ngay khi sinh ra. Ngược lại, non-streaming phù hợp hơn khi xử lý batch (xử lý hàng loạt câu hỏi trong background), khi cần toàn bộ response trước khi xử lý tiếp (ví dụ: parse JSON, đánh giá kết quả), hoặc khi gọi API từ backend mà không có người dùng trực tiếp chờ.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định
