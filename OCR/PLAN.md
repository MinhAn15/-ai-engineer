<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Tạm dừng lộ trình ở đây. Chuyển qua bước tìm hiểu về Document Intelligence, tập trung OCR trước, bắt đầu bằng việc xây dựng một prototype.

Perfect! Tôi đã tìm thấy lộ trình Document Intelligence từ guide. Bây giờ tôi sẽ cung cấp cho bạn một **lộ trình OCR + Document Intelligence prototype cụ thể**, khác với lộ trình Day 5-6 của guide, nhưng tập trung vào xây dựng nhanh từ nền tảng.

## 📋 Lộ trình Document Intelligence - OCR Focus Prototype

### Giai đoạn 1: Nền tảng OCR (1–2 ngày)

1. **Setup \& PDF Processing** – Cài đặt tools, learn pypdf/pdf2image/pytesseract
2. **OCR cơ bản** – Tesseract + Hugging Face (lightweight models)
3. **Text Block Extraction** – Lấy text + bounding box (tọa độ)
4. **Unit Test** – Verify extraction trên 5–10 PDF mẫu

### Giai đoạn 2: Model Council (1–2 ngày)

5. **Multi-model voting** – Kết hợp 2–3 model, chọn output tốt nhất
6. **Confidence scoring** – Tính độ tin cậy từ voting
7. **Performance tracking** – Latency, accuracy, confidence trends

### Giai đoạn 3: UI + Integration (1 ngày)

8. **Streamlit Playground** – Upload PDF, xem kết quả tức thì
9. **Metrics dashboard** – Hiển thị confidence, latency, model votes
10. **Export** – Tải kết quả JSON/CSV

### Giai đoạn 4: RAG Prep (optional, 1 ngày)

11. **Text chunks** – Chia text thành chunks cho RAG
12. **Embedding placeholder** – Chuẩn bị để kết nối embedding model

***

Bạn muốn tôi bắt đầu từ **Giai đoạn 1 – Setup \& OCR cơ bản** không? Tôi sẽ cung cấp:

✅ Step-by-step cài đặt pypdf, pytesseract, pdf2image, transformers
✅ Code để extract text + bounding box từ PDF
✅ So sánh Tesseract vs Hugging Face OCR
✅ Unit test template để verify kết quả

Hay bạn muốn khác? (vd. bắt từ Model Council, hoặc chỉ Tesseract trước)
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: 10-DAY-AI-ENGINEER-FULL-GUIDE.md

