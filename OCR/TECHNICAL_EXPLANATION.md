# Giải Thích Kỹ Thuật Chi Tiết - OCR Document Processing Platform

---

## 📚 Mục Lục

1. [Giới Thiệu Tổng Quan](#1-giới-thiệu-tổng-quan)
2. [OCR là gì? Tại sao cần OCR?](#2-ocr-là-gì-tại-sao-cần-ocr)
3. [Kỹ Thuật Core: Hybrid OCR](#3-kỹ-thuật-core-hybrid-ocr)
4. [Layout Analysis: Hiểu Cấu Trúc Tài Liệu](#4-layout-analysis-hiểu-cấu-trúc-tài-liệu)
5. [Reading Order Detection: Thứ Tự Đọc](#5-reading-order-detection-thứ-tự-đọc)
6. [Relationship Detection: Mối Quan Hệ Phân Cấp](#6-relationship-detection-mối-quan-hệ-phân-cấp)
7. [Table Detection: Phát Hiện Bảng](#7-table-detection-phát-hiện-bảng)
8. [LLM Extraction: Trích Xuất Dữ Liệu](#8-llm-extraction-trích-xuất-dữ-liệu)
9. [Kiến Trúc Hệ Thống](#9-kiến-trúc-hệ-thống)
10. [Giá Trị và Hiệu Quả](#10-giá-trị-và-hiệu-quả)

---

## 1. Giới Thiệu Tổng Quan

### 1.1 Vấn Đề Cần Giải Quyết

Hãy tưởng tượng bạn làm việc tại một công ty và mỗi ngày nhận được hàng trăm hóa đơn, hợp đồng, biểu mẫu. Bạn cần:

- Đọc từng tài liệu
- Tìm thông tin quan trọng (tên công ty, ngày tháng, số tiền...)
- Nhập vào hệ thống

**Vấn đề**: Công việc này tốn rất nhiều thời gian, dễ sai sót, và nhàm chán.

### 1.2 Giải Pháp: OCR Document Processing Platform

Hệ thống này tự động hóa toàn bộ quy trình:

```
Tài liệu (PDF/Ảnh) → OCR → Layout Analysis → Data Extraction → JSON Output
```

**Kết quả**: Thay vì mất 5-10 phút/tài liệu, giờ chỉ cần vài giây!

---

## 2. OCR là gì? Tại sao cần OCR?

### 2.1 Định Nghĩa

**OCR (Optical Character Recognition)** = Nhận dạng ký tự quang học.

Đơn giản: **Biến ảnh thành văn bản**.

```
┌─────────────────┐          ┌─────────────────┐
│  Ảnh chứa chữ   │  ──OCR──>│  Text: "Hello"  │
│    "Hello"      │          │                 │
└─────────────────┘          └─────────────────┘
```

### 2.2 Tại sao máy tính không thể "đọc" ảnh trực tiếp?

Khi bạn nhìn vào tấm ảnh có chữ "A", bạn thấy chữ A.

Nhưng với máy tính:

```
Ảnh = Ma trận số (pixels)

Ví dụ ảnh 5x5 pixel:
[255, 0,   0,   0,   255]   ← Dòng đen với 2 điểm trắng ở 2 đầu
[255, 255, 0,   255, 255]   ← Dòng gần như trắng
[255, 255, 0,   255, 255]
[0,   0,   0,   0,   0  ]   ← Dòng ngang (đường kẻ ngang chữ A)
[255, 0,   0,   0,   255]

(255 = trắng, 0 = đen)
```

Máy tính chỉ thấy **con số**, không thấy "ý nghĩa". OCR giúp máy tính hiểu ý nghĩa của các pixels này.

### 2.3 Tesseract OCR - Engine Chính

**Tesseract** là phần mềm OCR mã nguồn mở, được Google phát triển.

**Cách hoạt động (đơn giản hóa)**:

1. **Binarization**: Chuyển ảnh màu → đen trắng
2. **Line Detection**: Tìm các dòng văn bản
3. **Word Segmentation**: Tách từng từ
4. **Character Recognition**: Nhận dạng từng ký tự
5. **Language Model**: Sử dụng từ điển để sửa lỗi

```
Ảnh gốc → Đen trắng → Tìm dòng → Tách từ → Nhận ký tự → Sửa lỗi → Text
```

**Ưu điểm của Tesseract**:

- Miễn phí, mã nguồn mở
- Hỗ trợ 100+ ngôn ngữ (kể cả tiếng Việt)
- Nhanh, có thể chạy offline

**Nhược điểm**:

- Kém với chữ viết tay
- Gặp khó khăn với ảnh mờ, nghiêng
- Không tốt với font đặc biệt

---

## 3. Kỹ Thuật Core: Hybrid OCR

### 3.1 Vấn Đề với OCR Đơn Lẻ

Tesseract rất tốt nhưng không hoàn hảo. Xem ví dụ:

```
Ảnh gốc: "Hóa đơn số: 12345"

Tesseract output:
- Đoạn 1: "Hóa đơn số:" → Confidence: 95% ✓ (tin cậy cao)
- Đoạn 2: "l2345" → Confidence: 45% ✗ (tin cậy thấp, "1" bị nhận nhầm thành "l")
```

### 3.2 Giải Pháp: Hybrid OCR

**Ý tưởng**: Kết hợp nhiều engine OCR để bù đắp điểm yếu của nhau.

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID OCR PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                             │
│  │  Ảnh đầu vào │                                             │
│  └──────┬──────┘                                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────────────────────────────┐ │
│  │ Tesseract   │───▶│ Blocks với Confidence Score          │ │
│  │ OCR         │    │                                       │ │
│  └─────────────┘    │ [95%] "Hóa đơn số:"                   │ │
│                     │ [45%] "l2345" ← LOW CONFIDENCE        │ │
│                     └──────────────────┬──────────────────┘ │
│                                        │                     │
│                     Filter: Confidence < 75%                 │
│                                        │                     │
│                                        ▼                     │
│                     ┌─────────────────────────────────────┐ │
│                     │ Crop vùng low-confidence             │ │
│                     │ → Gửi lên Gemini Vision              │ │
│                     └──────────────────┬──────────────────┘ │
│                                        │                     │
│                                        ▼                     │
│                     ┌─────────────────────────────────────┐ │
│                     │ Gemini Vision trả về: "12345"        │ │
│                     │ (AI nhận dạng chính xác hơn)         │ │
│                     └──────────────────┬──────────────────┘ │
│                                        │                     │
│                                        ▼                     │
│                     ┌─────────────────────────────────────┐ │
│                     │ MERGE kết quả:                       │ │
│                     │ "Hóa đơn số: 12345" ✓                │ │
│                     └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Tại sao không dùng Gemini cho toàn bộ?

| Phương pháp    | Tốc độ    | Chi phí  | Độ chính xác |
| -------------- | --------- | -------- | ------------ |
| Tesseract only | Rất nhanh | Miễn phí | 80-90%       |
| Gemini only    | Chậm      | Tốn tiền | 95-99%       |
| **Hybrid**     | **Nhanh** | **Rẻ**   | **95-99%**   |

**Logic**: Tesseract xử lý 80% content dễ (miễn phí), chỉ gửi 20% content khó lên Gemini.

---

## 4. Layout Analysis: Hiểu Cấu Trúc Tài Liệu

### 4.1 Vấn Đề

OCR chỉ cho bạn **text**. Nhưng một tài liệu có nhiều hơn text:

```
┌──────────────────────────────────────────┐
│  INVOICE                     ← Title     │
│  ─────────────────                       │
│  Company: ABC Corp           ← Text      │
│  Date: 2024-01-15            ← Text      │
│                                          │
│  ┌─────────────────────────┐             │
│  │ Product │ Qty │ Price   │ ← Table    │
│  │─────────│─────│─────────│             │
│  │ Widget  │ 10  │ $100    │             │
│  └─────────────────────────┘             │
│                                          │
│  [✓] I agree to terms        ← Checkbox │
│                                          │
│  ────────────────                        │
│  Signature: [scribble]       ← Signature │
└──────────────────────────────────────────┘
```

Nếu không có Layout Analysis, bạn chỉ có:

```
"INVOICE Company ABC Corp Date 2024-01-15 Product Qty Price Widget 10 $100 I agree to terms Signature"
```

→ **Không thể hiểu đâu là tiêu đề, đâu là bảng!**

### 4.2 Multi-Stage Layout Pipeline

Hệ thống sử dụng nhiều "stage" để phân loại layout:

```
Stage 1: Reading Order Detection
         ↓
Stage 2: Layout Foundation Model (Table Transformer)
         ↓
Stage 3: Relationship Detection
         ↓
Stage 4: Multi-Method Table Detection
         ↓
      OUTPUT: Segments with type + reading_order + relationships
```

### 4.3 Các Loại Layout Được Phát Hiện

| Type        | Màu       | Mô tả                     | Ví dụ               |
| ----------- | --------- | ------------------------- | ------------------- |
| `text`      | 🟢 Green  | Đoạn văn bản thông thường | "Company: ABC Corp" |
| `title`     | 🔴 Red    | Tiêu đề, header           | "INVOICE"           |
| `table`     | 🟠 Orange | Bảng dữ liệu              | Bảng sản phẩm       |
| `figure`    | 🔵 Blue   | Hình ảnh, biểu đồ         | Logo công ty        |
| `checkbox`  | 🟢 Lime   | Ô check                   | ☑ I agree           |
| `form`      | 🩵 Cyan    | Form field                | Input boxes         |
| `list`      | 🟡 Yellow | Danh sách                 | Bullet points       |
| `signature` | 🟣 Purple | Chữ ký                    | Khu vực ký tên      |

---

## 5. Reading Order Detection: Thứ Tự Đọc

### 5.1 Vấn Đề

Khi đọc tài liệu có nhiều cột, con người biết đọc từ trên xuống dưới, từ trái sang phải **trong mỗi cột**.

```
┌─────────────────────────────────────────────┐
│  Column 1            │  Column 2            │
│  ─────────────       │  ─────────────       │
│  Paragraph 1A        │  Paragraph 2A        │
│                      │                      │
│  Paragraph 1B        │  Paragraph 2B        │
│                      │                      │
│  Paragraph 1C        │  Paragraph 2C        │
└─────────────────────────────────────────────┘

Thứ tự đọc đúng: 1A → 1B → 1C → 2A → 2B → 2C
Thứ tự đọc sai:   1A → 2A → 1B → 2B → 1C → 2C
```

### 5.2 Giải Pháp: Column Detection Algorithm

```python
# Pseudo-code
def detect_reading_order(segments):
    # Bước 1: Phát hiện các cột (columns)
    columns = detect_columns_by_x_overlap(segments)

    # Bước 2: Sort columns từ trái sang phải
    columns.sort(by=x_position)

    # Bước 3: Trong mỗi cột, sort từ trên xuống dưới
    for column in columns:
        column.sort(by=y_position)

    # Bước 4: Gán reading_order
    order = 1
    for column in columns:
        for segment in column:
            segment.reading_order = order
            segment.column = column.index
            order += 1

    return segments
```

### 5.3 X-Overlap Clustering

Làm sao biết 2 segment thuộc cùng 1 cột?

```
Segment A: x=50, width=150  → x_range = (50, 200)
Segment B: x=60, width=140  → x_range = (60, 200)

Overlap = intersection / union
        = (200-60) / (200-50)
        = 140 / 150
        = 0.93 (93%)

Nếu overlap > 0.3 (30%) → Cùng cột!
```

---

## 6. Relationship Detection: Mối Quan Hệ Phân Cấp

### 6.1 Vấn Đề

Trong tài liệu thực, các thành phần có **mối quan hệ** với nhau:

```
┌─────────────────────────────────────────────┐
│                                             │
│   [HÌNH ẢNH SẢN PHẨM]     ← Figure         │
│                                             │
│   Hình 1: Sản phẩm Widget  ← Caption       │
│                                             │
│   BẢNG GIÁ                 ← Table Header  │
│   ┌─────────────────────┐                   │
│   │ Sản phẩm │ Giá      │  ← Table         │
│   │──────────│──────────│                   │
│   │ Widget   │ $100     │  ← Table Cell    │
│   └─────────────────────┘                   │
│                                             │
└─────────────────────────────────────────────┘
```

**Mối quan hệ**:

- "Hình 1: Sản phẩm Widget" là **caption** của hình ảnh
- "BẢNG GIÁ" là **header** của bảng
- "Widget" và "$100" là **cells** trong bảng

### 6.2 Thuật Toán Phát Hiện

```python
def detect_caption_figure(chunks):
    """Caption thường nằm ngay dưới/trên hình ảnh"""

    figures = filter(chunks, type='figure')
    texts = filter(chunks, type='text')

    for figure in figures:
        figure_bottom = figure.y + figure.height

        for text in texts:
            text_top = text.y
            distance = abs(text_top - figure_bottom)

            # Nếu text nằm trong vòng 30px dưới figure
            # VÀ có overlap ngang > 50%
            if distance < 30 and horizontal_overlap(figure, text) > 0.5:
                text.relationship = "caption"
                text.parent = figure
                figure.children.append(text)
```

### 6.3 Các Loại Relationships

| Relationship   | Parent | Child | Điều kiện                                 |
| -------------- | ------ | ----- | ----------------------------------------- |
| `caption`      | Figure | Text  | Text ngay dưới/trên figure, overlap > 50% |
| `table_header` | Table  | Title | Title ngay trên table, trong 50px         |
| `table_cell`   | Table  | Text  | Text nằm trong bbox của table             |
| `form_label`   | Label  | Value | Pattern "Label: Value" trên cùng dòng     |

---

## 7. Table Detection: Phát Hiện Bảng

### 7.1 Thách Thức

Bảng có nhiều dạng:

**Dạng 1: Có đường kẻ rõ ràng**

```
┌─────────┬─────────┐
│ Name    │ Price   │
├─────────┼─────────┤
│ Widget  │ $100    │
└─────────┴─────────┘
```

**Dạng 2: Không có đường kẻ (Invoice style)**

```
Invoice No:     12345
Date:           2024-01-15
Amount:         $1,500.00
```

### 7.2 Multi-Method Detection

Hệ thống sử dụng 3 phương pháp:

#### Method 1: Grid Line Detection (OpenCV)

```python
# Pseudo-code
def detect_grid_tables(image):
    # Bước 1: Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bước 2: Tìm đường ngang
    horizontal_kernel = np.ones((1, 40))  # Kernel dài 40px
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Bước 3: Tìm đường dọc
    vertical_kernel = np.ones((40, 1))  # Kernel cao 40px
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    # Bước 4: Combine và tìm contours
    table_mask = horizontal_lines + vertical_lines
    contours = cv2.findContours(table_mask)

    # Bước 5: Filter contours có diện tích đủ lớn
    tables = [c for c in contours if cv2.contourArea(c) > 10000]

    return tables
```

#### Method 2: Alignment-Based Detection

```python
def detect_by_alignment(ocr_blocks):
    # Bước 1: Group blocks theo Y (cùng dòng)
    rows = group_by_y(ocr_blocks, tolerance=15)

    # Bước 2: Kiểm tra mỗi row
    table_rows = []
    for row in rows:
        if len(row) >= 2:  # Ít nhất 2 cột
            if is_label_value_pattern(row):  # Pattern "Label: Value"
                table_rows.append(row)

    # Bước 3: Merge consecutive rows thành table
    return merge_rows_into_tables(table_rows)
```

#### Method 3: Data Pattern Detection

```python
def detect_by_data_pattern(ocr_blocks):
    rows = group_into_rows(ocr_blocks)

    for i in range(len(rows) - 1):
        row_group = rows[i:i+3]  # Lấy 3 rows liên tiếp

        # Kiểm tra cấu trúc nhất quán
        if has_consistent_structure(row_group):
            # Kiểm tra có cột số không (giá tiền, số lượng...)
            if has_numeric_columns(row_group):
                # Đây là table!
                return create_table_region(row_group)

def is_numeric(text):
    """Kiểm tra text có phải số không (kể cả format tiền)"""
    clean = text.replace(',', '').replace('$', '').replace('VND', '')
    try:
        float(clean)
        return True
    except:
        return False
```

### 7.3 Deduplication

Khi chạy 3 methods, có thể detect **cùng 1 table** nhiều lần. Giải pháp:

```python
def deduplicate_tables(tables):
    # Sort theo confidence (cao → thấp)
    tables.sort(key=lambda t: t.confidence, reverse=True)

    result = []
    for table in tables:
        # Kiểm tra có overlap với table đã có không
        overlaps = any(iou_overlap(table, existing) > 0.5 for existing in result)

        if not overlaps:
            result.append(table)  # Chỉ giữ table có confidence cao nhất

    return result
```

---

## 8. LLM Extraction: Trích Xuất Dữ Liệu

### 8.1 Vấn Đề

Sau khi có OCR text, làm sao extract thông tin cụ thể?

```
OCR Output:
"INVOICE #12345
Company: ABC Corporation
Date: January 15, 2024
Ship to: 123 Main Street
Total: $1,500.00"

Cần extract:
{
    "invoice_number": "12345",
    "company": "ABC Corporation",
    "date": "2024-01-15",
    "total": 1500.00
}
```

### 8.2 Giải Pháp: Schema-Based LLM Extraction

Sử dụng **Gemini LLM** với prompt được thiết kế đặc biệt:

```python
prompt = """
Analyze this OCR text and extract information based on the schema.

OCR Text (with segment IDs):
[1] INVOICE #12345
[2] Company: ABC Corporation
[3] Date: January 15, 2024
[4] Ship to: 123 Main Street
[5] Total: $1,500.00

Schema to extract:
- invoice_number
- company_name
- date
- total_amount

Output format (TOON):
@ENTITIES
field_name: value [ref:segment_id]
"""
```

### 8.3 TOON Format

**TOON (Typed Object Oriented Notation)** là format output đặc biệt:

```
@ENTITIES
invoice_number: 12345 [ref:1]
company_name: ABC Corporation [ref:2]
date: 2024-01-15 [ref:3]
total_amount: $1,500.00 [ref:5]

@RELATIONS
belongs_to: invoice_number -> company_name
```

**Ưu điểm của TOON**:

1. **Traceability**: `[ref:1]` cho biết dữ liệu lấy từ segment nào → có thể highlight trên ảnh gốc
2. **Structured**: Dễ parse thành JSON
3. **Relations**: Có thể định nghĩa quan hệ giữa các entities

---

## 9. Kiến Trúc Hệ Thống

### 9.1 Frontend (Next.js)

```
frontend/
├── src/app/
│   ├── page.tsx          # Dashboard - hiển thị danh sách documents
│   ├── login/page.tsx    # Trang đăng nhập
│   └── documents/[id]/   # Chi tiết document
├── src/components/
│   ├── Sidebar.tsx       # Navigation menu
│   └── NeurondLogo.tsx   # Logo component
└── src/lib/
    └── api.ts            # Axios client gọi API
```

**Tại sao dùng Next.js?**

- Server-side rendering → SEO tốt
- File-based routing → Dễ tổ chức code
- TypeScript support → Type safety

### 9.2 Backend (FastAPI)

```
api/
├── main.py               # Entry point, CORS, middleware
├── routes/ocr.py         # API endpoints
├── models.py             # Database models (User, Document)
├── database.py           # SQLite connection
└── dependencies.py       # JWT authentication

src/
├── pdf_processor.py      # PDF → Image conversion
├── hybrid_ocr.py         # Tesseract + Gemini
├── layout_analysis.py    # Multi-stage layout pipeline
├── reading_order_detector.py   # Reading order
├── relationship_detector.py    # Hierarchical relationships
├── table_detector.py     # Grid + Alignment + Pattern detection
└── llm_extractor.py      # Gemini extraction
```

**Tại sao dùng FastAPI?**

- Async support → Xử lý nhiều requests đồng thời
- Auto-generated API docs (Swagger)
- Pydantic validation → Type safety

### 9.3 Database (SQLite)

```sql
-- Bảng Documents
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    file_path TEXT,
    status TEXT,  -- 'uploaded', 'processing', 'completed'
    ocr_result TEXT,  -- JSON data
    created_at TIMESTAMP
);

-- Bảng Users
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT,
    hashed_password TEXT
);
```

**Tại sao dùng SQLite?**

- Không cần cài đặt server riêng
- File-based → Dễ backup, migrate
- Phù hợp cho MVP/prototype

---

## 10. Giá Trị và Hiệu Quả

### 10.1 So Sánh: Manual vs Automated

| Tiêu chí               | Xử lý thủ công          | Hệ thống tự động      |
| ---------------------- | ----------------------- | --------------------- |
| **Thời gian/tài liệu** | 5-10 phút               | 5-10 giây             |
| **Độ chính xác**       | 95% (human error)       | 95-99% (AI)           |
| **Khả năng mở rộng**   | Tuyến tính (thêm người) | Không giới hạn        |
| **Chi phí**            | $15-20/giờ nhân công    | $0.001/tài liệu (API) |
| **Hoạt động 24/7**     | Không                   | Có                    |

### 10.2 ROI (Return on Investment)

**Ví dụ**: Công ty xử lý 1000 hóa đơn/ngày

**Trước**:

- 5 nhân viên × 8 giờ × $15/giờ = $600/ngày
- Chi phí/tháng: $18,000

**Sau**:

- Gemini API: 1000 × $0.01 = $10/ngày
- Server hosting: $100/tháng
- Chi phí/tháng: $400

**Tiết kiệm**: $17,600/tháng = **97.8%**

### 10.3 Use Cases Thực Tế

| Ngành         | Use Case                | Lợi ích                         |
| ------------- | ----------------------- | ------------------------------- |
| **Tài chính** | Xử lý hóa đơn, chứng từ | Giảm 95% thời gian nhập liệu    |
| **Y tế**      | Số hóa hồ sơ bệnh án    | Truy xuất nhanh, không thất lạc |
| **Pháp lý**   | Phân tích hợp đồng      | Tìm điều khoản trong giây       |
| **Logistics** | Xử lý vận đơn           | Tracking real-time              |
| **HR**        | Quản lý CV              | Sàng lọc tự động                |

### 10.4 Điểm Mạnh của Hệ Thống

1. **Hybrid Approach**: Kết hợp traditional (Tesseract) + AI (Gemini) → Tối ưu chi phí/chất lượng

2. **Multi-Stage Pipeline**: Mỗi stage giải quyết 1 vấn đề cụ thể → Dễ debug, maintain

3. **Reading Order Detection**: Hiểu cấu trúc tài liệu phức tạp (multi-column)

4. **Relationship Detection**: Không chỉ "thấy" mà còn "hiểu" mối quan hệ

5. **Schema-Based Extraction**: Linh hoạt cho nhiều loại tài liệu

6. **Visual Highlighting**: Có thể trace lại dữ liệu → Tăng trust

---

## 📝 Tóm Tắt

Hệ thống OCR Document Processing Platform kết hợp nhiều kỹ thuật tiên tiến:

1. **OCR**: Tesseract + Gemini Hybrid
2. **Layout Analysis**: Pre-trained Foundation Models
3. **Reading Order**: Column Detection Algorithm
4. **Relationships**: Hierarchical Parent-Child Detection
5. **Table Detection**: Grid + Alignment + Data Pattern
6. **Extraction**: LLM với TOON format

**Kết quả**: Tự động hóa 95%+ công việc xử lý tài liệu, tiết kiệm thời gian và chi phí đáng kể.

---

_Document version: 1.0_  
_Last updated: 2024-12-09_
