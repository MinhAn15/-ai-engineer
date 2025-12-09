# OCR Document Processing Platform - MVP Overview

## 📋 Mục Lục

1. [Tổng Quan Dự Án](#tổng-quan-dự-án)
2. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
3. [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
4. [Luồng Dữ Liệu](#luồng-dữ-liệu)
5. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
6. [Các Module Chính](#các-module-chính)
7. [Kỹ Thuật Nổi Bật](#kỹ-thuật-nổi-bật)
8. [Hướng Dẫn Chạy](#hướng-dẫn-chạy)

---

## 🎯 Tổng Quan Dự Án

**OCR Document Processing Platform** là một nền tảng xử lý tài liệu thông minh, kết hợp:

- **OCR (Optical Character Recognition)** để nhận dạng văn bản từ ảnh/PDF
- **Layout Analysis** để phân tích cấu trúc tài liệu (bảng, form, tiêu đề...)
- **LLM Extraction** để trích xuất dữ liệu có cấu trúc theo schema người dùng định nghĩa
- **🆕 Pipeline** để hiểu thứ tự đọc và mối quan hệ phân cấp

### Tính Năng Chính

| Tính năng                     | Mô tả                                                  |
| ----------------------------- | ------------------------------------------------------ |
| 📄 **Upload Documents**       | Hỗ trợ PDF và ảnh (JPG, PNG)                           |
| 🔍 **Hybrid OCR**             | Kết hợp Tesseract + Gemini Vision cho độ chính xác cao |
| 📊 **Layout Detection**       | Phát hiện bảng, checkbox, tiêu đề bằng OpenCV + ML     |
| 🔢 **Reading Order**          | 🆕 Phát hiện thứ tự đọc tài liệu (column detection)    |
| 🔗 **Relationship Detection** | 🆕 Caption→Figure, Header→Table relationships          |
| 📋 **Data Pattern Tables**    | 🆕 Detect tables không có grid lines (alignment-based) |
| ✨ **Smart Extraction**       | Trích xuất dữ liệu theo schema (TOON format)           |
| 🎯 **Visual Highlighting**    | Highlight text trên ảnh gốc với màu theo loại          |

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        🖥️ FRONTEND                               │
│                     (Next.js + TypeScript)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
│  │   Login    │  │  Dashboard │  │  Document  │                  │
│  │   Page     │  │   (List)   │  │   Detail   │                  │
│  └────────────┘  └────────────┘  └────────────┘                  │
│         │               │               │                         │
│         └───────────────┴───────────────┘                         │
│                         │ HTTP/REST                               │
└─────────────────────────┼─────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                        🔧 BACKEND API                             │
│                    (FastAPI + Python)                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    api/routes/ocr.py                      │    │
│  │  /upload  /process  /extract  /segments  /documents      │    │
│  └──────────────────────────────────────────────────────────┘    │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              src/ (Processing Pipeline)                   │    │
│  │                                                           │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │    │
│  │  │pdf_processor│  │ hybrid_ocr  │  │llm_extractor│       │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │    │
│  │                                                           │    │
│  │  ┌─────────────────────────────────────────────────────┐ │    │
│  │  │       🆕 Layout Pipeline            │ │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐  │ │    │
│  │  │  │ reading_   │ │ layout_    │ │ relationship_  │  │ │    │
│  │  │  │ order_     │ │ foundation_│ │ detector       │  │ │    │
│  │  │  │ detector   │ │ model      │ │                │  │ │    │
│  │  │  └────────────┘ └────────────┘ └────────────────┘  │ │    │
│  │  └─────────────────────────────────────────────────────┘ │    │
│  │                                                           │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │    │
│  │  │table_detect │  │layout_analy │  │table_parser │       │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ SQLite   │    │ Tesseract│    │ Gemini   │    │Table     │
    │ Database │    │  OCR     │    │ API      │    │Transform │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## 🛠️ Công Nghệ Sử Dụng

### Backend (Python)

| Công nghệ                | Phiên bản | Mục đích                          |
| ------------------------ | --------- | --------------------------------- |
| **FastAPI**              | 0.104+    | REST API framework                |
| **SQLAlchemy**           | 2.0+      | ORM cho database                  |
| **SQLite**               | -         | Database lưu trữ                  |
| **Tesseract**            | 5.0+      | OCR engine chính                  |
| **Pillow**               | 10.0+     | Xử lý ảnh                         |
| **OpenCV**               | 4.8+      | Computer vision (table detection) |
| **pdf2image**            | -         | Chuyển PDF → Image                |
| **Google Generative AI** | -         | Gemini Vision API                 |
| **PyJWT**                | -         | Authentication                    |

### Frontend (TypeScript)

| Công nghệ        | Phiên bản | Mục đích                       |
| ---------------- | --------- | ------------------------------ |
| **Next.js**      | 14+       | React framework với App Router |
| **React**        | 18+       | UI library                     |
| **TypeScript**   | 5+        | Type safety                    |
| **TailwindCSS**  | 3+        | Styling                        |
| **Axios**        | -         | HTTP client                    |
| **Lucide React** | -         | Icon library                   |

### AI/ML Services

| Công nghệ                   | Mục đích                                 |
| --------------------------- | ---------------------------------------- |
| **Gemini 2.0 Flash**        | Hybrid OCR cho low-confidence text       |
| **Gemini Vision**           | Layout classification (fallback)         |
| **Table Transformer** ✅    | 🆕 DocLayNet-style table detection       |
| **Transformers** ✅         | 🆕 Hugging Face model loading            |
| **LayoutLMv3** _(optional)_ | Document layout analysis                 |
| **timm** ✅                 | 🆕 PyTorch Image Models for vision tasks |

---

## 🔄 Luồng Dữ Liệu

### 1. Upload & OCR Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────▶│ Upload  │────▶│  Save   │────▶│ Return  │
│ uploads │     │  API    │     │  File   │     │ Doc ID  │
│  file   │     │         │     │ + DB    │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  /process   │
                              │  endpoint   │
                              └──────┬──────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Image Preproc  │        │   Tesseract     │        │  Hybrid OCR?    │
│  (deskew,       │───────▶│   OCR           │───────▶│  (if enabled)   │
│   contrast)     │        │                 │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
                                                              │
                           Low confidence                     │
                           blocks < threshold ◀───────────────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │  Gemini Vision  │
                           │  re-OCR         │
                           └─────────────────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │  Merge results  │
                           │  → JSON output  │
                           └─────────────────┘
```

### 2. Layout Classification Flow

```
┌─────────────────┐
│  /segments      │
│  ?classify=true │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│            🆕 Layout Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Reading Order Detection [reading_order_detector.py]   │
│  - Column detection (X overlap clustering)                       │
│  - Top→bottom, left→right ordering                               │
│  - Output: reading_order, column fields                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Layout Foundation Model [layout_foundation_model.py]   │
│  - Microsoft Table Transformer (pre-trained)                     │
│  - Fallback: Gemini Vision → Heuristics                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Relationship Detection [relationship_detector.py]     │
│  - Caption → Figure (text below/above image)                     │
│  - Header → Table (title above table)                            │
│  - Table Cell → Table (text inside table)                        │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Multi-Method Table Detection [table_detector.py]      │
│  - Grid lines (OpenCV morphology)                                │
│  - Alignment-based (≥2 columns)                                  │
│  - Data pattern (numeric columns)                                │
│  - Auto-deduplication of overlapping tables                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
├─────────────────────────────────────────────────────────────────┤
│  Segments with:                                                  │
│  - reading_order: 1, 2, 3...                                     │
│  - column: 1, 2...                                               │
│  - type: text, title, table, figure...                           │
│  - relationship: caption, table_header, table_cell...            │
│  - parent/children references                                    │
│  - color-coded for visualization                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Data Extraction Flow (TOON Format)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  OCR Text +     │────▶│  Build Prompt   │────▶│  Gemini LLM     │
│  Schema         │     │  with segments  │     │  (2.0 Flash)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  TOON Output    │
                                               │  (structured)   │
                                               └─────────────────┘
                                                        │
     Example TOON:                                      │
     ┌──────────────────────────────────────────────────┘
     │  @ENTITIES
     │  company_name: ABC Corporation [ref:1]
     │  date: 2024-01-15 [ref:3]
     │  total_amount: $1,500.00 [ref:7]
     │
     │  @RELATIONS
     │  invoice_for: company_name -> project_name
     └──────────────────────────────────────────────────
```

---

## 📁 Cấu Trúc Thư Mục

```
OCR/
├── api/                          # Backend API
│   ├── main.py                   # FastAPI app entry point
│   ├── auth.py                   # JWT authentication
│   ├── database.py               # SQLite connection
│   ├── models.py                 # SQLAlchemy models
│   ├── schemas.py                # Pydantic schemas
│   ├── dependencies.py           # Auth dependencies
│   └── routes/
│       ├── ocr.py                # Main OCR endpoints
│       └── auth.py               # Auth endpoints
│
├── src/                          # Processing modules
│   ├── pdf_processor.py          # PDF → Image + OCR segments
│   ├── hybrid_ocr.py             # Tesseract + Gemini hybrid
│   ├── llm_extractor.py          # Schema-based extraction
│   ├── layout_analysis.py        # Multi-stage layout classification
│   ├── layout_foundation_model.py # 🆕 DocLayNet-style pre-trained model
│   ├── reading_order_detector.py  # 🆕 Reading order detection
│   ├── relationship_detector.py   # 🆕 Hierarchical relationship detection
│   ├── table_detector.py         # OpenCV + alignment + data pattern table detection
│   ├── table_parser.py           # Table structure recognition
│   ├── image_preprocessing.py    # Deskew, contrast enhancement
│   └── llm_usage_tracker.py      # Token usage tracking
│
├── frontend/                     # Next.js frontend
│   ├── src/
│   │   ├── app/                  # App Router pages
│   │   │   ├── page.tsx          # Dashboard
│   │   │   ├── login/page.tsx    # Login page
│   │   │   └── documents/[id]/   # Document detail
│   │   ├── components/           # Reusable components
│   │   │   ├── Sidebar.tsx
│   │   │   └── NeurondLogo.tsx
│   │   ├── context/
│   │   │   └── AuthContext.tsx   # Auth state management
│   │   └── lib/
│   │       └── api.ts            # Axios API client
│   └── package.json
│
├── database/                     # SQLite database files
├── uploads/                      # Uploaded documents
├── outputs/                      # Processed outputs
├── config.py                     # App configuration
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables
```

---

## 🔧 Các Module Chính

### 1. `src/pdf_processor.py`

Xử lý PDF và trích xuất segments từ ảnh.

```python
# Key functions:
pdf_to_images(pdf_path)           # Convert PDF → list of PIL Images
ocr_image_to_segments(image)      # OCR → paragraph-level segments
```

### 2. `src/hybrid_ocr.py`

OCR hybrid kết hợp Tesseract + Gemini Vision.

```python
# Logic:
1. Chạy Tesseract OCR → get text blocks with confidence
2. Filter blocks có confidence < threshold (mặc định 75%)
3. Crop vùng low-confidence → gửi Gemini Vision
4. Merge kết quả → output cuối cùng
```

### 3. `src/layout_analysis.py`

Phân loại layout documents với multi-stage pipeline.

```python
# Types detected:
- text      (🟢 Green)    - Regular paragraphs
- title     (🔴 Red)      - Headers, section names
- table     (🟠 Orange)   - Table content
- figure    (🔵 Blue)     - Images, diagrams
- checkbox  (🟢 Lime)     - Form checkboxes
- form      (🩵 Cyan)     - Form fields
- list      (🟡 Yellow)   - List items
```

### 4. `src/table_detector.py`

Detect tables bằng OpenCV + alignment + data patterns.

```python
# Methods:
detect_tables(image)                      # Grid line detection (OpenCV)
detect_tables_by_alignment(blocks)        # Text alignment patterns (≥2 columns)
detect_tables_by_data_pattern(blocks)     # 🆕 Data pattern (numeric columns)
detect_all_tables(image, blocks)          # 🆕 Combined detection with deduplication
detect_checkboxes(image)                  # Small square contours
```

### 5. `src/llm_extractor.py`

Trích xuất dữ liệu có cấu trúc bằng Gemini LLM.

```python
# Output format: TOON (Typed Object Oriented Notation)
@ENTITIES
field_name: value [ref:segment_id]

@RELATIONS
relation_name: entity1 -> entity2
```

### 6. `src/reading_order_detector.py` 🆕

Phát hiện thứ tự đọc tài liệu.

```python
# Methods:
detect_reading_order(segments)    # Assign reading_order, column fields
detect_lines(segments)            # Group segments into lines
get_context_for_segment(segment)  # Get prev/next/above/below context

# Output fields added:
- reading_order: int (1, 2, 3...)
- column: int (1, 2...)
```

### 7. `src/layout_foundation_model.py` 🆕

Pre-trained layout model (DocLayNet-style).

```python
# Models supported:
- microsoft/table-transformer-detection
- facebook/detr-resnet-50

# Methods:
analyze_layout(image)    # Returns layout chunks with type, bbox, confidence
detect_tables(image)     # Table-only detection
visualize(image, chunks) # Draw detection boxes
```

### 8. `src/relationship_detector.py` 🆕

Phát hiện mối quan hệ phân cấp giữa các layout elements.

```python
# Relationships detected:
- caption → figure     # Text below/above image
- table_header → table # Title above table
- table_cell → table   # Text inside table bbox
- form_label → value   # "Invoice No:" → "12345"

# Methods:
detect_relationships(chunks)  # Add parent, children, relationship fields
```

---

## 🎨 Kỹ Thuật Nổi Bật

### 1. Hybrid OCR (Tesseract + Gemini Vision)

#### Vấn đề cần giải quyết

Tesseract OCR hoạt động tốt với văn bản in rõ ràng, nhưng gặp khó khăn với:

- Chữ viết tay
- Font đặc biệt hoặc cách điệu
- Ảnh có noise, mờ, nghiêng
- Text trên nền phức tạp

#### Giải pháp: Hybrid Approach

Thay vì gửi **toàn bộ** ảnh cho AI (tốn token), chỉ gửi những **vùng khó đọc**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HYBRID OCR PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │ Input Image │                                                   │
│   └──────┬──────┘                                                   │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────┐               │
│   │  STEP 1: Tesseract OCR                          │               │
│   │  • Chạy pytesseract.image_to_data()             │               │
│   │  • Output: text blocks + confidence score       │               │
│   └──────┬──────────────────────────────────────────┘               │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────┐               │
│   │  STEP 2: Filter by Confidence Threshold         │               │
│   │                                                 │               │
│   │  Threshold = 75% (user configurable)            │               │
│   │                                                 │               │
│   │  ┌─────────────┐    ┌─────────────┐             │               │
│   │  │ conf ≥ 75%  │    │ conf < 75%  │             │               │
│   │  │ ✓ Keep      │    │ ✗ Need AI   │             │               │
│   │  └──────┬──────┘    └──────┬──────┘             │               │
│   │         │                  │                     │               │
│   └─────────┼──────────────────┼─────────────────────┘               │
│             │                  │                                     │
│             │                  ▼                                     │
│             │   ┌─────────────────────────────────────┐              │
│             │   │  STEP 3: Crop & Send to Gemini     │              │
│             │   │                                     │              │
│             │   │  For each low-conf region:          │              │
│             │   │  1. Crop image region (bbox)        │              │
│             │   │  2. Send to Gemini Vision           │              │
│             │   │  3. Get corrected text              │              │
│             │   └──────────────┬──────────────────────┘              │
│             │                  │                                     │
│             ▼                  ▼                                     │
│   ┌─────────────────────────────────────────────────┐               │
│   │  STEP 4: Merge Results                          │               │
│   │                                                 │               │
│   │  Original blocks + AI-corrected blocks          │               │
│   │  → Final OCR output                             │               │
│   └─────────────────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Ví dụ thực tế

```
Input: Ảnh hóa đơn có chữ mờ

Tesseract output:
┌──────────────────────────────────────────────────────────┐
│ Block 1: "ABC Corporation"     conf: 95%  → KEEP        │
│ Block 2: "Invoice #12345"      conf: 92%  → KEEP        │
│ Block 3: "Đ1a ch1 giao hàng"   conf: 45%  → SEND TO AI  │
│ Block 4: "Total: $1,500.00"    conf: 88%  → KEEP        │
└──────────────────────────────────────────────────────────┘

Gemini Vision output for Block 3:
"Địa chỉ giao hàng"  (corrected!)

Final merged output:
"ABC Corporation | Invoice #12345 | Địa chỉ giao hàng | Total: $1,500.00"
```

#### Lợi ích

| Aspect           | Tesseract Only | Gemini Only | Hybrid     |
| ---------------- | -------------- | ----------- | ---------- |
| **Tốc độ**       | ⚡ Fast        | 🐢 Slow     | ⚡ Fast    |
| **Chi phí**      | 💚 Free        | 💰 $$       | 💛 Low     |
| **Độ chính xác** | 😐 Variable    | 😊 High     | 😊 High    |
| **Offline**      | ✅ Yes         | ❌ No       | ⚠️ Partial |

---

### 2. Multi-Stage Layout Detection

#### Vấn đề cần giải quyết

Document có nhiều loại nội dung khác nhau (text, table, form, figure...). Cần phân loại để:

- Highlight đúng màu trên UI
- Xử lý khác nhau (table → parse structure, text → extract entities)
- Cải thiện độ chính xác extraction

#### Giải pháp: Pipeline từ đơn giản → phức tạp

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE DETECTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: Visual Table Detection (OpenCV)         Cost: FREE  │   │
│  │                                                              │   │
│  │ Kỹ thuật:                                                    │   │
│  │ 1. Threshold ảnh → binary image                              │   │
│  │ 2. Detect horizontal lines: morphologyEx(MORPH_OPEN, [40,1]) │   │
│  │ 3. Detect vertical lines: morphologyEx(MORPH_OPEN, [1,40])   │   │
│  │ 4. Combine lines → table mask                                │   │
│  │ 5. Find contours → filter by area & aspect ratio             │   │
│  │                                                              │   │
│  │ Kết quả: Detect tables có đường kẻ rõ ràng                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: Alignment-Based Detection              Cost: FREE   │   │
│  │                                                              │   │
│  │ Kỹ thuật:                                                    │   │
│  │ 1. Group OCR blocks by Y coordinate (same row)               │   │
│  │ 2. For each row: count columns, check spacing                │   │
│  │ 3. If ≥3 columns + evenly spaced → likely table              │   │
│  │ 4. Merge consecutive table rows → table region               │   │
│  │                                                              │   │
│  │ Kết quả: Detect tables KHÔNG có đường kẻ (data alignment)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: Checkbox Detection (OpenCV)            Cost: FREE   │   │
│  │                                                              │   │
│  │ Kỹ thuật:                                                    │   │
│  │ 1. Find contours trong ảnh                                   │   │
│  │ 2. Filter: 10px < size < 50px, aspect ratio ≈ 1              │   │
│  │ 3. Mark adjacent text as "checkbox" type                     │   │
│  │                                                              │   │
│  │ Kết quả: Detect checkboxes, radio buttons trong forms        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: ML/LLM Classification                Cost: API      │   │
│  │                                                              │   │
│  │ Chỉ chạy cho segments CHƯA được phân loại ở stages trước     │   │
│  │                                                              │   │
│  │ Options:                                                     │   │
│  │ A. Gemini Vision: Send image + segment list → classify       │   │
│  │ B. LayoutLMv3: Transformer model for document layout         │   │
│  │ C. Heuristics: Pattern matching (uppercase → title, etc.)    │   │
│  │                                                              │   │
│  │ Kết quả: Classify text/title/figure/form/list/signature...   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Tại sao Multi-Stage?

| Approach        | Processing   | Cost    | Accuracy |
| --------------- | ------------ | ------- | -------- |
| LLM only        | All segments | 💰💰💰  | High     |
| OpenCV only     | All segments | 💚 Free | Medium   |
| **Multi-Stage** | Progressive  | 💛 Low  | High     |

**Logic:** Xử lý những thứ dễ bằng công cụ miễn phí trước, chỉ dùng AI cho những gì còn lại.

---

### 3. Segment-Based Extraction với Reference Linking

#### Vấn đề cần giải quyết

Khi LLM trích xuất dữ liệu, làm sao biết dữ liệu đó nằm ở đâu trong document?

#### Giải pháp: Segment References

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SEGMENT-BASED EXTRACTION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STEP 1: OCR Output với Segment IDs                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ [1] ABC Corporation                                         │    │
│  │ [2] Invoice #12345                                          │    │
│  │ [3] Date: 2024-01-15                                        │    │
│  │ [4] Ship to: 123 Main St, City                              │    │
│  │ [5] Item: Widget Pro                                        │    │
│  │ [6] Quantity: 10                                            │    │
│  │ [7] Total: $1,500.00                                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            ▼                                        │
│  STEP 2: LLM Extraction với Reference                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Prompt:                                                     │    │
│  │ "Extract fields. Include [ref:N] for each field."           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            ▼                                        │
│  STEP 3: TOON Output                                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ @ENTITIES                                                   │    │
│  │ company_name: ABC Corporation [ref:1]                       │    │
│  │ invoice_number: 12345 [ref:2]                               │    │
│  │ date: 2024-01-15 [ref:3]                                    │    │
│  │ shipping_address: 123 Main St, City [ref:4]                 │    │
│  │ item: Widget Pro [ref:5]                                    │    │
│  │ quantity: 10 [ref:6]                                        │    │
│  │ total_amount: $1,500.00 [ref:7]                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            ▼                                        │
│  STEP 4: Frontend Highlighting                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Hover "total_amount" → highlight segment [7] on image       │    │
│  │                                                             │    │
│  │  ┌──────────────────┐     ┌──────────────────────┐          │    │
│  │  │  Extracted Data  │     │   Document Image     │          │    │
│  │  │                  │     │                      │          │    │
│  │  │  total_amount:   │ ──▶ │  ┌────────────────┐  │          │    │
│  │  │  $1,500.00 [7]   │     │  │ Total: $1,500  │  │          │    │
│  │  │  ▲ (hover)       │     │  └────────────────┘  │          │    │
│  │  └──────────────────┘     └──────────────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Lợi ích

- **Traceability:** Biết chính xác dữ liệu từ đâu
- **Verification:** User có thể verify bằng cách hover
- **Debugging:** Dễ phát hiện lỗi extraction

---

### 4. Visual Feedback System

#### Color-Coded Layout Types

Mỗi loại layout có màu riêng để dễ phân biệt:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LAYOUT COLOR CODING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   TYPE           COLOR       HEX        USE CASE                    │
│   ─────────────────────────────────────────────────────────────     │
│   text           🟢 Green    #22c55e    Regular paragraphs          │
│   title          🔴 Red      #ef4444    Headers, section names      │
│   table          🟠 Orange   #f97316    Table content               │
│   figure         🔵 Blue     #3b82f6    Images, diagrams            │
│   list           🟡 Yellow   #eab308    Bullet points, numbered     │
│   checkbox       🟢 Lime     #84cc16    Form checkboxes             │
│   form           🩵 Cyan     #06b6d4    Input fields                │
│   attestation    🟣 Purple   #a855f7    Signatures, stamps          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Interactive Highlighting

```
User Action                      System Response
───────────────────────────────────────────────────────
Hover segment in list     →      Highlight bbox on image
Hover extracted field     →      Highlight source segment
Click segment             →      Show details panel
Mouse leave               →      Remove highlight
```

#### Confidence Indicators

```
Confidence     Display                  Meaning
─────────────────────────────────────────────────────
≥ 90%          🟢 Green text           Very reliable
75-89%         🟡 Yellow text          Moderate confidence
< 75%          🟠 Orange text          May need review
```

---

### 5. TOON Format (Typed Object-Oriented Notation)

#### Vấn đề với JSON output

- Dài dòng, nhiều dấu ngoặc
- Khó đọc cho human
- LLM hay generate invalid JSON

#### TOON: Format đơn giản hơn

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TOON FORMAT                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   @ENTITIES                     ← Section header                    │
│   field_name: value [ref:N]     ← Entity with reference             │
│   field_name: value             ← Entity without reference          │
│   field_name: null              ← Not found                         │
│                                                                     │
│   @RELATIONS                    ← Section header                    │
│   relation: entity1 -> entity2  ← Relationship                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│   EXAMPLE:                                                          │
│                                                                     │
│   @ENTITIES                                                         │
│   company_name: ABC Corporation [ref:1]                             │
│   invoice_number: INV-2024-001 [ref:2]                              │
│   date: 2024-01-15 [ref:3]                                          │
│   total_amount: 1500.00 [ref:7]                                     │
│   currency: USD                                                     │
│   tax_id: null                                                      │
│                                                                     │
│   @RELATIONS                                                        │
│   issued_by: company_name -> invoice_number                         │
│   dated: invoice_number -> date                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### So sánh với JSON

| Aspect             | JSON           | TOON                |
| ------------------ | -------------- | ------------------- |
| **Readability**    | 😐 Medium      | 😊 High             |
| **LLM Generation** | ❌ Error-prone | ✅ Reliable         |
| **Parsing**        | ✅ Standard    | ⚠️ Custom parser    |
| **References**     | 😐 Nested      | ✅ Inline `[ref:N]` |

---

## 🚀 Hướng Dẫn Chạy

### Prerequisites

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **Tesseract OCR** - [Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Quick Start (Windows)

**Terminal 1 - Backend:**

```powershell
cd "c:\Users\an.ly\OneDrive - Orient\2026\ai-engineer\OCR"
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**

```powershell
cd "c:\Users\an.ly\OneDrive - Orient\2026\ai-engineer\OCR\frontend"
npm install
npm run dev
```

### URLs

| Service     | URL                        |
| ----------- | -------------------------- |
| Frontend    | http://localhost:3000      |
| Backend API | http://localhost:8000      |
| API Docs    | http://localhost:8000/docs |

### Environment Variables

Tạo file `.env` trong thư mục `OCR/`:

```bash
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=your_jwt_secret_key
```

### Default Login

| Field    | Value      |
| -------- | ---------- |
| Username | `admin`    |
| Password | `admin123` |

### 🏗️ Kiến Trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 Layout Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Reading Order Detection [reading_order_detector.py]   │
│  - Column detection (X overlap clustering)                       │
│  - Top→bottom, left→right ordering                               │
│  - Assigns: reading_order, column fields                         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Layout Foundation Model [layout_foundation_model.py]  │
│  - Microsoft Table Transformer (pre-trained)                     │
│  - Detects: tables, figures, titles, lists                       │
│  - Fallback: Gemini Vision → Heuristics                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Relationship Detection [relationship_detector.py]     │
│  - Caption → Figure (text below/above image)                     │
│  - Header → Table (title above table)                            │
│  - Table Cell (text inside table bbox)                           │
│  - Form Label → Value pattern                                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Data Pattern Table Detection [table_detector.py]      │
│  - detect_tables_by_data_pattern() - no gridlines needed         │
│  - detect_all_tables() - combined detection                      │
│  - Numeric column detection (VND, $, €, %)                       │
│  - Auto-deduplication of overlapping tables                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 API Endpoints

| Method | Endpoint                       | Mô tả               |
| ------ | ------------------------------ | ------------------- |
| POST   | `/auth/login`                  | Login → JWT token   |
| POST   | `/ocr/upload`                  | Upload document     |
| POST   | `/ocr/process/{id}`            | Run OCR             |
| POST   | `/ocr/extract/{id}`            | Run LLM extraction  |
| GET    | `/ocr/documents`               | List documents      |
| GET    | `/ocr/documents/{id}`          | Get document detail |
| GET    | `/ocr/documents/{id}/segments` | Get OCR segments    |
| GET    | `/ocr/documents/{id}/image`    | Get document image  |

---

_Last updated: 2024-12-09_
