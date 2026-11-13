# Day 06 · Data Ingestion Engineering

**Học viên:** Dương Mạnh Hùng — **MSSV:** 2A202603431 — **Lớp:** L3  
**Báo cáo chi tiết:** [`report/individual/2A202603431_DuongManhHung.md`](report/individual/2A202603431_DuongManhHung.md)  
**Nhật ký thí nghiệm:** [`artifacts/experiment_log.csv`](artifacts/experiment_log.csv)

---

## Hướng dẫn cài đặt & tái lập kết quả

Giải pháp dùng **hoàn toàn local** (không cần Docker / PostgreSQL):

- **Vector Store:** `qdrant-client` local PersistentClient — lưu tại `starter/qdrant_db/`, metric `cosine`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` — 384 chiều, chạy CPU
- **Corpus:** *Introduction to Sociology 3e* — OpenStax, CC BY 4.0 — 662 trang nội dung

### 1. Cài đặt (Python 3.11+)

```powershell
cd starter
pip install -r requirements.txt
```

### 2. Chạy pipeline và benchmark

```powershell
python -m src.run
```

Lần đầu: tải model ~90MB, embed 3 chiến lược (~4–5 phút CPU).  
Lần sau: data có sẵn trong `qdrant_db/` → chỉ chạy benchmark, rất nhanh.

### 3. Kết quả đầu ra

| File | Nội dung |
|---|---|
| `starter/results/traces.json` | Trace top-3 cho từng câu hỏi, cả 3 chiến lược |
| `starter/results/summary.json` | Hit@1 và EvidHit@3 theo dev/test |
| `starter/results/chunk_stats.json` | Số chunk và avg char mỗi chiến lược |

---

## Kết quả Benchmark (21 câu: 15 dev / 6 test)

| Chiến lược | Chunking | Filter | Dev Hit@1 | Dev EvidHit@3 | Test Hit@1 | Test EvidHit@3 | Chunk |
|---|---|---|---|---|---|---|---|
| **A** | `fixed_size(500, overlap=50)` | — | **46.7%** (7/15) | 80.0% (12/15) | 33.3% (2/6) | **83.3%** (5/6) | 4481 |
| **B** | `by_paragraph(max=800)` | — | 26.7% (4/15) | **93.3%** (14/15) | 33.3% (2/6) | 66.7% (4/6) | 2726 |
| **C** | `by_paragraph(max=800)` | `document_type=textbook` | 26.7% (4/15) | **93.3%** (14/15) | 33.3% (2/6) | 66.7% (4/6) | 2726 |

**Chiến lược được chọn: B** — EvidHit@3 dev cao nhất (93.3%), ít chunk hơn A (inference nhanh hơn), bảo toàn ngữ cảnh đoạn văn tốt nhất.

---

## Cấu trúc repo

```
starter/
├── src/                    # Pipeline code (ingestion, chunking, embeddings, store, evaluation, run)
├── data/
│   ├── manifest.json       # Nguồn, giấy phép, số trang
│   ├── benchmark.json      # 21 câu hỏi với gold evidence (tách trước khi chạy)
│   └── openstax-*.pdf      # Corpus (CC BY 4.0)
├── results/                # traces.json, summary.json, chunk_stats.json
├── qdrant_db/              # Vector store local (gitignore)
├── requirements.txt        # pypdf, python-dotenv, sentence-transformers, qdrant-client
└── .env                    # EMBEDDING_MODEL, QDRANT_DIR (không commit)
artifacts/
└── experiment_log.csv      # Cấu hình A/B/C và kết quả
benchmark/
└── benchmark.json          # Bản sao benchmark questions
report/individual/
└── 2A202603431_DuongManhHung.md  # Báo cáo cá nhân đầy đủ
```

---

## Tài liệu hướng dẫn gốc

| Cần tìm | Tài liệu |
|---|---|
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Năm tiêu chí benchmark | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |
