# Báo cáo cá nhân — Day 6

**Họ và tên:** Dương Mạnh Hùng  
**MSSV:** 2A202603431  
**Lớp:** L3  
**Ngày chạy:** 2026-11-13  
**Thời gian embedding (3 strategies):** ~271s (~4.5 phút) trên CPU  
**Chi phí:** $0 (local model, không dùng API)

---

## Dữ liệu và pipeline

**Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:**
- Corpus: *Introduction to Sociology 3e* — OpenStax, giấy phép CC BY 4.0
- URL nguồn: https://assets.openstax.org/oscms-prodcms/media/documents/introduction-sociology-3e_-_WEB.pdf
- 664 trang vật lý; 662 trang có text trích xuất được; loại trang bìa và trang trắng → **662 trang nội dung hợp lệ** (vượt yêu cầu ≥30)
- Quy tắc: đếm trang có `extracted_text != ""` sau khi normalize Unicode

**Chuẩn hóa, lỗi và giới hạn:**
- Normalize NFKC, loại soft-hyphen `\u00ad`, collapse whitespace, giữ ranh giới paragraph (`\n\n`)
- Không có trang bị lỗi OCR (PDF đã có text layer sẵn)
- Giới hạn: pypdf đôi khi trích xuất header/footer dính vào body text

**Embedding model / vector store / metric / lệnh chạy / chi phí:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, L2-normalized → cosine
- Vector store: `qdrant-client==1.9.x` local PersistentClient (thư mục `starter/qdrant_db/`)
- Metric: `cosine` (cố định)
- Lệnh chạy: `cd starter && python -m src.run`
- Chi phí: $0 (tất cả local)

**Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:**
- Bắt buộc: `document_id`, `source`, `page` (truy vết chunk về trang PDF gốc)
- Strategy C thêm: `chapter` (ước lượng theo range trang), `document_type="textbook"`, `language="en"`
- `page` giúp truy vết tới trang PDF; `chapter` cho phép filter theo chương khi câu hỏi có ngữ cảnh chương cụ thể
- Hạn chế: `chapter` chỉ là phân loại thủ công gần đúng theo range trang, không parse TOC thực sự

---

## Cấu hình và benchmark

**A / B / C (tham số cụ thể, khác biệt duy nhất B–C):**

| Cấu hình | Chunking | Tham số | Metadata filter |
|---|---|---|---|
| A | `fixed_size` | size=500, overlap=50 | Không |
| B | `by_paragraph` | max_size=800 | Không |
| C | `by_paragraph` | max_size=800 | `document_type=textbook` |

Khác biệt B→C: thêm 3 trường metadata (`chapter`, `document_type`, `language`) và bật filter cho câu hỏi có trường `filter`.

**≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence:**
- Tổng: 21 câu (15 dev + 6 test)
- Loại câu: direct (11), paraphrase (3), filter (4) — ≥3 câu filter, trong đó 2 câu thuộc test
- Câu filter: q12, q15 (dev); q19, q21 (test) — tất cả dùng `document_type=textbook`
- Kiểm chứng: so sánh `evidence_quote` với text thực trong PDF tại `gold_page` trước khi ghi
- Split được chốt trước khi chạy thử nghiệm

**Tiêu chí chọn chiến lược đã chốt trước khi xem test:**
Ưu tiên theo thứ tự: (1) EvidHit@3 dev cao nhất → (2) Hit@1 cao nhất → (3) số chunk ít hơn (inference nhanh hơn)

---

## Kết quả

### Retrieval metrics

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian embed | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 7/15 (46.7%) | 12/15 (80.0%) | 4481 | ~99s | $0 |
| A | test | 2/6 (33.3%) | 5/6 (83.3%) | 4481 | — | $0 |
| A | overall | 9/21 (42.9%) | 17/21 (81.0%) | 4481 | — | $0 |
| B | dev | 4/15 (26.7%) | 14/15 (93.3%) | 2726 | ~97s | $0 |
| B | test | 2/6 (33.3%) | 4/6 (66.7%) | 2726 | — | $0 |
| B | overall | 6/21 (28.6%) | 18/21 (85.7%) | 2726 | — | $0 |
| C | dev | 4/15 (26.7%) | 14/15 (93.3%) | 2726 | ~76s | $0 |
| C | test | 2/6 (33.3%) | 4/6 (66.7%) | 2726 | — | $0 |
| C | overall | 6/21 (28.6%) | 18/21 (85.7%) | 2726 | — | $0 |

### Chunk coherence (manual audit — 10 vị trí cố định: page 31, 51, 81, 111, 141, 171, 201, 231, 261, 301)

Thang: 0 = bị cắt giữa ý/thiếu chủ thể; 1 = hiểu một phần; 2 = tự hiểu được và giữ trọn ý

| Vị trí (page) | Strategy A score | Strategy A lý do | Strategy B/C score | Strategy B/C lý do |
|---|---|---|---|---|
| p.31 | 1 | Chunk bắt đầu giữa câu "a hierarchy of employment..." thiếu chủ thể | 2 | Đoạn Criticism đầy đủ, tự hiểu được |
| p.51 | 1 | Cắt giữa ví dụ về independent/dependent variable | 2 | Giữ trọn đoạn ví dụ hypothesis |
| p.81 | 2 | Đủ ngữ cảnh về nhạc và cảm xúc | 2 | Đầy đủ, giữ trọn đoạn |
| p.111 | 1 | Cắt giữa mô tả Agricultural Revolution | 2 | Giữ trọn đoạn mô tả |
| p.141 | 2 | Đủ ngữ cảnh về textbook Nhật–Hàn | 2 | Đầy đủ |
| p.171 | 2 | Đủ về Newman và fast food | 2 | Đầy đủ |
| p.201 | 2 | Key terms đủ định nghĩa | 2 | Đầy đủ |
| p.231 | 1 | Cắt giữa đoạn về cyberfeminism | 2 | Giữ trọn đoạn |
| p.261 | 2 | Đủ về Marx và stratification | 2 | Đầy đủ |
| p.301 | 1 | Cắt giữa câu về miscegenation | 2 | Giữ trọn đoạn |

**Điểm trung bình:** A = 15/20 = **1.5/2** | B/C = 20/20 = **2.0/2**

> Người chấm: tác giả. Nguy cơ thiên lệch: đã xác định vị trí trước khi xem chunk.

### Metadata đủ và đúng (5 câu test × top-3 = 15 bản ghi)

Kiểm tra thủ công: `source`, `page` khớp PDF gốc; trường filter có mặt với giá trị đúng.

- **Bản ghi có metadata đủ và đúng:** 15/15 (100%)
- **Lỗi:** 0
- **Câu có filter, top-3 tuân thủ filter:** 4/4 câu có filter — tuy nhiên filter `document_type=textbook` không discriminative (mọi chunk đều là textbook), nên tỉ lệ 100% không phản ánh độ hữu ích thực sự
- **Trường hữu ích:** `source` giúp truy vết tài liệu gốc; `page` giúp đọc context thêm; `chapter` (Strategy C) cho phép lọc theo chương — hữu ích hơn nếu câu hỏi có ngữ cảnh chương rõ ràng
- **Hạn chế:** `document_type=textbook` không phân biệt được các chunk khác nhau vì tất cả cùng corpus

### Grounding quality (5 câu test, chấm top-3)

Thang: 0 = không có bằng chứng; 1 = có nhưng thiếu điều kiện/ngữ cảnh; 2 = đủ để trả lời và truy nguồn

| Câu | Strategy A | Strategy B | Strategy C | Lý do (A/B/C) |
|---|---|---|---|---|
| q16 (age old?) | 2 | 2 | 2 | Bảng số liệu rõ ràng ở p.381 |
| q17 (prime of life Boomer?) | 2 | 2 | 2 | Bảng số liệu ở p.381 |
| q18 (Korea-Japan?) | 2 | 2 | 2 | Đoạn lịch sử p.141 đầy đủ |
| q19 (miscegenation?) | 0 | 0 | 0 | Top-3 đều không chứa định nghĩa miscegenation tại p.301 |
| q21 (independent variable?) | 2 | 1 | 1 | A: p.51 có ví dụ đầy đủ; B/C rank-3 mới có, thiếu ngữ cảnh ở rank-1 |

**Tổng grounding:** A: 0×1, 1×0, 2×4 → trung bình **1.6/2** | B/C: 0×1, 1×1, 2×3 → trung bình **1.4/2**

---

## Phân tích và kết luận

**So sánh A–B và B–C; 2 ví dụ sai:**

*A vs B:*
- A có Hit@1 cao hơn (46.7% vs 26.7% dev): fixed-size overlap 50 đôi khi đặt keyword quan trọng vào top của chunk và score cao hơn
- B có EvidHit@3 cao hơn (93.3% vs 80.0% dev): paragraph chunking giữ trọn ý đoạn → semantic embedding chính xác hơn → evidence rơi vào top-3 nhiều hơn
- Lỗi A điển hình: **q03** (p.31) — gold evidence nằm cuối đoạn "Criticism", A cắt tại char 500 → evidence bị chia ra 2 chunk, không chunk nào có đủ context → miss. B giữ trọn đoạn "Criticism" → hit.
- Lỗi A điển hình: **q11** (p.201) — "corporate crime" là key term trong danh sách key terms; A cắt danh sách giữa chừng → rank1 lấy p.195 (crime types), không lấy p.201. B chunk theo đoạn key terms → hit.

*B vs C:*
- Kết quả giống hệt nhau (B = C) vì filter `document_type=textbook` không discriminative: tất cả 2726 chunk đều có `document_type=textbook` → filter không thu hẹp không gian tìm kiếm
- Nếu dùng filter `chapter=chapter_X` (cụ thể hơn), C có thể khác B với câu hỏi phạm vi chương

**Chiến lược chọn, đánh đổi, điều kiện áp dụng:**
- **Chọn Strategy B** (theo tiêu chí đã chốt: EvidHit@3 dev cao nhất = 93.3%)
- Đánh đổi: B hy sinh Hit@1 (26.7% vs A 46.7%) để đổi lấy EvidHit@3 cao hơn — phù hợp hơn với RAG thực tế vì cần evidence trong top-3 để LLM có thể trả lời đúng
- Số chunk ít hơn (2726 vs 4481): inference nhanh hơn, storage nhỏ hơn
- A phù hợp hơn khi cần precision@1 cao (ví dụ: trả lời trực tiếp không cần reranker)

**Hạn chế:**
- Tập test nhỏ (6 câu) → sai số lớn, không khái quát hóa được
- `chapter` phân loại thủ công gần đúng → có thể sai vài chương
- Filter B–C không có tác dụng do thiết kế filter không discriminative
- Chấm coherence và grounding do tác giả tự chấm → nguy cơ thiên lệch xác nhận
- Không đo thời gian/chi phí query (chỉ đo embedding)

**Nếu làm lại hoặc mở rộng:**
- Dùng filter `chapter` thay `document_type` để C thực sự khác B
- Thêm sliding window overlap vào by_paragraph (Strategy D)
- Thử model lớn hơn: `all-mpnet-base-v2` (768d) hoặc multilingual `paraphrase-multilingual-MiniLM-L12-v2`
- Hybrid retrieval: kết hợp BM25 (keyword) + dense similarity → cải thiện câu hỏi có keyword đặc thù như "miscegenation", "MDG 5"

---

*Đường dẫn trace:* `starter/results/traces.json` | *Chunk stats:* `starter/results/chunk_stats.json` | *Benchmark:* `starter/data/benchmark.json`

