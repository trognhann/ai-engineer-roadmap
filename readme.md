# AI Engineer Roadmap 🚀

Kho lưu trữ tài liệu, mã nguồn thực hành và các module chuyên sâu trên lộ trình trở thành **AI Engineer (GenAI / LLM Infrastructure & Applications)**.

---

## 📁 Cấu trúc thư mục

```text
ai-engineer-roadmap/
├── notebooks/                     # Thử nghiệm nhanh với Jupyter / Colab
├── tests/                         # Unit tests & integration tests (pytest)
│   ├── __init__.py
│   ├── test_gateway.py            # Test suite cho ModelGateway & Guardrail (Lab 02)
│   ├── test_product_tagger.py     # Test suite cho Semantic Cache & Tagger (Lab 03)
│   └── test_model_router.py       # Test suite cho ModelRouter & Cost Audit (Lab 04)
├── src/
│   ├── config/                    # Cấu hình dự án (Pydantic Settings, .env)
│   │   ├── __init__.py
│   │   └── config.py
│   ├── foundations/               # Module Foundations & Core APIs (Clean Architecture)
│   │   ├── __init__.py            # Facade re-exports toàn bộ các module
│   │   ├── base/                  # [Tầng 1] Base Classes & Schemas dùng chung
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py         # Data Contracts (CustomerSupportTicket, ExecutionMetrics, ...)
│   │   │   └── extractor.py       # Lớp trừu tượng BaseExtractionService (ABC)
│   │   ├── providers/             # [Tầng 2] Tích hợp SDK của các nhà cung cấp LLM
│   │   │   ├── __init__.py
│   │   │   ├── gemini/            # Google Gemini API
│   │   │   ├── openai/            # OpenAI API (GPT-4o, GPT-4o-mini)
│   │   │   └── claude/            # Anthropic Claude API
│   │   ├── infrastructure/        # [Tầng 3] Hạ tầng vận hành & Định tuyến
│   │   │   ├── __init__.py
│   │   │   ├── gateway.py         # [LAB 02] Production AI Gateway (SLA, Guardrail, Fallback)
│   │   │   └── model_router.py    # [LAB 04] Dynamic Model Router & Cost Optimization
│   │   └── services/              # [Tầng 4] Tính năng & Use-case thực tế
│   │       ├── __init__.py
│   │       └── product_tagger.py  # [LAB 03] Smart Feature Service & Semantic Cache
│   ├── rag/                       # Ingestion, Chunking, Embeddings, Vector Stores, Reranking
│   ├── agents/                    # Tools, Memory, Multi-agent Systems (LangGraph, CrewAI)
│   ├── evals/                     # LLM-as-a-judge, Benchmarks, Tracing (LangSmith, Phoenix)
│   └── serving/                   # Serving APIs (FastAPI), Local inference (vLLM, Ollama), Docker
├── .env.example                   # Biến môi trường mẫu
├── .env                           # Biến môi trường local (không commit lên git)
├── requirements.txt               # Danh sách thư viện phụ thuộc trực tiếp
├── main.py                        # Entrypoint chạy demo Lab 01, Lab 02, Lab 03 & Lab 04
└── README.md
```

---

## 🔬 Các bài Lab thực hành

### 📌 Lab 01: Robust Structured Outputs & Data Contracts
* **Vị trí:** [`src/foundations/base/`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base) và [`src/foundations/providers/gemini/`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/providers/gemini)
* **Mục tiêu:** Xây dựng hệ thống trích xuất dữ liệu có cấu trúc từ văn bản thô (ticket hỗ trợ khách hàng) tuân thủ nghiêm ngặt schema định sẵn bằng Pydantic.
* **Đặc điểm:**
  * Kế thừa cấu trúc OOP với [`BaseExtractionService`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base/extractor.py).
  * Chuẩn hóa Schema [`CustomerSupportTicket`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base/schemas.py) tương thích với Native Structured Outputs (Gemini & OpenAI).
  * Cơ chế tự động retry và sửa lỗi khi dữ liệu trả về sai cấu trúc JSON.

---

### 📌 Lab 02: Production AI Gateway (SLA, Token Guardrail & Dynamic Fallback)
* **Vị trí:** [`src/foundations/infrastructure/gateway.py`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/gateway.py)
* **Mục tiêu:** Đóng vai trò là AI Engineer phụ trách hạ tầng và độ ổn định của hệ thống, xây dựng module `ModelGateway` quản lý kết nối LLM đáp ứng tiêu chuẩn sản xuất (Production-ready).

#### 3 Trụ cột cốt lõi:
1. **Token Budget Guardrail ([`TokenBudgetGuardrail`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/gateway.py#L55)):**
   * Ước tính trước số lượng token đầu vào (heuristic cho tiếng Việt/UTF-8).
   * Từ chối xử lý ngay lập tức (`latency < 1ms`) nếu văn bản vượt quá ngân sách token cho phép (`GATEWAY_MAX_INPUT_TOKENS`), ngăn chặn tấn công DoS và nguy cơ cạn kiệt chi phí/quota.
2. **Dynamic Fallback Chain:**
   * Tự động điều hướng và hạ cấp mô hình (Fallback Chain / Waterfall) khi mô hình chính gặp sự cố: Quota Exceeded (`429`), Quá thời gian (`DeadlineExceeded / Timeout`), Lỗi dịch vụ (`503`),... mà không làm gián đoạn trải nghiệm của người dùng cuối.
   * Hỗ trợ cấu hình số lần retry và exponential backoff trước khi kích hoạt chuyển đổi.
3. **Structured Audit Metrics ([`GatewayAuditMetrics`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/gateway.py#L36)):**
   * Trả về báo cáo kiểm toán đầy đủ cho mỗi request: `status` (`SUCCESS`, `FALLBACK_SUCCESS`, `REJECTED_GUARDRAIL`, `FAILED`), mô hình thực tế xử lý (`model_executed`), độ trễ (`latency_ms`), lý do fallback (`fallback_reason`), danh sách các model đã thử (`fallback_chain_attempted`) và lượng token tiêu thụ.

---

### 📌 Lab 03: Feature Engineering: Dynamic Smart Tagger & Semantic Cache
* **Vị trí:** [`src/foundations/services/product_tagger.py`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/services/product_tagger.py)
* **Mục tiêu:** Tự động sinh danh mục phân loại, gắn tags từ khóa và đánh giá an toàn nội dung cho sản phẩm/bài viết mới, đồng thời tối ưu hóa chi phí và tốc độ qua 2 lớp kiến trúc:

#### 2 Lớp kiến trúc:
1. **Semantic In-Memory Cache ([`SemanticInMemoryCache`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/services/product_tagger.py#L93)):**
   * **Tier 1 (Exact Hash Match):** Bảng băm SHA-256 chuỗi văn bản đã chuẩn hóa, cho tốc độ phản hồi cực nhanh (`latency < 0.1ms`).
   * **Tier 2 (Vector Cosine Match):** So khớp vector embedding (`gemini-embedding-001`) bằng Cosine Similarity trong RAM. Nếu độ tương đồng vượt ngưỡng (`threshold >= 0.85`), trả về kết quả ngay lập tức với **độ trễ < 5ms và $0 chi phí API**.
2. **AI-driven Processing ([`SmartFeatureService`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/services/product_tagger.py#L152)):**
   * Khi **Cache Miss**, ủy quyền qua [`ModelGateway`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/gateway.py) để gọi Gemini API trích xuất schema nghiêm ngặt:
     * [`ProductMetadata`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/services/product_tagger.py#L42): `primary_category`, `sub_category`, `tags`, `suggested_title`.
     * [`ContentSafetyAssessment`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/services/product_tagger.py#L31): `is_safe`, `risk_level` (`LOW`, `MEDIUM`, `HIGH`), `flagged_categories`, `safety_reason`.
   * Tự động lưu vector và metadata vào cache sau khi trích xuất thành công để phục vụ các yêu cầu tiếp theo.

---

### 📌 Lab 04: Dynamic Semantic Model Router & Cost Optimization
* **Vị trí:** [`src/foundations/infrastructure/model_router.py`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/model_router.py)
* **Mục tiêu:** Xây dựng module `ModelRouter` tự động phân loại intent và đánh giá độ phức tạp của yêu cầu người dùng, từ đó điều phối đến mô hình tối ưu nhất nhằm tiết kiệm chi phí vận hành.

#### 3 Trụ cột cốt lõi:
1. **High-Speed Heuristic Classifier ([`HeuristicIntentClassifier`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/model_router.py#L98)):**
   * Phân loại intent và chấm điểm độ phức tạp (0.0 - 1.0) với tốc độ siêu cao (`latency < 1ms`) bằng Regex và Rule-based Heuristics:
     * **Nhẹ (< 0.6):** `SUMMARIZATION`, `EXTRACTION`, `TRANSLATION`, `GENERAL_QA`.
     * **Nặng (>= 0.6):** `CODE_ANALYSIS`, `MULTI_STEP_REASONING`, `COMPLEX_ANALYSIS` (hợp đồng pháp lý, kiến trúc hệ thống).
2. **Dynamic Routing Strategy:**
   * **LIGHTWEIGHT_ROUTE:** Điều phối đến các mô hình nhỏ, tốc độ cao (`gemini-3.5-flash-lite`, `gemini-1.5-flash`).
   * **HEAVYWEIGHT_ROUTE:** Điều phối đến các mô hình năng lực cao (`gemini-2.5-flash`, `gemini-1.5-pro`) có khả năng suy luận sâu.
   * Tích hợp [`ModelGateway`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/gateway.py) để tự động fallback nếu model chính gặp lỗi quota (`429`) hoặc timeout.
3. **Pricing Engine & Cost Audit Report ([`PricingEngine`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/infrastructure/model_router.py#L65)):**
   * Đo lường chi phí thực tế (`actual_cost_usd`) so với chi phí đối chuẩn giả định nếu luôn gọi model đắt nhất (`benchmark_cost_usd` - Gemini 1.5 Pro).
   * Báo cáo tỷ lệ tiết kiệm chi phí (`savings_percentage`), chứng minh **tiết kiệm từ 90% - 95% chi phí** cho các tác vụ thông thường.

---

## 🛠️ Hướng dẫn cài đặt & Chạy dự án

1. **Chuẩn bị môi trường Python:**
   ```bash
   conda activate aie   # hoặc: python -m venv .venv && source .venv/bin/activate
   ```

2. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình biến môi trường:**
   Sao chép `.env.example` thành `.env` và điền `GEMINI_API_KEY`:
   ```bash
   cp .env.example .env
   ```

4. **Chạy demo tổng hợp:**
   ```bash
   python main.py
   ```

5. **Chạy Automated Unit Tests:**
   ```bash
   pytest tests/ -v
   ```
