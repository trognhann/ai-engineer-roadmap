# AI Engineer Roadmap 🚀

Kho lưu trữ tài liệu, mã nguồn thực hành và các module chuyên sâu trên lộ trình trở thành **AI Engineer (GenAI / LLM Infrastructure & Applications)**.

---

## 📁 Cấu trúc thư mục

```text
ai-engineer-roadmap/
├── notebooks/                     # Thử nghiệm nhanh với Jupyter / Colab
├── tests/                         # Unit tests & integration tests (pytest)
│   ├── __init__.py
│   └── test_gateway.py            # Test suite cho ModelGateway & Guardrail
├── src/
│   ├── config/                    # Cấu hình dự án (Pydantic Settings, .env)
│   │   ├── __init__.py
│   │   └── config.py
│   ├── foundations/               # Module Foundations & Core APIs
│   │   ├── __init__.py
│   │   ├── gateway.py             # [LAB 02] Production AI Gateway (SLA, Guardrail, Fallback)
│   │   ├── base/                  # Thành phần chung dùng theo OOP
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py         # Data Contracts (CustomerSupportTicket, ExecutionMetrics, ...)
│   │   │   └── extractor.py       # Lớp trừu tượng BaseExtractionService (ABC)
│   │   ├── gemini/                # Triển khai cho Google Gemini API
│   │   │   ├── __init__.py
│   │   │   └── extractor.py       # GeminiExtractionService
│   │   ├── openai/                # Triển khai cho OpenAI API (GPT-4o, GPT-4o-mini)
│   │   │   ├── __init__.py
│   │   │   └── extractor.py       # OpenAIExtractionService
│   │   └── claude/                # Triển khai cho Anthropic Claude API
│   │       ├── __init__.py
│   │       └── extractor.py       # ClaudeExtractionService
│   ├── rag/                       # Ingestion, Chunking, Embeddings, Vector Stores, Reranking
│   ├── agents/                    # Tools, Memory, Multi-agent Systems (LangGraph, CrewAI)
│   ├── evals/                     # LLM-as-a-judge, Benchmarks, Tracing (LangSmith, Phoenix)
│   └── serving/                   # Serving APIs (FastAPI), Local inference (vLLM, Ollama), Docker
├── .env.example                   # Biến môi trường mẫu
├── .env                           # Biến môi trường local (không commit lên git)
├── requirements.txt               # Danh sách thư viện phụ thuộc trực tiếp
├── main.py                        # Entrypoint chạy demo Lab 01 & Lab 02
└── README.md
```

---

## 🔬 Các bài Lab thực hành

### 📌 Lab 01: Robust Structured Outputs & Data Contracts
* **Vị trí:** [`src/foundations/base/`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base) và [`src/foundations/gemini/`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/gemini)
* **Mục tiêu:** Xây dựng hệ thống trích xuất dữ liệu có cấu trúc từ văn bản thô (ticket hỗ trợ khách hàng) tuân thủ nghiêm ngặt schema định sẵn bằng Pydantic.
* **Đặc điểm:**
  * Kế thừa cấu trúc OOP với [`BaseExtractionService`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base/extractor.py).
  * Chuẩn hóa Schema [`CustomerSupportTicket`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/base/schemas.py) tương thích với Native Structured Outputs (Gemini & OpenAI).
  * Cơ chế tự động retry và sửa lỗi khi dữ liệu trả về sai cấu trúc JSON.

---

### 📌 Lab 02: Production AI Gateway (SLA, Token Guardrail & Dynamic Fallback)
* **Vị trí:** [`src/foundations/gateway.py`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/gateway.py)
* **Mục tiêu:** Đóng vai trò là AI Engineer phụ trách hạ tầng và độ ổn định của hệ thống, xây dựng module `ModelGateway` quản lý kết nối LLM đáp ứng tiêu chuẩn sản xuất (Production-ready).

#### 3 Trụ cột cốt lõi:
1. **Token Budget Guardrail ([`TokenBudgetGuardrail`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/gateway.py#L55)):**
   * Ước tính trước số lượng token đầu vào (heuristic cho tiếng Việt/UTF-8).
   * Từ chối xử lý ngay lập tức (`latency < 1ms`) nếu văn bản vượt quá ngân sách token cho phép (`GATEWAY_MAX_INPUT_TOKENS`), ngăn chặn tấn công DoS và nguy cơ cạn kiệt chi phí/quota.
2. **Dynamic Fallback Chain:**
   * Tự động điều hướng và hạ cấp mô hình (Fallback Chain / Waterfall) khi mô hình chính gặp sự cố: Quota Exceeded (`429`), Quá thời gian (`DeadlineExceeded / Timeout`), Lỗi dịch vụ (`503`),... mà không làm gián đoạn trải nghiệm của người dùng cuối.
   * Hỗ trợ cấu hình số lần retry và exponential backoff trước khi kích hoạt chuyển đổi.
3. **Structured Audit Metrics ([`GatewayAuditMetrics`](file:///Users/trognhann/ai-engineer-roadmap/src/foundations/gateway.py#L36)):**
   * Trả về báo cáo kiểm toán đầy đủ cho mỗi request: `status` (`SUCCESS`, `FALLBACK_SUCCESS`, `REJECTED_GUARDRAIL`, `FAILED`), mô hình thực tế xử lý (`model_executed`), độ trễ (`latency_ms`), lý do fallback (`fallback_reason`), danh sách các model đã thử (`fallback_chain_attempted`) và lượng token tiêu thụ.

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

4. **Chạy demo tổng hợp (Lab 01 & Lab 02):**
   ```bash
   python main.py
   ```

5. **Chạy Automated Unit Tests:**
   ```bash
   pytest tests/test_gateway.py -v
   ```
