# AI Engineer Roadmap 🚀

Kho lưu trữ tài liệu, mã nguồn thực hành và các module chuyên sâu trên lộ trình trở thành **AI Engineer (GenAI / LLM Engineering)**.

---

## 📁 Cấu trúc thư mục

```text
ai-engineer-roadmap/
├── notebooks/                     # Thử nghiệm nhanh với Jupyter / Colab
├── tests/                         # Unit tests & integration tests
├── src/
│   ├── config/                    # Cấu hình dự án (Pydantic Settings, .env)
│   ├── foundations/               # Prompt Engineering, Tokens, LLM APIs (OpenAI, Gemini, Anthropic)
│   ├── rag/                       # Ingestion, Chunking, Embeddings, Vector Stores, Reranking
│   ├── agents/                    # Tools, Memory, Multi-agent Systems (LangGraph, CrewAI)
│   ├── evals/                     # LLM-as-a-judge, Benchmarks, Tracing (LangSmith, Phoenix)
│   └── serving/                   # Serving APIs (FastAPI), Local inference (vLLM, Ollama), Docker
├── .env.example                   # Biến môi trường mẫu
├── .env                           # Biến môi trường local (không commit lên git)
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── main.py                        # Entrypoint kiểm tra hoặc chạy thử nghiệm
└── README.md
```

---

## 🛠️ Hướng dẫn cài đặt

1. **Clone repository và chuẩn bị môi trường:**
   ```bash
   conda activate aie   # hoặc dùng venv: python -m venv .venv && source .venv/bin/activate
   ```

2. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình môi trường:**
   Sao chép `.env.example` thành `.env` và điền các khóa API cần thiết:
   ```bash
   cp .env.example .env
   ```

4. **Chạy thử nghiệm:**
   ```bash
   python main.py
   ```
