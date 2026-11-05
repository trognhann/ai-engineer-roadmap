import json
from src.foundations import (
    CustomerSupportTicket,
    GeminiExtractionService,
    ModelGateway,
    GatewayStatus,
    SmartFeatureService,
)
from src.config.config import settings


def run_lab01_demo():
    print("\n" + "=" * 60)
    print("▶ LAB 01: Structured Outputs Extraction (Gemini)")
    print("=" * 60)

    sample_text = (
        "Xin chào, tôi là Nguyễn Trọng Nhân. "
        "Dịch vụ lưu trữ dữ liệu S3-compatible của bên bạn từ sáng nay liên tục báo lỗi 503 Service Unavailable, "
        "khiến toàn bộ hệ thống xử lý ảnh của công ty chúng tôi bị đình trệ. "
        "Cần kiểm tra ngay gateway và phản hồi phương án khắc phục trước 11h trưa nay!"
    )

    gemini_service = GeminiExtractionService()
    result = gemini_service.extract_ticket(sample_text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


def run_lab02_demo():
    print("\n" + "=" * 60)
    print("▶ LAB 02: Production AI Gateway (SLA, Guardrail & Dynamic Fallback)")
    print("=" * 60)

    gateway = ModelGateway()

    # Kịch bản 1: Thực thi chuẩn và trích xuất dữ liệu qua Gateway
    print("\n[Kịch bản 1] Gọi Gateway xử lý tác vụ thông thường:")
    sample_text = "Khách hàng Trần Văn Nam yêu cầu hỗ trợ gấp do không thể tạo đơn hàng trên cổng thanh toán."
    res1 = gateway.extract_ticket(sample_text)
    print(f"• Trạng thái: {res1.metrics.status.value}")
    print(f"• Model thực thi: {res1.metrics.model_executed}")
    print(f"• Thời gian phản hồi: {res1.metrics.latency_ms} ms")
    print(f"• Fallback đã kích hoạt: {res1.metrics.is_fallback}")
    print(f"• Kết quả trích xuất: {res1.data}")

    # Kịch bản 2: Token Budget Guardrail chặn yêu cầu quá tải (DoS Protection)
    print("\n[Kịch bản 2] Token Budget Guardrail chặn input vượt ngưỡng:")
    strict_gateway = ModelGateway(max_input_tokens=80)
    oversized_text = "Cảnh báo lỗi cơ sở dữ liệu trên cụm máy chủ sản xuất! " * 25
    res2 = strict_gateway.execute(oversized_text)
    print(f"• Trạng thái: {res2.metrics.status.value}")
    print(f"• Token ước tính: {res2.metrics.estimated_input_tokens} tokens")
    print(f"• Thông báo từ chối: {res2.metrics.error_message}")
    print(f"• Latency chặn: {res2.metrics.latency_ms} ms (không tốn chi phí gọi LLM)")

    # Kịch bản 3: Dynamic Fallback khi Primary Model gặp sự cố (429 Quota Exceeded / Timeout)
    print("\n[Kịch bản 3] Dynamic Fallback tự động hạ cấp xuống Backup Model:")
    fallback_gateway = ModelGateway(
        primary_model="gemini-pro-latest",  # Mô hình bị 429 Quota Exceeded trên gói Free Tier
        fallback_models=["gemini-2.5-flash", "gemini-3.5-flash-lite"],
        max_input_tokens=1000,
    )
    res3 = fallback_gateway.execute("Hãy cho 1 lời chào ngắn gọn và chuyên nghiệp.")
    print(f"• Trạng thái: {res3.metrics.status.value}")
    print(f"• Model ban đầu dự kiến: gemini-pro-latest")
    print(f"• Model thực tế đã xử lý: {res3.metrics.model_executed}")
    print(f"• Đã kích hoạt Fallback: {res3.metrics.is_fallback}")
    print(f"• Lý do Fallback: {res3.metrics.fallback_reason}")
    print(f"• Danh sách model đã thử: {res3.metrics.fallback_chain_attempted}")
    print(f"• Nội dung phản hồi: {res3.content.strip() if res3.content else None}")


def run_lab03_demo():
    print("\n" + "=" * 60)
    print("▶ LAB 03: Feature Engineering: Smart Tagger & Semantic Cache")
    print("=" * 60)

    service = SmartFeatureService(similarity_threshold=0.85)

    # 1. Cache Miss: Nội dung mới lần đầu xuất hiện -> Gọi LLM qua Gateway
    print("\n[Lần 1 - Cache Miss] Phân tích sản phẩm mới đăng tải:")
    item_1 = "Cần bán MacBook Pro 14 inch M3 Pro 18GB 512GB màu Space Black mới 99% chính hãng AppleCare+"
    print(f"• Input: \"{item_1}\"")
    res1 = service.process(item_1)
    print(f"• Cache Hit: {res1.metrics.cache_hit}")
    print(f"• Thời gian phản hồi: {res1.metrics.latency_ms} ms")
    print(f"• Model xử lý: {res1.metrics.model_used}")
    if res1.data:
        print(f"• Danh mục: {res1.data.primary_category} > {res1.data.sub_category}")
        print(f"• Tags gợi ý: {res1.data.tags}")
        print(f"• Tiêu đề SEO: {res1.data.suggested_title}")
        print(f"• An toàn nội dung: is_safe={res1.data.safety.is_safe}, risk={res1.data.safety.risk_level}")

    # 2. Semantic Cache Hit: Người dùng khác đăng bài có nội dung tương đồng
    print("\n[Lần 2 - Semantic Cache Hit] Bài viết tương tự về mặt ngữ nghĩa (paraphrased):")
    item_2 = "Pass lại MacBook Pro M3 Pro bản 18GB RAM 512GB SSD màu đen không gian còn bảo hành AppleCare"
    print(f"• Input tương tự: \"{item_2}\"")
    res2 = service.process(item_2)
    print(f"• Cache Hit: {res2.metrics.cache_hit}")
    print(f"• Độ tương đồng (Cosine Similarity): {res2.metrics.similarity_score}")
    print(f"• Thời gian phản hồi: {res2.metrics.latency_ms} ms (cực nhanh)")
    print(f"• Chi phí API: ${res2.metrics.cost_usd}")
    if res2.data:
        print(f"• Dữ liệu lấy từ Cache: {res2.data.primary_category} > {res2.data.sub_category} | Tags: {res2.data.tags}")

    # 3. Exact Hash Match: Bài viết trùng khớp chính xác 100%
    print("\n[Lần 3 - Exact Hash Match Hit] Yêu cầu trùng khớp chính xác (< 5ms):")
    res3 = service.process(item_1)
    print(f"• Cache Hit: {res3.metrics.cache_hit}")
    print(f"• Độ tương đồng: {res3.metrics.similarity_score}")
    print(f"• Thời gian phản hồi: {res3.metrics.latency_ms} ms (< 5ms theo SLA)")
    print(f"• Chi phí API: ${res3.metrics.cost_usd}")


def main():
    if not settings.GEMINI_API_KEY:
        print("[LƯU Ý] Chưa cấu hình GEMINI_API_KEY trong file .env!")
        return

    # Chạy demo Lab 03
    run_lab03_demo()


if __name__ == "__main__":
    main()
