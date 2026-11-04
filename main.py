import json
from src.foundations import (
    CustomerSupportTicket,
    GeminiExtractionService,
    ModelGateway,
    GatewayStatus,
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


def main():
    if not settings.GEMINI_API_KEY:
        print("[LƯU Ý] Chưa cấu hình GEMINI_API_KEY trong file .env!")
        return

    # Chạy demo Lab 01
    # run_lab01_demo()

    # Chạy demo Lab 02
    run_lab02_demo()


if __name__ == "__main__":
    main()
