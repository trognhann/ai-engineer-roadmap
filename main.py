import json
from src.foundations import (
    CustomerSupportTicket,
    GeminiExtractionService,
    OpenAIExtractionService,
    ClaudeExtractionService,
)
from src.config.config import settings

def main():
    if not settings.GEMINI_API_KEY:
        print("[LƯU Ý] Chưa cấu hình GEMINI_API_KEY trong file .env!")
        print("Truy cập https://aistudio.google.com/app/apikey để lấy key miễn phí.")
        return

    sample_text = (
        "Xin chào, tôi là Nguyễn Trọng Nhân. "
        "Dịch vụ lưu trữ dữ liệu S3-compatible của bên bạn từ sáng nay liên tục báo lỗi 503 Service Unavailable, "
        "khiến toàn bộ hệ thống xử lý ảnh của công ty chúng tôi bị đình trệ. "
        "Cần kiểm tra ngay gateway và phản hồi phương án khắc phục trước 11h trưa nay!"
    )

    print("--- Đang phân tích ticket qua Gemini API ---")
    gemini_service = GeminiExtractionService()
    result = gemini_service.extract_ticket(sample_text)

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()