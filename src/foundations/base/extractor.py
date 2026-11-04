from abc import ABC, abstractmethod
from typing import Optional
from src.foundations.base.schemas import (
    CustomerSupportTicket,
    ServiceResponse,
    ExecutionMetrics,
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "Bạn là hệ thống trích xuất thông tin tự động chuẩn hóa. "
    "Phân tích văn bản đầu vào và trích xuất thông tin khách hàng chính xác theo cấu trúc được yêu cầu."
)


class BaseExtractionService(ABC):
    """Lớp cơ sở (Base Class) cho các dịch vụ trích xuất thông tin từ LLM (Gemini, OpenAI, Claude).
    
    Chứa logic chung về cấu hình, định dạng prompt, xử lý retry và tạo response chuẩn hóa.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION

    def format_prompt(self, raw_text: str) -> str:
        """Định dạng prompt đầu vào gửi tới model."""
        return f"Văn bản cần xử lý:\n{raw_text}"

    def build_retry_prompt(self, raw_text: str, error_message: str) -> str:
        """Tạo prompt yêu cầu model sửa lại khi output trước đó không khớp Pydantic schema."""
        return (
            f"Văn bản gốc:\n{raw_text}\n\n"
            f"Dữ liệu JSON trước đó bị lỗi cấu trúc: {error_message}\n"
            "Hãy sửa lại và trả về đúng schema đã định nghĩa."
        )

    def create_error_response(
        self,
        error_message: str,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        retries_count: int = 0,
    ) -> ServiceResponse[CustomerSupportTicket]:
        """Tạo response chuẩn khi gặp lỗi."""
        return ServiceResponse(
            data=None,
            metrics=ExecutionMetrics(
                latency_ms=round(latency_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_cost_usd=0.0,
                retries_count=retries_count,
            ),
            success=False,
            error_message=error_message,
        )

    @abstractmethod
    def extract_ticket(self, raw_text: str, max_retries: int = 2) -> ServiceResponse[CustomerSupportTicket]:
        """Phương thức trừu tượng thực hiện trích xuất ticket.
        
        Các lớp con (GeminiExtractionService, OpenAIExtractionService, ClaudeExtractionService)
        bắt buộc phải hiện thực (override) phương thức này.
        """
        pass

