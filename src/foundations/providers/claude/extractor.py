import time
from typing import Optional

from src.config.config import settings
from src.foundations.base import (
    BaseExtractionService,
    CustomerSupportTicket,
    ServiceResponse,
    ExecutionMetrics,
)


class ClaudeExtractionService(BaseExtractionService):
    """Dịch vụ trích xuất thông tin qua Anthropic Claude API (Tool Use extraction), kế thừa BaseExtractionService."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        key = api_key or settings.ANTHROPIC_API_KEY
        resolved_model = model_name or "claude-3-5-sonnet-20241022"
        super().__init__(api_key=key, model_name=resolved_model)

    def extract_ticket(self, raw_text: str, max_retries: int = 2) -> ServiceResponse[CustomerSupportTicket]:
        start_time = time.perf_counter()

        if not self.api_key:
            return self.create_error_response(
                error_message="Chưa cấu hình ANTHROPIC_API_KEY trong file .env hoặc tham số truyền vào.",
                latency_ms=0.0,
            )

        try:
            import anthropic
        except ImportError:
            return self.create_error_response(
                error_message="Thư viện 'anthropic' chưa được cài đặt trong môi trường. Vui lòng chạy: pip install anthropic",
                latency_ms=0.0,
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            # Claude hỗ trợ structured output thông qua Tool Calling
            tool_schema = {
                "name": "record_customer_ticket",
                "description": "Ghi nhận thông tin ticket của khách hàng theo cấu trúc chuẩn",
                "input_schema": CustomerSupportTicket.model_json_schema(),
            }

            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=self.system_instruction,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "record_customer_ticket"},
                messages=[
                    {"role": "user", "content": self.format_prompt(raw_text)}
                ],
            )

            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            tool_input = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "record_customer_ticket":
                    tool_input = block.input
                    break

            if tool_input is None:
                raise ValueError("Claude không gọi tool trích xuất hợp lệ.")

            ticket = CustomerSupportTicket.model_validate(tool_input)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ServiceResponse(
                data=ticket,
                metrics=ExecutionMetrics(
                    latency_ms=round(elapsed_ms, 2),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_cost_usd=0.0,
                    retries_count=0,
                ),
                success=True,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return self.create_error_response(
                error_message=f"Claude API Error: {str(e)}",
                latency_ms=elapsed_ms,
            )

