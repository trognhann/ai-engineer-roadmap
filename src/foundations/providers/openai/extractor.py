import time
from typing import Optional
from openai import OpenAI

from src.config.config import settings
from src.foundations.base import (
    BaseExtractionService,
    CustomerSupportTicket,
    ServiceResponse,
    ExecutionMetrics,
)


class OpenAIExtractionService(BaseExtractionService):
    """Dịch vụ trích xuất thông tin qua OpenAI API (Structured Outputs), kế thừa BaseExtractionService."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        key = api_key or settings.OPENAI_API_KEY
        resolved_model = model_name or settings.DEFAULT_MODEL or "gpt-4o-mini"
        super().__init__(api_key=key, model_name=resolved_model)
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def extract_ticket(self, raw_text: str, max_retries: int = 2) -> ServiceResponse[CustomerSupportTicket]:
        start_time = time.perf_counter()

        if not self.api_key or not self.client:
            return self.create_error_response(
                error_message="Chưa cấu hình OPENAI_API_KEY trong file .env hoặc tham số truyền vào.",
                latency_ms=0.0,
            )

        try:
            # Sử dụng OpenAI Structured Outputs (beta.chat.completions.parse)
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": self.format_prompt(raw_text)},
                ],
                response_format=CustomerSupportTicket,
                temperature=settings.DEFAULT_TEMPERATURE,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
            completion_tokens = completion.usage.completion_tokens if completion.usage else 0

            ticket = completion.choices[0].message.parsed

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
                error_message=f"OpenAI API Error: {str(e)}",
                latency_ms=elapsed_ms,
            )

