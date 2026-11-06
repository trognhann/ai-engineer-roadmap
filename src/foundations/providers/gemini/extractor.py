import time
import json
from typing import Optional
from pydantic import ValidationError
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.config.config import settings
from src.foundations.base import (
    BaseExtractionService,
    CustomerSupportTicket,
    ServiceResponse,
    ExecutionMetrics,
)


class GeminiExtractionService(BaseExtractionService):
    """Dịch vụ trích xuất thông tin qua Google Gemini API, kế thừa BaseExtractionService."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=key)

        resolved_model = (
            model_name
            or getattr(settings, "GEMINI_MODEL", None)
            or (settings.DEFAULT_MODEL if "gemini" in getattr(settings, "DEFAULT_MODEL", "") else "gemini-2.5-flash")
        )
        super().__init__(api_key=key, model_name=resolved_model)

    def extract_ticket(self, raw_text: str, max_retries: int = 2) -> ServiceResponse[CustomerSupportTicket]:
        start_time = time.perf_counter()
        retries = 0
        prompt_tokens = 0
        completion_tokens = 0

        # Cấu hình Native Structured Outputs với Pydantic schema
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=CustomerSupportTicket,
            temperature=settings.DEFAULT_TEMPERATURE,
        )

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=self.system_instruction,
        )

        current_prompt = self.format_prompt(raw_text)

        while retries <= max_retries:
            try:
                response = model.generate_content(current_prompt)

                # Thu thập thông tin usage token từ metadata của Gemini
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens += response.usage_metadata.prompt_token_count or 0
                    completion_tokens += response.usage_metadata.candidates_token_count or 0

                raw_json = response.text
                parsed_ticket = CustomerSupportTicket.model_validate_json(raw_json)

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return ServiceResponse(
                    data=parsed_ticket,
                    metrics=ExecutionMetrics(
                        latency_ms=round(elapsed_ms, 2),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_cost_usd=0.0,  # Free Tier trên Google AI Studio
                        retries_count=retries,
                    ),
                    success=True,
                )

            except (ValidationError, json.JSONDecodeError) as err:
                retries += 1
                if retries > max_retries:
                    break
                current_prompt = self.build_retry_prompt(raw_text, str(err))

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return self.create_error_response(
                    error_message=f"Gemini API Error: {str(e)}",
                    latency_ms=elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retries_count=retries,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return self.create_error_response(
            error_message="Vượt quá số lần retry: Dữ liệu không thỏa mãn Pydantic schema.",
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries_count=retries,
        )


# Alias tương thích ngược cho code cũ
RobustExtractionService = GeminiExtractionService



# Alias tương thích ngược cho code cũ
RobustExtractionService = GeminiExtractionService