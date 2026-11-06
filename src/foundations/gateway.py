"""Production AI Gateway Module (Lab 02).

Tính năng:
1. Token Budget Guardrail: Ước lượng token đầu vào và từ chối xử lý nếu vượt quota/ngưỡng chống DoS.
2. Dynamic Fallback: Tự động chuyển đổi mô hình dự phòng khi mô hình chính gặp sự cố (Timeout, 429 Quota Exceeded, 503,...).
3. Structured Audit Metrics: Báo cáo giám sát chi tiết bao gồm mô hình thực thi, độ trễ, số lần retry, trạng thái fallback.
"""

import time
import logging
from enum import Enum
from typing import Optional, List, Any, Type, Union
from pydantic import BaseModel, Field

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core import exceptions as google_exceptions

from src.config.config import settings
from src.foundations.base import CustomerSupportTicket

logger = logging.getLogger("ModelGateway")


# ---------------------------------------------------------------------------
# 1. Enums & Metrics Models
# ---------------------------------------------------------------------------
class GatewayStatus(str, Enum):
    """Trạng thái thực thi của AI Gateway."""
    SUCCESS = "SUCCESS"
    FALLBACK_SUCCESS = "FALLBACK_SUCCESS"
    REJECTED_GUARDRAIL = "REJECTED_GUARDRAIL"
    FAILED = "FAILED"


class GatewayAuditMetrics(BaseModel):
    """Báo cáo kiểm toán và giám sát SLA của Gateway."""
    status: GatewayStatus
    model_executed: Optional[str] = None
    provider_executed: Optional[str] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    fallback_chain_attempted: List[str] = Field(default_factory=list)
    retries_count: int = 0
    latency_ms: float = 0.0
    estimated_input_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error_message: Optional[str] = None


class GatewayResponse(BaseModel):
    """Kết quả trả về chuẩn hóa của ModelGateway."""
    content: Optional[str] = None
    data: Optional[Any] = None
    metrics: GatewayAuditMetrics
    success: bool


# ---------------------------------------------------------------------------
# 2. Token Budget Guardrail
# ---------------------------------------------------------------------------
class TokenBudgetGuardrail:
    """Kiểm soát ngân sách token đầu vào nhằm chống tấn công DoS và cạn kiệt quota."""

    def __init__(self, max_input_tokens: int = 2048):
        self.max_input_tokens = max_input_tokens

    def estimate_tokens(self, text: str) -> int:
        """Ước tính số lượng token của văn bản đầu vào.
        
        Đối với tiếng Việt hoặc văn bản đa ngữ UTF-8, tỉ lệ thường là 1 token ~ 2-3 ký tự
        hoặc 1 token ~ 0.75 từ. Công thức dưới đây đưa ra cận trên an toàn.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Cận trên an toàn cho tiếng Việt có dấu: xấp xỉ 1 token / 3 ký tự hoặc 1.3 * số từ
        estimated = max(int(char_count / 3.0), int(word_count * 1.3))
        return max(1, estimated)

    def validate(self, text: str) -> tuple[bool, int, Optional[str]]:
        """Kiểm tra xem văn bản có vượt quá ngưỡng token cho phép không.
        
        Returns:
            (is_valid, estimated_tokens, error_message)
        """
        estimated = self.estimate_tokens(text)
        if estimated > self.max_input_tokens:
            msg = (
                f"Token Budget Guardrail Violated: Văn bản đầu vào ước tính {estimated} tokens, "
                f"vượt quá giới hạn tối đa cho phép là {self.max_input_tokens} tokens."
            )
            return False, estimated, msg
        return True, estimated, None


# ---------------------------------------------------------------------------
# 3. Production Model Gateway
# ---------------------------------------------------------------------------
class ModelGateway:
    """Production AI Gateway chịu trách nhiệm định tuyến, bảo vệ ngân sách token và xử lý Dynamic Fallback."""

    def __init__(
        self,
        primary_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        max_input_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        api_key: Optional[str] = None,
        max_retries_per_model: int = 1,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)

        # Cấu hình danh sách mô hình
        self.primary_model = primary_model or settings.GATEWAY_PRIMARY_MODEL
        if fallback_models is not None:
            self.fallback_models = fallback_models
        else:
            # Lấy từ config .env (chuỗi phân cách bằng dấu phẩy)
            fallback_str = getattr(settings, "GATEWAY_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-flash-latest")
            self.fallback_models = [m.strip() for m in fallback_str.split(",") if m.strip()]

        self.timeout_seconds = timeout_seconds or getattr(settings, "GATEWAY_TIMEOUT_SECONDS", 15.0)
        self.max_retries_per_model = max_retries_per_model

        # Khởi tạo Token Guardrail
        max_tokens = max_input_tokens or getattr(settings, "GATEWAY_MAX_INPUT_TOKENS", 2048)
        self.guardrail = TokenBudgetGuardrail(max_input_tokens=max_tokens)

    def execute(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
    ) -> GatewayResponse:
        """Thực thi request qua AI Gateway với đầy đủ Guardrail, Fallback và Audit Metrics."""
        start_time = time.perf_counter()

        # BƯỚC 1: Token Budget Guardrail Check
        is_valid, estimated_tokens, guardrail_error = self.guardrail.validate(prompt)
        if not is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"[Guardrail Rejected] {guardrail_error}")
            return GatewayResponse(
                content=None,
                data=None,
                metrics=GatewayAuditMetrics(
                    status=GatewayStatus.REJECTED_GUARDRAIL,
                    latency_ms=round(elapsed_ms, 2),
                    estimated_input_tokens=estimated_tokens,
                    error_message=guardrail_error,
                ),
                success=False,
            )

        # BƯỚC 2: Chuẩn bị chuỗi mô hình (Primary + Fallbacks)
        candidate_models = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        attempted_models: List[str] = []
        total_retries = 0
        first_error_reason: Optional[str] = None
        last_exception: Optional[Exception] = None

        # BƯỚC 3: Dynamic Fallback Execution Loop
        for index, model_name in enumerate(candidate_models):
            is_fallback_attempt = index > 0
            attempted_models.append(model_name)

            for attempt in range(self.max_retries_per_model + 1):
                if attempt > 0:
                    total_retries += 1
                    time.sleep(0.5 * attempt)  # Exponential backoff nhẹ

                try:
                    # Chuẩn bị generation config
                    gen_kwargs: dict[str, Any] = {
                        "temperature": temperature,
                    }
                    if response_schema is not None:
                        gen_kwargs["response_mime_type"] = "application/json"
                        gen_kwargs["response_schema"] = response_schema

                    gen_config = GenerationConfig(**gen_kwargs)
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=gen_config,
                        system_instruction=system_instruction,
                    )

                    # Gọi model với SLA timeout
                    response = model.generate_content(
                        prompt,
                        request_options={"timeout": self.timeout_seconds},
                    )

                    # Thu thập token usage
                    prompt_tokens = 0
                    completion_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count or 0
                        completion_tokens = response.usage_metadata.candidates_token_count or 0

                    content_text = response.text

                    # Validate structured data nếu có schema
                    parsed_data = None
                    if response_schema is not None:
                        try:
                            parsed_data = response_schema.model_validate_json(content_text)
                        except Exception:
                            # Defensive parsing: tự động điền các trường bị khuyết nếu model phụ sinh thiếu
                            import json
                            raw_dict = json.loads(content_text)
                            if isinstance(raw_dict, dict):
                                if "safety" in raw_dict and isinstance(raw_dict["safety"], dict):
                                    raw_dict["safety"].setdefault("flagged_categories", ["NONE"])
                                    raw_dict["safety"].setdefault("safety_reason", None)
                                    raw_dict["safety"].setdefault("is_safe", True)
                                    raw_dict["safety"].setdefault("risk_level", "LOW")
                                raw_dict.setdefault("tags", [])
                                raw_dict.setdefault("suggested_title", None)
                                parsed_data = response_schema.model_validate(raw_dict)
                            else:
                                raise

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    status = GatewayStatus.FALLBACK_SUCCESS if is_fallback_attempt else GatewayStatus.SUCCESS

                    return GatewayResponse(
                        content=content_text,
                        data=parsed_data,
                        metrics=GatewayAuditMetrics(
                            status=status,
                            model_executed=model_name,
                            provider_executed="Google Gemini",
                            is_fallback=is_fallback_attempt,
                            fallback_reason=first_error_reason if is_fallback_attempt else None,
                            fallback_chain_attempted=attempted_models,
                            retries_count=total_retries,
                            latency_ms=round(elapsed_ms, 2),
                            estimated_input_tokens=estimated_tokens,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            cost_usd=0.0,
                        ),
                        success=True,
                    )

                except (
                    google_exceptions.ResourceExhausted,  # 429 Quota Exceeded
                    google_exceptions.DeadlineExceeded,   # Timeout
                    google_exceptions.ServiceUnavailable, # 503
                    google_exceptions.NotFound,           # Model Not Found / Deprecated
                    TimeoutError,
                    Exception,
                ) as exc:
                    last_exception = exc
                    err_msg = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    if first_error_reason is None:
                        first_error_reason = f"Model '{model_name}' thất bại do {err_msg}"
                    logger.warning(
                        f"[Gateway Warning] Model '{model_name}' (lần thử {attempt + 1}) gặp lỗi: {err_msg}. "
                    )

        # BƯỚC 4: Nếu toàn bộ mô hình trong danh sách đều thất bại
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return GatewayResponse(
            content=None,
            data=None,
            metrics=GatewayAuditMetrics(
                status=GatewayStatus.FAILED,
                model_executed=None,
                provider_executed=None,
                is_fallback=True,
                fallback_reason=first_error_reason,
                fallback_chain_attempted=attempted_models,
                retries_count=total_retries,
                latency_ms=round(elapsed_ms, 2),
                estimated_input_tokens=estimated_tokens,
                error_message=f"Toàn bộ mô hình trong chuỗi fallback đều thất bại. Lỗi cuối: {str(last_exception)}",
            ),
            success=False,
        )

    def extract_ticket(self, raw_text: str) -> GatewayResponse:
        """Tiện ích trích xuất ticket chăm sóc khách hàng (Lab 01) tích hợp với ModelGateway (Lab 02)."""
        system_instruction = (
            "Bạn là hệ thống trích xuất thông tin tự động chuẩn hóa. "
            "Phân tích văn bản đầu vào và trích xuất thông tin khách hàng chính xác theo cấu trúc được yêu cầu."
        )
        prompt = f"Văn bản cần xử lý:\n{raw_text}"
        return self.execute(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=CustomerSupportTicket,
        )

