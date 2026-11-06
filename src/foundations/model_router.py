"""Dynamic Semantic Model Router Module (Lab 04).

Tính năng:
1. Heuristic / Lightweight Classification Engine:
   - Đánh giá độ phức tạp và intent của prompt với tốc độ cao (< 1ms).
2. Tự động điều phối (Route) tới model phù hợp:
   - LIGHTWEIGHT_ROUTE: Dành cho tóm tắt, trích xuất, dịch thuật, Q&A đơn giản.
   - HEAVYWEIGHT_ROUTE: Dành cho lập luận đa bước, debug code, phân tích hợp đồng / logic sâu.
3. Đo lường & Báo cáo chi phí tiết kiệm (Cost Audit):
   - So sánh chi phí thực tế với chi phí giả định nếu luôn gọi model đắt nhất (Heavyweight Benchmark).
"""

import re
import time
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src.config.config import settings
from src.foundations.gateway import ModelGateway, GatewayStatus

logger = logging.getLogger("ModelRouter")


# ---------------------------------------------------------------------------
# 1. Enums & Data Contracts
# ---------------------------------------------------------------------------
class RouteType(str, Enum):
    """Loại tuyến điều phối mô hình."""
    LIGHTWEIGHT = "LIGHTWEIGHT_ROUTE"
    HEAVYWEIGHT = "HEAVYWEIGHT_ROUTE"


class UserIntent(str, Enum):
    """Ý định chính của người dùng được phân loại."""
    SUMMARIZATION = "SUMMARIZATION"
    EXTRACTION = "EXTRACTION"
    TRANSLATION = "TRANSLATION"
    GENERAL_QA = "GENERAL_QA"
    CODE_ANALYSIS = "CODE_ANALYSIS"
    MULTI_STEP_REASONING = "MULTI_STEP_REASONING"
    COMPLEX_ANALYSIS = "COMPLEX_ANALYSIS"


class RouterDecision(BaseModel):
    """Quyết định điều phối mô hình cùng phân tích intent & độ phức tạp."""
    route: RouteType
    target_model: str
    intent: UserIntent
    complexity_score: float = Field(ge=0.0, le=1.0)
    decision_reasons: List[str] = Field(default_factory=list)
    routing_latency_ms: float = 0.0


class CostAuditReport(BaseModel):
    """Báo cáo kiểm toán và đo lường chi phí tiết kiệm."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    actual_model: str
    actual_cost_usd: float = 0.0
    benchmark_model: str
    benchmark_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    savings_percentage: float = 0.0


class RouterExecutionResponse(BaseModel):
    """Kết quả phản hồi hoàn chỉnh của ModelRouter."""
    content: Optional[str] = None
    routing_decision: RouterDecision
    cost_audit: CostAuditReport
    execution_latency_ms: float = 0.0
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Pricing Engine (Google AI Studio / Industry Standards)
# ---------------------------------------------------------------------------
class PricingEngine:
    """Công cụ tính toán chi phí token theo chuẩn bảng giá Google AI Studio.
    
    Giá tiêu chuẩn (USD / 1.000.000 tokens):
    - Heavyweight Benchmark (Gemini 1.5 Pro / 2.5 Pro): $1.25 input / $5.00 output
    - Lightweight / Flash Models (Gemini Flash / Flash-Lite): $0.075 input / $0.30 output
    """

    BENCHMARK_MODEL_NAME = "gemini-1.5-pro (Benchmark)"
    BENCHMARK_PROMPT_PRICE_PER_M = 1.25
    BENCHMARK_COMPLETION_PRICE_PER_M = 5.00

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gemini-3.5-flash-lite": {"prompt": 0.075, "completion": 0.30},
        "gemini-flash-latest": {"prompt": 0.075, "completion": 0.30},
        "gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
        "gemini-2.5-pro": {"prompt": 1.25, "completion": 5.00},
        "gemini-pro-latest": {"prompt": 1.25, "completion": 5.00},
    }

    @classmethod
    def calculate_cost(cls, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = cls.MODEL_PRICING.get(model_name, {"prompt": 0.075, "completion": 0.30})
        prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
        return round(prompt_cost + completion_cost, 7)

    @classmethod
    def calculate_benchmark_cost(cls, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1_000_000.0) * cls.BENCHMARK_PROMPT_PRICE_PER_M
        completion_cost = (completion_tokens / 1_000_000.0) * cls.BENCHMARK_COMPLETION_PRICE_PER_M
        return round(prompt_cost + completion_cost, 7)


# ---------------------------------------------------------------------------
# 3. High-Speed Heuristic Intent & Complexity Classifier
# ---------------------------------------------------------------------------
class HeuristicIntentClassifier:
    """Bộ phân loại intent và độ phức tạp tốc độ siêu cao (< 1ms) bằng Regex và Heuristics."""

    CODE_PATTERNS = [
        re.compile(r"```(?:python|javascript|js|ts|go|cpp|c|java|sql|html|css|json)?", re.IGNORECASE),
        re.compile(r"\b(def\s+\w+|class\s+\w+|import\s+\w+|function\s+\w+|traceback|syntaxerror|nullpointer|exception|refactor|debug|segfault|sql query|algorithm)\b", re.IGNORECASE),
        re.compile(r"\b(mã nguồn|sửa lỗi code|viết code|debug code|thuật toán|tối ưu mã|phân tích bug)\b", re.IGNORECASE),
    ]

    REASONING_PATTERNS = [
        re.compile(r"\b(từng bước|step by step|chứng minh|giải thích tại sao|suy luận logic|chuỗi suy luận|lý giải cơ chế|bài toán logic)\b", re.IGNORECASE),
        re.compile(r"\b(prove that|multi-step|chain of thought|calculate step-by-step|mathematical proof)\b", re.IGNORECASE),
    ]

    COMPLEX_ANALYSIS_PATTERNS = [
        re.compile(r"\b(phân tích chuyên sâu|phân tích điều khoản|thẩm định hợp đồng|rủi ro pháp lý|kiến trúc hệ thống|system architecture|đánh giá rủi ro|so sánh chuyên sâu|chính sách bảo mật)\b", re.IGNORECASE),
        re.compile(r"\b(contract analysis|legal risk|system design trade-offs|deep architectural analysis)\b", re.IGNORECASE),
    ]

    SUMMARIZATION_PATTERNS = [
        re.compile(r"\b(tóm tắt|rút gọn|summarize|summary|tổng kết ngắn|ý chính của|nội dung cốt lõi)\b", re.IGNORECASE),
    ]

    EXTRACTION_PATTERNS = [
        re.compile(r"\b(trích xuất|extract|lấy ra|bóc tách|danh sách thực thể|thông tin liên hệ|số điện thoại|email)\b", re.IGNORECASE),
    ]

    TRANSLATION_PATTERNS = [
        re.compile(r"\b(dịch|translate|dịch sang tiếng|tiếng anh sang tiếng việt|tiếng việt sang tiếng anh|nghĩa tiếng việt là gì)\b", re.IGNORECASE),
    ]

    @classmethod
    def evaluate(cls, text: str, complexity_threshold: float = 0.6) -> tuple[UserIntent, float, List[str]]:
        """Đánh giá intent và chấm điểm độ phức tạp (0.0 đến 1.0)."""
        reasons: List[str] = []
        clean_text = text.strip()

        # 1. Kiểm tra Code & Debugging (Ưu tiên cao nhất)
        for pat in cls.CODE_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Phát hiện cú pháp mã nguồn, từ khóa lập trình hoặc yêu cầu debug code")
                return UserIntent.CODE_ANALYSIS, 0.85, reasons

        # 2. Kiểm tra Multi-step Reasoning
        for pat in cls.REASONING_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Phát hiện yêu cầu lập luận đa bước, tư duy logic hoặc chứng minh bài toán")
                return UserIntent.MULTI_STEP_REASONING, 0.90, reasons

        # 3. Kiểm tra Tóm tắt / Rút gọn (Tác vụ nhẹ)
        for pat in cls.SUMMARIZATION_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Yêu cầu tóm tắt, rút gọn văn bản mức độ thông thường")
                return UserIntent.SUMMARIZATION, 0.20, reasons

        # 4. Kiểm tra Trích xuất thực thể (Tác vụ nhẹ)
        for pat in cls.EXTRACTION_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Yêu cầu trích xuất thông tin, thực thể cơ bản")
                return UserIntent.EXTRACTION, 0.25, reasons

        # 5. Kiểm tra Dịch thuật (Tác vụ nhẹ)
        for pat in cls.TRANSLATION_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Yêu cầu dịch thuật văn bản thông thường")
                return UserIntent.TRANSLATION, 0.20, reasons

        # 6. Kiểm tra Phân tích hợp đồng / Pháp lý / Kiến trúc phức tạp
        for pat in cls.COMPLEX_ANALYSIS_PATTERNS:
            if pat.search(clean_text):
                reasons.append("Phát hiện yêu cầu phân tích hợp đồng, pháp lý hoặc kiến trúc hệ thống chuyên sâu")
                return UserIntent.COMPLEX_ANALYSIS, 0.80, reasons

        # 7. Heuristic độ dài cho General Q&A
        word_count = len(clean_text.split())
        if word_count > 150:
            reasons.append(f"Văn bản đầu vào dài ({word_count} từ), đòi hỏi khả năng xử lý ngữ cảnh rộng")
            return UserIntent.COMPLEX_ANALYSIS, 0.70, reasons

        reasons.append("Yêu cầu hỏi đáp, cung cấp thông tin tổng quát ngắn gọn")
        return UserIntent.GENERAL_QA, 0.30, reasons


# ---------------------------------------------------------------------------
# 4. Production Model Router
# ---------------------------------------------------------------------------
class ModelRouter:
    """Hệ thống Dynamic Semantic Router định tuyến thông minh giữa mô hình nhẹ và mạnh."""

    def __init__(
        self,
        lightweight_model: Optional[str] = None,
        heavyweight_model: Optional[str] = None,
        complexity_threshold: Optional[float] = None,
        gateway: Optional[ModelGateway] = None,
    ):
        self.lightweight_model = lightweight_model or getattr(
            settings, "ROUTER_LIGHTWEIGHT_MODEL", "gemini-3.5-flash-lite"
        )
        self.heavyweight_model = heavyweight_model or getattr(
            settings, "ROUTER_HEAVYWEIGHT_MODEL", "gemini-2.5-flash"
        )
        self.complexity_threshold = complexity_threshold or getattr(
            settings, "ROUTER_COMPLEXITY_THRESHOLD", 0.6
        )
        self.gateway = gateway or ModelGateway(
            primary_model=self.heavyweight_model,
            fallback_models=[self.lightweight_model, "gemini-flash-latest"],
        )

    def route_decision(self, prompt: str) -> RouterDecision:
        """Đưa ra quyết định định tuyến trong thời gian cực ngắn (< 1ms)."""
        t0 = time.perf_counter()

        intent, score, reasons = HeuristicIntentClassifier.evaluate(
            text=prompt,
            complexity_threshold=self.complexity_threshold,
        )

        if score >= self.complexity_threshold:
            route = RouteType.HEAVYWEIGHT
            target_model = self.heavyweight_model
            reasons.append(f"Điểm độ phức tạp ({score:.2f}) >= ngưỡng ({self.complexity_threshold:.2f}) -> Chọn Heavyweight Route")
        else:
            route = RouteType.LIGHTWEIGHT
            target_model = self.lightweight_model
            reasons.append(f"Điểm độ phức tạp ({score:.2f}) < ngưỡng ({self.complexity_threshold:.2f}) -> Chọn Lightweight Route")

        routing_time_ms = (time.perf_counter() - t0) * 1000

        return RouterDecision(
            route=route,
            target_model=target_model,
            intent=intent,
            complexity_score=score,
            decision_reasons=reasons,
            routing_latency_ms=round(routing_time_ms, 3),
        )

    def route_and_execute(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> RouterExecutionResponse:
        """Định tuyến và thực thi request qua ModelGateway, đồng thời kiểm toán chi phí tiết kiệm."""
        start_time = time.perf_counter()

        # BƯỚC 1: Đưa ra quyết định điều phối
        decision = self.route_decision(prompt)
        logger.info(
            f"[ModelRouter] Route: {decision.route.value} | Model: {decision.target_model} | Intent: {decision.intent.value} | Latency: {decision.routing_latency_ms}ms"
        )

        # BƯỚC 2: Cấu hình Gateway với model đã chọn làm Primary
        # Đảm bảo vẫn có Fallback nếu model được chọn gặp sự cố quota
        if decision.route == RouteType.LIGHTWEIGHT:
            fallback_models = [self.heavyweight_model, "gemini-flash-latest"]
        else:
            fallback_models = [self.lightweight_model, "gemini-flash-latest"]

        gateway_for_route = ModelGateway(
            primary_model=decision.target_model,
            fallback_models=fallback_models,
            api_key=self.gateway.api_key,
        )

        # BƯỚC 3: Thực thi gọi mô hình
        gw_response = gateway_for_route.execute(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2 if decision.route == RouteType.LIGHTWEIGHT else 0.7,
        )

        total_exec_time_ms = (time.perf_counter() - start_time) * 1000

        if gw_response.success and gw_response.content:
            actual_model = gw_response.metrics.model_executed or decision.target_model
            prompt_tokens = gw_response.metrics.prompt_tokens
            completion_tokens = gw_response.metrics.completion_tokens

            # BƯỚC 4: Tính toán Cost Audit & Tiết kiệm chi phí
            actual_cost = PricingEngine.calculate_cost(
                model_name=actual_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            benchmark_cost = PricingEngine.calculate_benchmark_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            cost_saved = max(0.0, round(benchmark_cost - actual_cost, 7))
            savings_pct = round((cost_saved / benchmark_cost) * 100, 2) if benchmark_cost > 0 else 0.0

            cost_audit = CostAuditReport(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                actual_model=actual_model,
                actual_cost_usd=actual_cost,
                benchmark_model=PricingEngine.BENCHMARK_MODEL_NAME,
                benchmark_cost_usd=benchmark_cost,
                cost_saved_usd=cost_saved,
                savings_percentage=savings_pct,
            )

            return RouterExecutionResponse(
                content=gw_response.content,
                routing_decision=decision,
                cost_audit=cost_audit,
                execution_latency_ms=round(total_exec_time_ms, 2),
                success=True,
            )
        else:
            return RouterExecutionResponse(
                content=None,
                routing_decision=decision,
                cost_audit=CostAuditReport(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    actual_model=decision.target_model,
                    actual_cost_usd=0.0,
                    benchmark_model=PricingEngine.BENCHMARK_MODEL_NAME,
                    benchmark_cost_usd=0.0,
                    cost_saved_usd=0.0,
                    savings_percentage=0.0,
                ),
                execution_latency_ms=round(total_exec_time_ms, 2),
                success=False,
                error_message=gw_response.metrics.error_message or "Không thể thực thi yêu cầu qua ModelRouter.",
            )
