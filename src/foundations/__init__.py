"""Foundations Layer: Clean Architecture.

Tầng 1: base/           - Lớp cơ sở (Base Class) & Data Contracts dùng chung
Tầng 2: providers/      - Các SDK tích hợp LLM (Gemini, OpenAI, Claude)
Tầng 3: infrastructure/ - Hạ tầng vận hành (ModelGateway, ModelRouter)
Tầng 4: services/       - Tính năng nghiệp vụ thực tế (SmartFeatureService, SemanticCache)
"""

# 1. Base Layer
from src.foundations.base import (
    BaseExtractionService,
    CustomerSupportTicket,
    ExecutionMetrics,
    ServiceResponse,
)

# 2. Providers Layer
from src.foundations.providers import (
    GeminiExtractionService,
    RobustExtractionService,
    OpenAIExtractionService,
    ClaudeExtractionService,
)

# 3. Infrastructure Layer
from src.foundations.infrastructure import (
    ModelGateway,
    TokenBudgetGuardrail,
    GatewayAuditMetrics,
    GatewayResponse,
    GatewayStatus,
    ModelRouter,
    RouteType,
    UserIntent,
    RouterDecision,
    CostAuditReport,
    RouterExecutionResponse,
    HeuristicIntentClassifier,
    PricingEngine,
)

# 4. Services Layer
from src.foundations.services import (
    ContentSafetyAssessment,
    ProductMetadata,
    TaggingMetrics,
    SmartFeatureResponse,
    SemanticInMemoryCache,
    SmartFeatureService,
)

__all__ = [
    # Base
    "BaseExtractionService",
    "CustomerSupportTicket",
    "ExecutionMetrics",
    "ServiceResponse",
    # Providers
    "GeminiExtractionService",
    "RobustExtractionService",
    "OpenAIExtractionService",
    "ClaudeExtractionService",
    # Infrastructure
    "ModelGateway",
    "TokenBudgetGuardrail",
    "GatewayAuditMetrics",
    "GatewayResponse",
    "GatewayStatus",
    "ModelRouter",
    "RouteType",
    "UserIntent",
    "RouterDecision",
    "CostAuditReport",
    "RouterExecutionResponse",
    "HeuristicIntentClassifier",
    "PricingEngine",
    # Services
    "ContentSafetyAssessment",
    "ProductMetadata",
    "TaggingMetrics",
    "SmartFeatureResponse",
    "SemanticInMemoryCache",
    "SmartFeatureService",
]
