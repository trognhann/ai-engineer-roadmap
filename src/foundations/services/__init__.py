"""Services / Features Layer: Smart Product Tagger & Semantic In-Memory Cache."""

from src.foundations.services.product_tagger import (
    ContentSafetyAssessment,
    ProductMetadata,
    TaggingMetrics,
    SmartFeatureResponse,
    SemanticInMemoryCache,
    SmartFeatureService,
)

__all__ = [
    "ContentSafetyAssessment",
    "ProductMetadata",
    "TaggingMetrics",
    "SmartFeatureResponse",
    "SemanticInMemoryCache",
    "SmartFeatureService",
]

