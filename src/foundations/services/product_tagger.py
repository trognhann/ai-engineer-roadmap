"""Smart Feature Engineering & Semantic Cache Module (Lab 03).

Tính năng:
1. Semantic In-Memory Cache:
   - Kiểm tra bộ nhớ cache vector cho các nội dung tương tự.
   - Cache hit: Trả về kết quả tức thì với độ trễ < 5ms và $0 chi phí API.
2. AI-driven Processing:
   - Khi cache miss: Gọi LLM qua ModelGateway để trích xuất metadata có cấu trúc (Strict Pydantic).
   - Tự động phân loại danh mục, gắn tags SEO và đánh giá mức độ an toàn nội dung.
"""

import time
import math
import hashlib
import logging
from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel, Field

import google.generativeai as genai

from src.config.config import settings
from src.foundations.infrastructure.gateway import ModelGateway, GatewayStatus

logger = logging.getLogger("SmartFeatureService")


# ---------------------------------------------------------------------------
# 1. Data Contracts (Strict Pydantic Models)
# ---------------------------------------------------------------------------
class ContentSafetyAssessment(BaseModel):
    """Đánh giá an toàn và kiểm duyệt nội dung theo tiêu chuẩn sản xuất."""
    is_safe: bool = Field(description="Nội dung an toàn, không vi phạm tiêu chuẩn cộng đồng")
    risk_level: str = Field(description="Mức độ rủi ro: LOW, MEDIUM, HIGH")
    flagged_categories: List[str] = Field(
        description="Các danh mục vi phạm nếu có: SPAM, SCAM, INAPPROPRIATE, HATE_SPEECH, NONE"
    )
    safety_reason: Optional[str] = Field(description="Giải thích ngắn gọn lý do đánh giá an toàn")


class ProductMetadata(BaseModel):
    """Metadata có cấu trúc sinh từ nội dung sản phẩm hoặc bài viết người dùng."""
    primary_category: str = Field(description="Ngành hàng hoặc danh mục chính (e.g. Điện tử & Công nghệ, Thời trang)")
    sub_category: str = Field(description="Danh mục con chi tiết (e.g. Điện thoại di động, Tai nghe)")
    tags: List[str] = Field(description="Danh sách 3-6 tags từ khóa chuẩn hóa (chữ thường, không dấu/hashtag)")
    suggested_title: Optional[str] = Field(description="Tiêu đề chuẩn hóa SEO hoặc tiêu đề ngắn gọn cho sản phẩm")
    safety: ContentSafetyAssessment = Field(description="Đánh giá mức độ an toàn nội dung")


class TaggingMetrics(BaseModel):
    """Chỉ số hiệu năng và chi phí của quá trình Feature Tagging."""
    cache_hit: bool
    similarity_score: Optional[float] = None
    matched_text_snippet: Optional[str] = None
    latency_ms: float
    cost_usd: float = 0.0
    tokens_used: int = 0
    model_used: Optional[str] = None


class SmartFeatureResponse(BaseModel):
    """Kết quả trả về chuẩn hóa của SmartFeatureService."""
    data: Optional[ProductMetadata] = None
    metrics: TaggingMetrics
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Vector Math Utilities
# ---------------------------------------------------------------------------
def normalize_vector(vector: List[float]) -> List[float]:
    """Chuẩn hóa L2 vector để tính Cosine Similarity nhanh qua tích vô hướng (Dot Product)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Tính Cosine Similarity giữa 2 vector đã được chuẩn hóa L2."""
    if len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


# ---------------------------------------------------------------------------
# 3. Layer 1: Semantic In-Memory Cache
# ---------------------------------------------------------------------------
class CacheEntry:
    """Bản ghi lưu trữ trong Semantic Cache."""
    def __init__(self, text: str, embedding: List[float], metadata: ProductMetadata):
        self.text = text
        self.normalized_text = text.strip().lower()
        self.embedding = normalize_vector(embedding)
        self.metadata = metadata
        self.created_at = time.time()
        self.hit_count = 0


class SemanticInMemoryCache:
    """Bộ nhớ đệm ngữ nghĩa (Semantic Cache) 2 lớp: Exact Hash Match + Vector Cosine Match."""

    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._exact_index: Dict[str, CacheEntry] = {}
        self._entries: List[CacheEntry] = []

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def lookup_exact(self, text: str) -> Optional[Tuple[ProductMetadata, float, str]]:
        """Kiểm tra khớp chính xác qua Hash O(1) - thời gian < 0.05ms."""
        h = self._hash_text(text)
        entry = self._exact_index.get(h)
        if entry:
            entry.hit_count += 1
            return entry.metadata, 1.0, entry.text
        return None

    def lookup_semantic(
        self, query_embedding: List[float], threshold: Optional[float] = None
    ) -> Optional[Tuple[ProductMetadata, float, str]]:
        """Kiểm tra tương đồng ngữ nghĩa qua Cosine Similarity - thời gian < 2ms."""
        if not self._entries:
            return None

        norm_query = normalize_vector(query_embedding)
        thresh = threshold if threshold is not None else self.similarity_threshold

        best_score = -1.0
        best_entry: Optional[CacheEntry] = None

        for entry in self._entries:
            sim = cosine_similarity(norm_query, entry.embedding)
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= thresh:
            best_entry.hit_count += 1
            return best_entry.metadata, round(best_score, 4), best_entry.text

        return None

    def add(self, text: str, embedding: List[float], metadata: ProductMetadata) -> None:
        """Lưu trữ vào cả bảng băm chính xác và chỉ mục vector."""
        # Giới hạn kích thước cache (LRU đơn giản)
        if len(self._entries) >= self.max_entries:
            oldest = self._entries.pop(0)
            old_hash = self._hash_text(oldest.text)
            self._exact_index.pop(old_hash, None)

        entry = CacheEntry(text, embedding, metadata)
        h = self._hash_text(text)
        self._exact_index[h] = entry
        self._entries.append(entry)

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._exact_index.clear()
        self._entries.clear()


# ---------------------------------------------------------------------------
# 4. Layer 2: Smart Feature Service (Facade)
# ---------------------------------------------------------------------------
class SmartFeatureService:
    """Dịch vụ Feature Engineering cao cấp kết hợp Semantic Cache và Model Gateway."""

    SYSTEM_INSTRUCTION = (
        "Bạn là hệ thống Smart Feature Tagger & Content Safety Engine cho nền tảng thương mại điện tử / mạng xã hội. "
        "Phân tích nội dung sản phẩm hoặc bài đăng của người dùng để:\n"
        "1. Xác định primary_category (danh mục chính) và sub_category (danh mục con).\n"
        "2. Tạo danh sách 3-6 tags từ khóa phù hợp (viết thường, ngắn gọn).\n"
        "3. Gợi ý suggested_title chuẩn SEO.\n"
        "4. Đánh giá mức độ an toàn (safety): is_safe, risk_level (LOW, MEDIUM, HIGH), flagged_categories, safety_reason."
    )

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
        gateway: Optional[ModelGateway] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)

        threshold = similarity_threshold or getattr(settings, "SEMANTIC_CACHE_THRESHOLD", 0.85)
        max_entries = getattr(settings, "SEMANTIC_CACHE_MAX_ENTRIES", 1000)
        self.cache = SemanticInMemoryCache(similarity_threshold=threshold, max_entries=max_entries)

        self.embedding_model = embedding_model or getattr(settings, "EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.gateway = gateway or ModelGateway(
            primary_model=getattr(settings, "GATEWAY_PRIMARY_MODEL", "gemini-2.5-flash"),
            fallback_models=["gemini-3.5-flash-lite", "gemini-flash-latest"],
            api_key=self.api_key,
        )

    def get_embedding(self, text: str) -> List[float]:
        """Tạo embedding cho văn bản qua Gemini API."""
        response = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="semantic_similarity",
        )
        return response["embedding"]

    def process(self, content_text: str) -> SmartFeatureResponse:
        """Xử lý phân loại, gắn tags và kiểm duyệt nội dung kết hợp Semantic Cache.
        
        Quy trình:
        1. Kiểm tra Exact Match Hash Cache -> Nếu trúng: trả về ngay (< 0.05ms, $0).
        2. Sinh vector embedding của văn bản.
        3. Kiểm tra Semantic Cosine Similarity trong Vector Cache -> Nếu trúng: trả về ngay (< 2ms, $0).
        4. Nếu Cache Miss: Gọi LLM qua ModelGateway (với Dynamic Fallback & SLA Guardrail).
        5. Lưu kết quả vào Semantic Cache để phục vụ các yêu cầu tiếp theo.
        """
        start_time = time.perf_counter()

        # BƯỚC 1: Exact Hash Match Lookup
        exact_match = self.cache.lookup_exact(content_text)
        if exact_match:
            cached_data, sim_score, matched_text = exact_match
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[Semantic Cache HIT - Exact] Latency: {elapsed_ms:.2f}ms")
            return SmartFeatureResponse(
                data=cached_data,
                metrics=TaggingMetrics(
                    cache_hit=True,
                    similarity_score=sim_score,
                    matched_text_snippet=matched_text[:80],
                    latency_ms=round(elapsed_ms, 2),
                    cost_usd=0.0,
                    tokens_used=0,
                    model_used="semantic-cache-in-memory",
                ),
                success=True,
            )

        # BƯỚC 2: Sinh embedding và Semantic Similarity Lookup
        try:
            embedding = self.get_embedding(content_text)
            semantic_match = self.cache.lookup_semantic(embedding)
            if semantic_match:
                cached_data, sim_score, matched_text = semantic_match
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"[Semantic Cache HIT - Semantic] Similarity: {sim_score}, Latency: {elapsed_ms:.2f}ms")
                return SmartFeatureResponse(
                    data=cached_data,
                    metrics=TaggingMetrics(
                        cache_hit=True,
                        similarity_score=sim_score,
                        matched_text_snippet=matched_text[:80],
                        latency_ms=round(elapsed_ms, 2),
                        cost_usd=0.0,
                        tokens_used=0,
                        model_used="semantic-cache-in-memory",
                    ),
                    success=True,
                )
        except Exception as emb_err:
            logger.warning(f"[Embedding Error] Không thể tạo embedding: {emb_err}. Bỏ qua cache và gọi LLM trực tiếp.")
            embedding = None

        # BƯỚC 3: AI-driven Processing (Cache Miss)
        logger.info("[Semantic Cache MISS] Đang gọi LLM qua ModelGateway...")
        prompt = f"Phân tích và trích xuất thông tin cho nội dung sản phẩm/bài viết sau:\n\"\"\"{content_text}\"\"\""

        gw_response = self.gateway.execute(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            response_schema=ProductMetadata,
            temperature=0.0,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if gw_response.success and gw_response.data:
            metadata: ProductMetadata = gw_response.data

            # BƯỚC 4: Lưu vào Semantic Cache nếu có embedding
            if embedding is not None:
                self.cache.add(content_text, embedding, metadata)
                logger.info(f"[Semantic Cache Stored] Hiện có {self.cache.size()} bản ghi trong cache.")

            return SmartFeatureResponse(
                data=metadata,
                metrics=TaggingMetrics(
                    cache_hit=False,
                    similarity_score=0.0,
                    matched_text_snippet=None,
                    latency_ms=round(elapsed_ms, 2),
                    cost_usd=gw_response.metrics.cost_usd,
                    tokens_used=gw_response.metrics.total_tokens,
                    model_used=gw_response.metrics.model_executed,
                ),
                success=True,
            )
        else:
            return SmartFeatureResponse(
                data=None,
                metrics=TaggingMetrics(
                    cache_hit=False,
                    latency_ms=round(elapsed_ms, 2),
                    cost_usd=0.0,
                    tokens_used=0,
                    model_used=gw_response.metrics.model_executed,
                ),
                success=False,
                error_message=gw_response.metrics.error_message or "Không thể xử lý yêu cầu gắn tag sản phẩm.",
            )

