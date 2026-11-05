import time
import pytest
from src.foundations.product_tagger import (
    SemanticInMemoryCache,
    ProductMetadata,
    ContentSafetyAssessment,
    normalize_vector,
    cosine_similarity,
)


def _create_sample_metadata(name: str) -> ProductMetadata:
    return ProductMetadata(
        primary_category="Thiết bị công nghệ",
        sub_category="Tai nghe",
        tags=["tai nghe", "sony", "chong on"],
        suggested_title=f"Tai nghe {name}",
        safety=ContentSafetyAssessment(
            is_safe=True,
            risk_level="LOW",
            flagged_categories=["NONE"],
            safety_reason="Nội dung an toàn, không có yếu tố vi phạm",
        ),
    )


def test_vector_normalization_and_cosine():
    v1 = [1.0, 2.0, 3.0]
    norm_v1 = normalize_vector(v1)
    
    # Độ dài vector chuẩn hóa L2 phải xấp xỉ 1.0
    length = sum(x * x for x in norm_v1)
    assert abs(length - 1.0) < 1e-6

    # Vector giống hệt nhau có cosine similarity = 1.0
    sim_identical = cosine_similarity(norm_v1, norm_v1)
    assert abs(sim_identical - 1.0) < 1e-6

    # Vector vuông góc có cosine similarity = 0.0
    v_a = [1.0, 0.0]
    v_b = [0.0, 1.0]
    assert cosine_similarity(v_a, v_b) == 0.0


def test_semantic_cache_exact_match():
    cache = SemanticInMemoryCache(similarity_threshold=0.85)
    sample_meta = _create_sample_metadata("Sony WH-1000XM5")
    sample_text = "Bán tai nghe Sony WH-1000XM5 chống ồn tốt"
    fake_embedding = [0.1] * 100

    cache.add(sample_text, fake_embedding, sample_meta)

    # Đo thời gian tra cứu Exact match
    t0 = time.perf_counter()
    exact_res = cache.lookup_exact("  bán tai nghe sony wh-1000xm5 chống ồn tốt  ")
    t1 = time.perf_counter()

    exact_latency_ms = (t1 - t0) * 1000
    assert exact_res is not None
    matched_meta, score, matched_text = exact_res
    assert score == 1.0
    assert matched_meta.primary_category == "Thiết bị công nghệ"
    assert exact_latency_ms < 5.0, f"Latency {exact_latency_ms}ms vượt quá 5ms"


def test_semantic_cache_vector_similarity():
    cache = SemanticInMemoryCache(similarity_threshold=0.85)
    sample_meta = _create_sample_metadata("Sony WH-1000XM5")
    sample_text = "Tai nghe Sony chống ồn WH-1000XM5"
    base_embedding = [0.5] * 64

    cache.add(sample_text, base_embedding, sample_meta)

    # 1. Query có vector rất gần (độ tương đồng cao > 0.9)
    close_embedding = [0.49] * 64
    t0 = time.perf_counter()
    hit_res = cache.lookup_semantic(close_embedding, threshold=0.85)
    t1 = time.perf_counter()

    lookup_latency_ms = (t1 - t0) * 1000
    assert hit_res is not None
    _, score, _ = hit_res
    assert score >= 0.85
    assert lookup_latency_ms < 5.0, f"Lookup latency {lookup_latency_ms}ms vượt quá 5ms"

    # 2. Query có vector khác biệt (độ tương đồng thấp < 0.85)
    dissimilar_embedding = [1.0 if i % 2 == 0 else -1.0 for i in range(64)]
    miss_res = cache.lookup_semantic(dissimilar_embedding, threshold=0.85)
    assert miss_res is None


def test_semantic_cache_max_entries_eviction():
    cache = SemanticInMemoryCache(max_entries=3)
    meta = _create_sample_metadata("Item")

    for i in range(4):
        cache.add(f"Text {i}", [float(i)] * 10, meta)

    assert cache.size() == 3
    # Text 0 phải bị loại bỏ do vượt max_entries (LRU)
    assert cache.lookup_exact("Text 0") is None
    # Text 3 phải tồn tại
    assert cache.lookup_exact("Text 3") is not None

