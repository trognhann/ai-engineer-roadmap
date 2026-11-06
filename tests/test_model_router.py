import time
import pytest
from src.foundations.infrastructure.model_router import (
    ModelRouter,
    RouteType,
    UserIntent,
    HeuristicIntentClassifier,
    PricingEngine,
)


def test_heuristic_classifier_speed():
    prompt = "Tóm tắt ngắn gọn các điểm chính trong báo cáo tài chính quý 3."
    t0 = time.perf_counter()
    intent, score, reasons = HeuristicIntentClassifier.evaluate(prompt)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    assert latency_ms < 5.0, f"Phân loại quá chậm: {latency_ms}ms"
    assert intent == UserIntent.SUMMARIZATION
    assert score < 0.6


def test_lightweight_route_intents():
    router = ModelRouter()

    # 1. Tác vụ tóm tắt
    p_sum = "Tóm tắt giúp tôi đoạn văn bản sau trong 2 câu ngắn gọn."
    dec_sum = router.route_decision(p_sum)
    assert dec_sum.route == RouteType.LIGHTWEIGHT
    assert dec_sum.intent == UserIntent.SUMMARIZATION

    # 2. Tác vụ dịch thuật
    p_trans = "Dịch đoạn này sang tiếng Anh: Chào buổi sáng quý khách."
    dec_trans = router.route_decision(p_trans)
    assert dec_trans.route == RouteType.LIGHTWEIGHT
    assert dec_trans.intent == UserIntent.TRANSLATION

    # 3. Tác vụ trích xuất thực thể
    p_ext = "Trích xuất danh sách số điện thoại và email từ danh bạ sau."
    dec_ext = router.route_decision(p_ext)
    assert dec_ext.route == RouteType.LIGHTWEIGHT
    assert dec_ext.intent == UserIntent.EXTRACTION


def test_heavyweight_route_intents():
    router = ModelRouter()

    # 1. Tác vụ Code / Debugging
    p_code = "Đoạn code sau bị lỗi RecursionError, hãy debug và sửa lại:\ndef foo(n): return foo(n-1)"
    dec_code = router.route_decision(p_code)
    assert dec_code.route == RouteType.HEAVYWEIGHT
    assert dec_code.intent == UserIntent.CODE_ANALYSIS
    assert dec_code.complexity_score >= 0.6

    # 2. Tác vụ Multi-step Reasoning
    p_reason = "Hãy giải bài toán sau từng bước (step by step) và chứng minh nghiệm duy nhất."
    dec_reason = router.route_decision(p_reason)
    assert dec_reason.route == RouteType.HEAVYWEIGHT
    assert dec_reason.intent == UserIntent.MULTI_STEP_REASONING
    assert dec_reason.complexity_score >= 0.6

    # 3. Tác vụ Hợp đồng / Pháp lý phức tạp
    p_legal = "Phân tích điều khoản bồi thường thiệt hại và rủi ro pháp lý trong hợp đồng ngoại thương sau."
    dec_legal = router.route_decision(p_legal)
    assert dec_legal.route == RouteType.HEAVYWEIGHT
    assert dec_legal.intent == UserIntent.COMPLEX_ANALYSIS
    assert dec_legal.complexity_score >= 0.6


def test_pricing_engine_savings():
    # Giả sử 1000 prompt tokens và 500 completion tokens
    prompt_tokens = 1000
    completion_tokens = 500

    actual_cost = PricingEngine.calculate_cost("gemini-3.5-flash-lite", prompt_tokens, completion_tokens)
    benchmark_cost = PricingEngine.calculate_benchmark_cost(prompt_tokens, completion_tokens)

    assert benchmark_cost > actual_cost
    savings = benchmark_cost - actual_cost
    savings_pct = (savings / benchmark_cost) * 100

    # Tuyến Lightweight phải tiết kiệm được trên 85% chi phí so với mô hình Pro đắt đỏ
    assert savings_pct > 85.0

