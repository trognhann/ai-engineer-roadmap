import pytest
from src.foundations.gateway import (
    TokenBudgetGuardrail,
    ModelGateway,
    GatewayStatus,
)


def test_token_budget_guardrail_estimation():
    guardrail = TokenBudgetGuardrail(max_input_tokens=100)
    
    # Text ngắn phải hợp lệ
    valid_text = "Xin chào, hệ thống cần hỗ trợ."
    is_valid, tokens, err = guardrail.validate(valid_text)
    assert is_valid is True
    assert tokens > 0
    assert err is None

    # Text dài vượt ngưỡng 100 token phải bị chặn
    long_text = "Nội dung kiểm tra guardrail vượt ngưỡng. " * 30
    is_valid, tokens, err = guardrail.validate(long_text)
    assert is_valid is False
    assert tokens > 100
    assert "Token Budget Guardrail Violated" in err


def test_model_gateway_guardrail_rejection():
    gateway = ModelGateway(max_input_tokens=50)
    long_input = "Cảnh báo quá tải token! " * 50
    
    response = gateway.execute(long_input)
    assert response.success is False
    assert response.metrics.status == GatewayStatus.REJECTED_GUARDRAIL
    assert response.metrics.latency_ms >= 0.0
    assert "Token Budget Guardrail Violated" in response.metrics.error_message


def test_model_gateway_dynamic_fallback_real():
    # Sử dụng model gemini-pro-latest (vốn hết quota 429 trên Free Tier) để kích hoạt fallback sang gemini-2.5-flash
    gateway = ModelGateway(
        primary_model="gemini-pro-latest",
        fallback_models=["gemini-2.5-flash"],
        max_input_tokens=500,
    )
    
    response = gateway.execute("Trả lời đúng 1 từ: OK")
    assert response.success is True
    assert response.metrics.status == GatewayStatus.FALLBACK_SUCCESS
    assert response.metrics.is_fallback is True
    assert response.metrics.model_executed == "gemini-2.5-flash"
    assert response.metrics.fallback_reason is not None
    assert response.content is not None

