from typing import Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------
class CustomerSupportTicket(BaseModel):
    """Schema dữ liệu chuẩn hóa của Ticket chăm sóc khách hàng."""
    customer_name: Optional[str] = Field(description="Họ và tên khách hàng nếu có")
    urgency_level: str = Field(description="Mức độ khẩn cấp: LOW, MEDIUM, HIGH, CRITICAL")
    sentiment: str = Field(description="Cảm xúc khách hàng: POSITIVE, NEUTRAL, FRUSTRATED")
    product_mentioned: Optional[str] = Field(description="Tên sản phẩm/dịch vụ được đề cập")
    action_items: List[str] = Field(description="Danh sách các đầu việc đội ngũ kỹ thuật cần làm")


class ExecutionMetrics(BaseModel):
    """Các chỉ số hiệu năng và chi phí của quá trình gọi model."""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    retries_count: int = 0


class ServiceResponse(BaseModel, Generic[T]):
    """Kết quả trả về chuẩn hóa cho toàn bộ các Extraction Services."""
    data: Optional[T] = None
    metrics: ExecutionMetrics
    success: bool
    error_message: Optional[str] = None

