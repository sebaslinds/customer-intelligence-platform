from typing import Literal

from pydantic import BaseModel, Field

InsightCategory = Literal["churn", "reorder_trend", "top_products", "customer_behavior", "retention", "clv", "general"]
ImpactLevel = Literal["low", "medium", "high"]
AIResponseSource = Literal["openai", "local_fallback"]


class CopilotRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Insight(BaseModel):
    title: str
    category: InsightCategory
    finding: str
    evidence: list[str]
    impact: ImpactLevel
    recommended_action: str


class ImpactedSegment(BaseModel):
    segment_name: str
    metric: str
    value: str
    why_it_matters: str


class Recommendation(BaseModel):
    action: str
    expected_impact: str
    priority: ImpactLevel


class CopilotResponse(BaseModel):
    question: str
    ai_source: AIResponseSource = "local_fallback"
    ai_source_detail: str | None = None
    summary: str
    explanation: str
    impacted_segments: list[ImpactedSegment]
    recommendations: list[Recommendation]
    insights: list[Insight]
    follow_up_questions: list[str]
    data_sources: list[str]
