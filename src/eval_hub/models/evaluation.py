"""Evaluation data models and schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


def get_utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class RiskCategory(str, Enum):
    """Risk categories for evaluations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BackendType(str, Enum):
    """Supported evaluation backends."""

    LMEVAL = "lm-evaluation-harness"
    GUIDELLM = "guidellm"
    NEMO_EVALUATOR = "nemo-evaluator"
    CUSTOM = "custom"


class EvaluationStatus(str, Enum):
    """Status of an evaluation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkSpec(BaseModel):
    """Specification for a benchmark within an evaluation."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Benchmark name")
    tasks: List[str] = Field(..., description="List of tasks to evaluate")
    num_fewshot: Optional[int] = Field(None, description="Number of few-shot examples")
    batch_size: Optional[int] = Field(None, description="Batch size for evaluation")
    limit: Optional[int] = Field(None, description="Limit number of examples")
    device: Optional[str] = Field(None, description="Device to run evaluation on")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class BackendSpec(BaseModel):
    """Specification for an evaluation backend."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Backend name/identifier")
    type: BackendType = Field(..., description="Backend type")
    endpoint: Optional[str] = Field(None, description="Backend API endpoint")
    config: Dict[str, Any] = Field(default_factory=dict, description="Backend-specific configuration")
    benchmarks: List[BenchmarkSpec] = Field(..., description="Benchmarks to run on this backend")


class EvaluationSpec(BaseModel):
    """Specification for a single evaluation request."""

    model_config = ConfigDict(extra="allow")

    id: UUID = Field(default_factory=uuid4, description="Unique evaluation ID")
    name: str = Field(..., description="Human-readable evaluation name")
    description: Optional[str] = Field(None, description="Evaluation description")
    model_server_id: str = Field(..., description="Model server identifier")
    model_name: str = Field(..., description="Name of the model on the server")
    model_configuration: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    backends: List[BackendSpec] = Field(..., description="Backends to run evaluations on")
    risk_category: Optional[RiskCategory] = Field(None, description="Risk category for automatic benchmark selection")
    priority: int = Field(default=0, description="Evaluation priority (higher = more urgent)")
    timeout_minutes: int = Field(default=60, description="Timeout for the entire evaluation")
    retry_attempts: int = Field(default=3, description="Number of retry attempts on failure")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SingleBenchmarkEvaluationRequest(BaseModel):
    """Simplified request for running a single benchmark evaluation."""

    model_config = ConfigDict(extra="allow")

    model_server_id: str = Field(..., description="Model server identifier")
    model_name: str = Field(..., description="Name of the model on the server")
    model_configuration: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    timeout_minutes: int = Field(default=60, description="Timeout for the evaluation")
    retry_attempts: int = Field(default=3, description="Number of retry attempts on failure")
    limit: Optional[int] = Field(None, description="Limit number of samples to evaluate")
    num_fewshot: Optional[int] = Field(None, description="Number of few-shot examples")
    experiment_name: Optional[str] = Field(None, description="MLFlow experiment name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the evaluation run")


class EvaluationRequest(BaseModel):
    """Request payload for starting evaluations."""

    model_config = ConfigDict(extra="allow")

    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    evaluations: List[EvaluationSpec] = Field(..., description="List of evaluations to run")
    experiment_name: Optional[str] = Field(None, description="MLFlow experiment name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the evaluation run")
    async_mode: bool = Field(default=True, description="Whether to run evaluations asynchronously")
    callback_url: Optional[str] = Field(None, description="URL to call when evaluation completes")
    created_at: datetime = Field(default_factory=get_utc_now, description="Request creation timestamp")


class EvaluationResult(BaseModel):
    """Result of a single evaluation."""

    model_config = ConfigDict(extra="allow")

    evaluation_id: UUID = Field(..., description="Evaluation ID")
    backend_name: str = Field(..., description="Backend that ran the evaluation")
    benchmark_name: str = Field(..., description="Benchmark name")
    status: EvaluationStatus = Field(..., description="Evaluation status")
    metrics: Dict[str, Union[float, int, str]] = Field(default_factory=dict, description="Evaluation metrics")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Paths to result artifacts")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Evaluation duration")
    mlflow_run_id: Optional[str] = Field(None, description="MLFlow run ID")


class EvaluationResponse(BaseModel):
    """Response payload for evaluation requests."""

    model_config = ConfigDict(extra="allow")

    request_id: UUID = Field(..., description="Original request ID")
    status: EvaluationStatus = Field(..., description="Overall request status")
    total_evaluations: int = Field(..., description="Total number of evaluations")
    completed_evaluations: int = Field(default=0, description="Number of completed evaluations")
    failed_evaluations: int = Field(default=0, description="Number of failed evaluations")
    results: List[EvaluationResult] = Field(default_factory=list, description="Evaluation results")
    aggregated_metrics: Dict[str, Union[float, int, str]] = Field(
        default_factory=dict, description="Aggregated metrics across all evaluations"
    )
    experiment_url: Optional[str] = Field(None, description="MLFlow experiment URL")
    created_at: datetime = Field(..., description="Request creation timestamp")
    updated_at: datetime = Field(default_factory=get_utc_now, description="Last update timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    progress_percentage: float = Field(default=0.0, description="Overall progress percentage")