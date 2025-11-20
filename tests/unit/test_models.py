"""Unit tests for data models."""

from datetime import datetime
from uuid import uuid4

import pytest
from eval_hub.models.evaluation import (
    BackendSpec,
    BackendType,
    BenchmarkSpec,
    EvaluationRequest,
    EvaluationResult,
    EvaluationSpec,
    EvaluationStatus,
    RiskCategory,
)


class TestEvaluationModels:
    """Test evaluation data models."""

    def test_benchmark_spec_creation(self):
        """Test BenchmarkSpec model creation."""
        benchmark = BenchmarkSpec(
            name="hellaswag",
            tasks=["hellaswag"],
            num_fewshot=5,
            batch_size=8,
            limit=1000,
            device="cuda",
            config={"custom_param": "value"},
        )

        assert benchmark.name == "hellaswag"
        assert benchmark.tasks == ["hellaswag"]
        assert benchmark.num_fewshot == 5
        assert benchmark.batch_size == 8
        assert benchmark.limit == 1000
        assert benchmark.device == "cuda"
        assert benchmark.config == {"custom_param": "value"}

    def test_backend_spec_creation(self):
        """Test BackendSpec model creation."""
        benchmark = BenchmarkSpec(name="arc_easy", tasks=["arc_easy"], num_fewshot=5)

        backend = BackendSpec(
            name="lm-evaluation-harness",
            type=BackendType.LMEVAL,
            endpoint="http://localhost:8080",
            config={"timeout": 3600},
            benchmarks=[benchmark],
        )

        assert backend.name == "lm-evaluation-harness"
        assert backend.type == BackendType.LMEVAL
        assert backend.endpoint == "http://localhost:8080"
        assert backend.config == {"timeout": 3600}
        assert len(backend.benchmarks) == 1
        assert backend.benchmarks[0].name == "arc_easy"

    def test_evaluation_spec_creation(self):
        """Test EvaluationSpec model creation."""
        benchmark = BenchmarkSpec(name="test", tasks=["test"])
        backend = BackendSpec(
            name="test-backend", type=BackendType.CUSTOM, benchmarks=[benchmark]
        )

        eval_spec = EvaluationSpec(
            name="Test Evaluation",
            model={"url": "http://test-server:8000", "name": "test-model"},
            backends=[backend],
            risk_category=RiskCategory.MEDIUM,
            priority=1,
            timeout_minutes=30,
            retry_attempts=2,
        )

        assert eval_spec.name == "Test Evaluation"
        assert eval_spec.model_name == "test-model"
        assert eval_spec.model_url == "http://test-server:8000"
        assert len(eval_spec.backends) == 1
        assert eval_spec.risk_category == RiskCategory.MEDIUM
        assert eval_spec.priority == 1
        assert eval_spec.timeout_minutes == 30
        assert eval_spec.retry_attempts == 2
        assert isinstance(eval_spec.id, type(uuid4()))

    def test_evaluation_request_creation(self):
        """Test EvaluationRequest model creation."""
        benchmark = BenchmarkSpec(name="test", tasks=["test"])
        backend = BackendSpec(
            name="test-backend", type=BackendType.CUSTOM, benchmarks=[benchmark]
        )
        eval_spec = EvaluationSpec(
            name="Test Evaluation",
            model={"url": "http://test-server:8000", "name": "test-model"},
            backends=[backend],
        )

        request = EvaluationRequest(
            evaluations=[eval_spec],
            experiment_name="test-experiment",
            tags={"team": "ai", "project": "eval"},
            async_mode=True,
        )

        assert len(request.evaluations) == 1
        assert request.experiment_name == "test-experiment"
        assert request.tags == {"team": "ai", "project": "eval"}
        assert request.async_mode is True
        assert isinstance(request.request_id, type(uuid4()))
        assert isinstance(request.created_at, datetime)

    def test_evaluation_result_creation(self):
        """Test EvaluationResult model creation."""
        eval_id = uuid4()

        result = EvaluationResult(
            evaluation_id=eval_id,
            backend_name="test-backend",
            benchmark_name="test-benchmark",
            status=EvaluationStatus.COMPLETED,
            metrics={"accuracy": 0.85, "f1_score": 0.78},
            artifacts={"results": "/path/to/results.json"},
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=120.5,
            mlflow_run_id="test-run-123",
        )

        assert result.evaluation_id == eval_id
        assert result.backend_name == "test-backend"
        assert result.benchmark_name == "test-benchmark"
        assert result.status == EvaluationStatus.COMPLETED
        assert result.metrics["accuracy"] == 0.85
        assert result.metrics["f1_score"] == 0.78
        assert result.artifacts["results"] == "/path/to/results.json"
        assert result.duration_seconds == 120.5
        assert result.mlflow_run_id == "test-run-123"

    def test_risk_category_values(self):
        """Test RiskCategory enum values."""
        assert RiskCategory.LOW.value == "low"
        assert RiskCategory.MEDIUM.value == "medium"
        assert RiskCategory.HIGH.value == "high"
        assert RiskCategory.CRITICAL.value == "critical"

    def test_backend_type_values(self):
        """Test BackendType enum values."""
        assert BackendType.LMEVAL.value == "lm-evaluation-harness"
        assert BackendType.GUIDELLM.value == "guidellm"
        assert BackendType.CUSTOM.value == "custom"

    def test_evaluation_status_values(self):
        """Test EvaluationStatus enum values."""
        assert EvaluationStatus.PENDING.value == "pending"
        assert EvaluationStatus.RUNNING.value == "running"
        assert EvaluationStatus.COMPLETED.value == "completed"
        assert EvaluationStatus.FAILED.value == "failed"
        assert EvaluationStatus.CANCELLED.value == "cancelled"

    def test_model_validation(self):
        """Test model validation."""
        # Test required fields
        with pytest.raises(ValueError):
            BenchmarkSpec(tasks=["test"])  # Missing name

        with pytest.raises(ValueError):
            BenchmarkSpec(name="test")  # Missing tasks

        # Test empty tasks list is allowed
        benchmark = BenchmarkSpec(name="test", tasks=[])
        assert benchmark.name == "test"
        assert benchmark.tasks == []

        # Test negative values are allowed (no validation currently)
        benchmark_negative = BenchmarkSpec(name="test", tasks=["test"], num_fewshot=-1)
        assert benchmark_negative.num_fewshot == -1

        benchmark_zero = BenchmarkSpec(name="test", tasks=["test"], batch_size=0)
        assert benchmark_zero.batch_size == 0

    def test_model_defaults(self):
        """Test model default values."""
        benchmark = BenchmarkSpec(name="test", tasks=["test"])
        assert benchmark.num_fewshot is None
        assert benchmark.batch_size is None
        assert benchmark.limit is None
        assert benchmark.device is None
        assert benchmark.config == {}

        eval_spec = EvaluationSpec(
            name="Test",
            model={"url": "http://test-server:8000", "name": "test-model"},
            backends=[],
        )
        assert eval_spec.risk_category is None
        assert eval_spec.priority == 0
        assert eval_spec.timeout_minutes == 60
        assert eval_spec.retry_attempts == 3
        assert eval_spec.metadata == {}

    def test_model_extra_fields(self):
        """Test that models allow extra fields."""
        benchmark = BenchmarkSpec(
            name="test",
            tasks=["test"],
            extra_field="extra_value",  # This should be allowed
        )
        assert hasattr(benchmark, "extra_field")
        assert benchmark.extra_field == "extra_value"
