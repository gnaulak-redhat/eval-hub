"""Unit tests for single benchmark evaluation API endpoint."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from eval_hub.api.app import create_app
from eval_hub.models.provider import (
    Benchmark,
    BenchmarkDetail,
    Provider,
    ProviderType,
)
from eval_hub.services.provider_service import ProviderService
from fastapi.testclient import TestClient


@pytest.fixture
def mock_provider_service():
    """Create a mock provider service for testing."""
    service = Mock(spec=ProviderService)

    # Create mock benchmark
    benchmark = BenchmarkDetail(
        benchmark_id="blimp",
        provider_id="lm_evaluation_harness",
        provider_name="LM Evaluation Harness",
        name="Blimp",
        description="Blimp evaluation benchmark",
        category="general",
        metrics=["accuracy"],
        num_few_shot=0,
        dataset_size=1000,
        tags=["general", "lm_eval"],
        provider_type=ProviderType.BUILTIN,
        base_url=None,
    )

    # Create mock provider
    provider = Provider(
        provider_id="lm_evaluation_harness",
        provider_name="LM Evaluation Harness",
        description="Comprehensive evaluation framework",
        provider_type=ProviderType.BUILTIN,
        base_url=None,
        benchmarks=[
            Benchmark(
                benchmark_id="blimp",
                name="Blimp",
                description="Blimp evaluation benchmark",
                category="general",
                metrics=["accuracy"],
                num_few_shot=0,
                dataset_size=1000,
                tags=["general", "lm_eval"],
            )
        ],
    )

    service.get_benchmark_by_id.return_value = benchmark
    service.get_provider_by_id.return_value = provider

    return service


@pytest.fixture
def client_with_mock_provider(mock_provider_service):
    """Create test client with mocked provider service."""
    from eval_hub.api.routes import (
        get_evaluation_executor,
        get_mlflow_client,
        get_provider_service,
        get_request_parser,
        get_response_builder,
    )

    app = create_app()

    # Mock all dependencies
    def override_provider_service():
        return mock_provider_service

    async def override_parser():
        from unittest.mock import AsyncMock

        parser = Mock()
        parser.parse_evaluation_request = AsyncMock(return_value=Mock())
        return parser

    async def override_executor():
        from unittest.mock import AsyncMock

        executor = Mock()
        executor.execute_evaluation_request = AsyncMock(return_value=[])
        return executor

    async def override_mlflow_client():
        from unittest.mock import AsyncMock

        client = Mock()
        client.create_experiment = AsyncMock(return_value="exp-123")
        client.get_experiment_url = AsyncMock(return_value="http://mlflow:5000/exp/123")
        return client

    async def override_response_builder():
        from datetime import datetime
        from unittest.mock import AsyncMock

        builder = Mock()
        builder.build_response = AsyncMock(
            return_value=Mock(
                request_id=uuid4(),
                status="pending",
                total_evaluations=1,
                completed_evaluations=0,
                failed_evaluations=0,
                results=[],
                aggregated_metrics={},
                experiment_url="http://mlflow:5000/exp/123",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                estimated_completion=None,
                progress_percentage=0.0,
            )
        )
        return builder

    app.dependency_overrides[get_provider_service] = override_provider_service
    app.dependency_overrides[get_request_parser] = override_parser
    app.dependency_overrides[get_evaluation_executor] = override_executor
    app.dependency_overrides[get_mlflow_client] = override_mlflow_client
    app.dependency_overrides[get_response_builder] = override_response_builder

    with TestClient(app) as client:
        yield client, mock_provider_service

    app.dependency_overrides.clear()


class TestSingleBenchmarkEvaluation:
    """Test cases for single benchmark evaluation endpoint."""

    def test_create_single_benchmark_evaluation_success(
        self, client_with_mock_provider
    ):
        """Test successful creation of single benchmark evaluation."""
        client, mock_service = client_with_mock_provider

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "gpt-4o-mini"},
            "model_configuration": {"temperature": 0.0, "max_tokens": 512},
            "timeout_minutes": 30,
            "retry_attempts": 1,
            "limit": 100,
            "num_fewshot": 0,
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/lm_evaluation_harness/blimp",
            json=request_data,
        )

        assert response.status_code == 202
        data = response.json()
        assert "request_id" in data
        assert data["status"] == "pending"
        assert data["total_evaluations"] == 1

        # Verify service was called correctly
        mock_service.get_benchmark_by_id.assert_called_once_with(
            "lm_evaluation_harness", "blimp"
        )
        mock_service.get_provider_by_id.assert_called_once_with("lm_evaluation_harness")

    def test_create_single_benchmark_evaluation_with_minimal_request(
        self, client_with_mock_provider
    ):
        """Test creation with minimal required fields."""
        client, _ = client_with_mock_provider

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "gpt-4o-mini"},
            "model_configuration": {},
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/lm_evaluation_harness/blimp",
            json=request_data,
        )

        assert response.status_code == 202
        data = response.json()
        assert "request_id" in data

    def test_create_single_benchmark_evaluation_benchmark_not_found(
        self, client_with_mock_provider
    ):
        """Test error when benchmark is not found."""
        client, mock_service = client_with_mock_provider

        mock_service.get_benchmark_by_id.return_value = None

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "gpt-4o-mini"},
            "model_configuration": {},
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/lm_evaluation_harness/nonexistent",
            json=request_data,
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_create_single_benchmark_evaluation_provider_not_found(
        self, client_with_mock_provider
    ):
        """Test error when provider is not found."""
        client, mock_service = client_with_mock_provider

        mock_service.get_provider_by_id.return_value = None

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "gpt-4o-mini"},
            "model_configuration": {},
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/nonexistent_provider/blimp",
            json=request_data,
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_create_single_benchmark_evaluation_with_custom_config(
        self, client_with_mock_provider
    ):
        """Test creation with custom configuration options."""
        client, _ = client_with_mock_provider

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "gpt-4o-mini"},
            "model_configuration": {
                "temperature": 0.1,
                "max_tokens": 256,
                "top_p": 0.95,
            },
            "timeout_minutes": 60,
            "retry_attempts": 2,
            "limit": 50,
            "num_fewshot": 5,
            "experiment_name": "Custom Experiment",
            "tags": {"custom_tag": "value", "test": "true"},
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/lm_evaluation_harness/blimp",
            json=request_data,
        )

        assert response.status_code == 202
        data = response.json()
        assert "request_id" in data

    def test_create_single_benchmark_evaluation_missing_model(
        self, client_with_mock_provider
    ):
        """Test error when model is missing."""
        client, _ = client_with_mock_provider

        request_data = {"model_configuration": {}}

        response = client.post(
            "/api/v1/evaluations/benchmarks/lm_evaluation_harness/blimp",
            json=request_data,
        )

        assert response.status_code == 422  # Validation error

    def test_create_single_benchmark_evaluation_nemo_evaluator_provider(
        self, client_with_mock_provider
    ):
        """Test with nemo-evaluator provider type."""
        client, mock_service = client_with_mock_provider

        # Create nemo-evaluator provider
        nemo_provider = Provider(
            provider_id="nemo_evaluator",
            provider_name="NeMo Evaluator",
            description="NeMo Evaluator provider",
            provider_type=ProviderType.NEMO_EVALUATOR,
            base_url="http://nemo:3825",
            benchmarks=[],
        )

        nemo_benchmark = BenchmarkDetail(
            benchmark_id="test_benchmark",
            provider_id="nemo_evaluator",
            provider_name="NeMo Evaluator",
            name="Test Benchmark",
            description="Test",
            category="test",
            metrics=["accuracy"],
            num_few_shot=0,
            dataset_size=100,
            tags=[],
            provider_type=ProviderType.NEMO_EVALUATOR,
            base_url="http://nemo:3825",
        )

        mock_service.get_benchmark_by_id.return_value = nemo_benchmark
        mock_service.get_provider_by_id.return_value = nemo_provider

        request_data = {
            "model": {"url": "http://vllm-server:8000", "name": "test-model"},
            "model_configuration": {},
        }

        response = client.post(
            "/api/v1/evaluations/benchmarks/nemo_evaluator/test_benchmark",
            json=request_data,
        )

        assert response.status_code == 202
        data = response.json()
        assert "request_id" in data
