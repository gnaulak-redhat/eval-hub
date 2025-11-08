"""Base executor class for evaluation backends."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from ..models.evaluation import EvaluationResult, BackendSpec, BenchmarkSpec


class ExecutionContext:
    """Context information for evaluation execution."""

    def __init__(
        self,
        evaluation_id: UUID,
        model_server_id: str,
        model_name: str,
        backend_spec: BackendSpec,
        benchmark_spec: BenchmarkSpec,
        timeout_minutes: int,
        retry_attempts: int,
        started_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_server_base_url: Optional[str] = None,
    ):
        self.evaluation_id = evaluation_id
        self.model_server_id = model_server_id
        self.model_name = model_name
        self.model_server_base_url = model_server_base_url
        self.backend_spec = backend_spec
        self.benchmark_spec = benchmark_spec
        self.timeout_minutes = timeout_minutes
        self.retry_attempts = retry_attempts
        self.started_at = started_at or datetime.utcnow()
        self.metadata = metadata or {}


class Executor(ABC):
    """Abstract base class for evaluation backend executors."""

    def __init__(self, backend_config: Dict[str, Any]):
        """Initialize the executor with backend-specific configuration.

        Args:
            backend_config: Configuration dictionary for this backend
        """
        self.backend_config = backend_config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the backend configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and available.

        Returns:
            bool: True if backend is healthy, False otherwise
        """
        pass

    @abstractmethod
    async def execute_benchmark(
        self,
        context: ExecutionContext,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> EvaluationResult:
        """Execute a benchmark evaluation.

        Args:
            context: Execution context with evaluation parameters
            progress_callback: Optional callback for progress updates

        Returns:
            EvaluationResult: Result of the evaluation

        Raises:
            BackendError: If execution fails
            TimeoutError: If execution times out
        """
        pass

    @classmethod
    @abstractmethod
    def get_backend_type(cls) -> str:
        """Get the backend type identifier.

        Returns:
            str: Backend type (e.g., "nemo-evaluator", "lm-evaluation-harness")
        """
        pass

    @classmethod
    def validate_backend_config(cls, config: Dict[str, Any]) -> bool:
        """Validate backend configuration without instantiating.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Create temporary instance to validate
            temp_executor = cls(config)
            return True
        except Exception:
            return False

    def get_display_name(self) -> str:
        """Get a human-readable name for this executor.

        Returns:
            str: Display name
        """
        return f"{self.get_backend_type()} Executor"

    async def cleanup(self) -> None:
        """Perform any cleanup after execution.

        This method is called after evaluation completion (success or failure)
        and can be used to clean up resources, send notifications, etc.
        """
        pass

    def supports_parallel_execution(self) -> bool:
        """Check if this executor supports parallel benchmark execution.

        Returns:
            bool: True if parallel execution is supported
        """
        return True

    def get_recommended_timeout_minutes(self) -> int:
        """Get the recommended timeout for this executor.

        Returns:
            int: Recommended timeout in minutes
        """
        return 60  # Default 1 hour

    def get_max_retry_attempts(self) -> int:
        """Get the maximum recommended retry attempts.

        Returns:
            int: Maximum retry attempts
        """
        return 3