"""MLFlow client service for experiment tracking and results storage."""

from typing import Any
from uuid import uuid4

from ..core.config import Settings
from ..core.logging import get_logger
from ..models.evaluation import EvaluationRequest, EvaluationResult, EvaluationSpec
from ..utils import utcnow


class MLFlowClient:
    """Client for interacting with MLFlow tracking server (mocked for now)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        self._mock_experiments: dict[str, dict[str, Any]] = {}
        self._mock_runs: dict[str, dict[str, Any]] = {}
        self.logger.info(
            "MLFlow client configured (mock mode)",
            tracking_uri=self.settings.mlflow_tracking_uri,
        )

    def _setup_mlflow(self) -> None:
        """Set up MLFlow configuration (no-op in mock mode)."""
        pass

    async def create_experiment(self, request: EvaluationRequest) -> str:
        """Create or get an MLFlow experiment for the evaluation request (mocked)."""
        experiment_name = self._generate_experiment_name(request)

        # Check if experiment already exists in mock storage
        for exp_id, exp_data in self._mock_experiments.items():
            if exp_data["name"] == experiment_name:
                self.logger.info(
                    "Using existing MLFlow experiment (mock)",
                    experiment_name=experiment_name,
                    experiment_id=exp_id,
                )
                return exp_id

        # Create new mock experiment
        experiment_id = f"exp_{uuid4().hex[:8]}"
        self._mock_experiments[experiment_id] = {
            "name": experiment_name,
            "request_id": str(request.request_id),
            "created_at": request.created_at.isoformat(),
            "evaluation_count": len(request.evaluations),
            "service_version": self.settings.version,
        }

        self.logger.info(
            "Created new MLFlow experiment (mock)",
            experiment_name=experiment_name,
            experiment_id=experiment_id,
        )

        return experiment_id

    async def start_evaluation_run(
        self,
        experiment_id: str,
        evaluation: EvaluationSpec,
        backend_name: str,
        benchmark_name: str,
    ) -> str:
        """Start an MLFlow run for a specific evaluation (mocked)."""
        run_name = f"{evaluation.model_url}::{evaluation.model_name}_{backend_name}_{benchmark_name}"

        # Create mock run
        run_id = f"run_{uuid4().hex[:8]}"
        self._mock_runs[run_id] = {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "evaluation_id": str(evaluation.id),
            "model_server_id": evaluation.model_url,
            "model_name": evaluation.model_name,
            "backend_name": backend_name,
            "benchmark_name": benchmark_name,
            "risk_category": (
                evaluation.risk_category.value if evaluation.risk_category else None
            ),
            "priority": str(evaluation.priority),
            "started_at": utcnow().isoformat(),
        }

        self.logger.info(
            "Started MLFlow run (mock)",
            run_id=run_id,
            run_name=run_name,
            evaluation_id=str(evaluation.id),
        )

        return run_id

    async def _log_evaluation_parameters(
        self, evaluation: EvaluationSpec, backend_name: str, benchmark_name: str
    ) -> None:
        """Log evaluation parameters to the current MLFlow run (mocked)."""
        # Mock implementation - parameters are stored in run data
        self.logger.debug(
            "Logging evaluation parameters (mock)",
            evaluation_id=str(evaluation.id),
            backend_name=backend_name,
            benchmark_name=benchmark_name,
        )

    async def log_evaluation_result(self, result: EvaluationResult) -> None:
        """Log evaluation result to MLFlow (mocked)."""
        if not result.mlflow_run_id:
            self.logger.warning(
                "No MLFlow run ID found for result",
                evaluation_id=str(result.evaluation_id),
            )
            return

        # Update mock run with result data
        if result.mlflow_run_id in self._mock_runs:
            self._mock_runs[result.mlflow_run_id].update(
                {
                    "status": result.status.value,
                    "metrics": result.metrics,
                    "artifacts": result.artifacts,
                    "duration_seconds": result.duration_seconds,
                    "started_at": (
                        result.started_at.isoformat() if result.started_at else None
                    ),
                    "completed_at": (
                        result.completed_at.isoformat() if result.completed_at else None
                    ),
                    "error_message": result.error_message,
                }
            )

        self.logger.info(
            "Logged evaluation result to MLFlow (mock)",
            run_id=result.mlflow_run_id,
            evaluation_id=str(result.evaluation_id),
            status=result.status.value,
        )

    async def get_experiment_url(self, experiment_id: str) -> str:
        """Get the URL for viewing an experiment in the MLFlow UI (mocked)."""
        base_url = self.settings.mlflow_tracking_uri.rstrip("/")
        return f"{base_url}/#/experiments/{experiment_id}"

    async def get_run_url(self, run_id: str) -> str:
        """Get the URL for viewing a run in the MLFlow UI (mocked)."""
        base_url = self.settings.mlflow_tracking_uri.rstrip("/")
        return f"{base_url}/#/experiments/0/runs/{run_id}"

    async def search_runs(
        self,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for runs in an experiment (mocked)."""
        # Return mock runs for the experiment
        runs = [
            run_data
            for run_id, run_data in self._mock_runs.items()
            if run_data.get("experiment_id") == experiment_id
        ]
        return runs[:max_results]

    async def get_run_metrics(self, run_id: str) -> dict[str, float]:
        """Get metrics for a specific run (mocked)."""
        if run_id in self._mock_runs:
            metrics = self._mock_runs[run_id].get("metrics", {})
            # Filter to only numeric metrics
            return {
                k: float(v) for k, v in metrics.items() if isinstance(v, int | float)
            }
        return {}

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment (mocked)."""
        if experiment_id in self._mock_experiments:
            del self._mock_experiments[experiment_id]
            # Also delete associated runs
            run_ids_to_delete = [
                run_id
                for run_id, run_data in self._mock_runs.items()
                if run_data.get("experiment_id") == experiment_id
            ]
            for run_id in run_ids_to_delete:
                del self._mock_runs[run_id]
            self.logger.info(
                "Deleted MLFlow experiment (mock)", experiment_id=experiment_id
            )

    def _generate_experiment_name(self, request: EvaluationRequest) -> str:
        """Generate a unique experiment name for the request."""
        if request.experiment_name:
            base_name = request.experiment_name
        else:
            # Generate name based on timestamp and request content
            timestamp = request.created_at.strftime("%Y%m%d_%H%M%S")
            model_identifiers = list(
                {f"{eval.model_url}::{eval.model_name}" for eval in request.evaluations}
            )
            if len(model_identifiers) == 1:
                base_name = f"{model_identifiers[0]}_{timestamp}"
            else:
                base_name = f"multi_model_{timestamp}"

        # Add prefix
        return f"{self.settings.mlflow_experiment_prefix}_{base_name}"

    async def aggregate_experiment_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Aggregate metrics across all runs in an experiment (mocked)."""
        runs = await self.search_runs(experiment_id)

        if not runs:
            return {}

        # Aggregate metrics
        all_metrics: dict[str, list[float]] = {}

        for run in runs:
            metrics = run.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, int | float):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(float(metric_value))

        # Calculate aggregated statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)
                aggregated[f"{metric_name}_count"] = len(values)

        return aggregated
