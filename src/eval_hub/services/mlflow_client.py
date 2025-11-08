"""MLFlow client service for experiment tracking and results storage."""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

import mlflow
import mlflow.tracking
from mlflow.entities import Run, Experiment
from mlflow.exceptions import MlflowException

from ..core.config import Settings
from ..core.exceptions import MLFlowError
from ..core.logging import get_logger
from ..models.evaluation import EvaluationRequest, EvaluationResult, EvaluationSpec


class MLFlowClient:
    """Client for interacting with MLFlow tracking server."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLFlow configuration."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        # Set S3 credentials if configured
        if self.settings.s3_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.settings.s3_access_key_id
        if self.settings.s3_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.s3_secret_access_key
        if self.settings.s3_endpoint_url:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.settings.s3_endpoint_url

        self.logger.info(
            "MLFlow client configured",
            tracking_uri=self.settings.mlflow_tracking_uri
        )

    async def create_experiment(self, request: EvaluationRequest) -> str:
        """Create or get an MLFlow experiment for the evaluation request."""
        experiment_name = self._generate_experiment_name(request)

        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                self.logger.info(
                    "Using existing MLFlow experiment",
                    experiment_name=experiment_name,
                    experiment_id=experiment_id
                )
            else:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=self.settings.mlflow_artifact_location,
                    tags={
                        "request_id": str(request.request_id),
                        "created_at": request.created_at.isoformat(),
                        "evaluation_count": str(len(request.evaluations)),
                        "service_version": self.settings.version,
                    }
                )
                self.logger.info(
                    "Created new MLFlow experiment",
                    experiment_name=experiment_name,
                    experiment_id=experiment_id
                )

            return experiment_id

        except MlflowException as e:
            raise MLFlowError(f"Failed to create/get MLFlow experiment: {e}")

    async def start_evaluation_run(
        self,
        experiment_id: str,
        evaluation: EvaluationSpec,
        backend_name: str,
        benchmark_name: str
    ) -> str:
        """Start an MLFlow run for a specific evaluation."""
        run_name = f"{evaluation.model_server_id}::{evaluation.model_name}_{backend_name}_{benchmark_name}"

        try:
            # Start MLFlow run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags={
                    "evaluation_id": str(evaluation.id),
                    "model_server_id": evaluation.model_server_id,
                    "model_name": evaluation.model_name,
                    "backend_name": backend_name,
                    "benchmark_name": benchmark_name,
                    "risk_category": evaluation.risk_category.value if evaluation.risk_category else None,
                    "priority": str(evaluation.priority),
                    "started_at": datetime.utcnow().isoformat(),
                }
            )

            # Log evaluation parameters
            await self._log_evaluation_parameters(evaluation, backend_name, benchmark_name)

            self.logger.info(
                "Started MLFlow run",
                run_id=run.info.run_id,
                run_name=run_name,
                evaluation_id=str(evaluation.id)
            )

            return run.info.run_id

        except MlflowException as e:
            raise MLFlowError(f"Failed to start MLFlow run: {e}")

    async def _log_evaluation_parameters(
        self,
        evaluation: EvaluationSpec,
        backend_name: str,
        benchmark_name: str
    ) -> None:
        """Log evaluation parameters to the current MLFlow run."""
        try:
            # Log basic parameters
            mlflow.log_param("model_server_id", evaluation.model_server_id)
            mlflow.log_param("model_name", evaluation.model_name)
            mlflow.log_param("backend_name", backend_name)
            mlflow.log_param("benchmark_name", benchmark_name)
            mlflow.log_param("timeout_minutes", evaluation.timeout_minutes)
            mlflow.log_param("retry_attempts", evaluation.retry_attempts)

            if evaluation.risk_category:
                mlflow.log_param("risk_category", evaluation.risk_category.value)

            # Log model configuration
            for key, value in evaluation.model_configuration.items():
                mlflow.log_param(f"model_config_{key}", str(value))

            # Log backend-specific configuration
            backend_spec = next(
                (b for b in evaluation.backends if b.name == backend_name), None
            )
            if backend_spec:
                for key, value in backend_spec.config.items():
                    mlflow.log_param(f"backend_config_{key}", str(value))

                # Log benchmark-specific configuration
                benchmark_spec = next(
                    (b for b in backend_spec.benchmarks if b.name == benchmark_name), None
                )
                if benchmark_spec:
                    if benchmark_spec.num_fewshot is not None:
                        mlflow.log_param("num_fewshot", benchmark_spec.num_fewshot)
                    if benchmark_spec.batch_size is not None:
                        mlflow.log_param("batch_size", benchmark_spec.batch_size)
                    if benchmark_spec.limit is not None:
                        mlflow.log_param("limit", benchmark_spec.limit)
                    if benchmark_spec.device:
                        mlflow.log_param("device", benchmark_spec.device)

                    # Log benchmark configuration
                    for key, value in benchmark_spec.config.items():
                        mlflow.log_param(f"benchmark_config_{key}", str(value))

            # Log metadata
            for key, value in evaluation.metadata.items():
                mlflow.log_param(f"metadata_{key}", str(value))

        except MlflowException as e:
            self.logger.warning(
                "Failed to log evaluation parameters to MLFlow",
                error=str(e)
            )

    async def log_evaluation_result(self, result: EvaluationResult) -> None:
        """Log evaluation result to MLFlow."""
        if not result.mlflow_run_id:
            self.logger.warning(
                "No MLFlow run ID found for result",
                evaluation_id=str(result.evaluation_id)
            )
            return

        try:
            # Set the active run
            with mlflow.start_run(run_id=result.mlflow_run_id):
                # Log metrics
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                    else:
                        mlflow.log_param(f"metric_{metric_name}", str(metric_value))

                # Log execution metadata
                mlflow.log_param("status", result.status.value)
                if result.duration_seconds:
                    mlflow.log_metric("duration_seconds", result.duration_seconds)
                if result.started_at:
                    mlflow.log_param("started_at", result.started_at.isoformat())
                if result.completed_at:
                    mlflow.log_param("completed_at", result.completed_at.isoformat())

                # Log error message if failed
                if result.error_message:
                    mlflow.log_param("error_message", result.error_message)

                # Log artifacts
                for artifact_name, artifact_path in result.artifacts.items():
                    try:
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path, artifact_name)
                        else:
                            # Create a simple text file with the artifact path
                            temp_file = f"/tmp/{artifact_name}.txt"
                            with open(temp_file, "w") as f:
                                f.write(f"Artifact path: {artifact_path}")
                            mlflow.log_artifact(temp_file, artifact_name)
                            os.remove(temp_file)
                    except Exception as e:
                        self.logger.warning(
                            "Failed to log artifact",
                            artifact_name=artifact_name,
                            artifact_path=artifact_path,
                            error=str(e)
                        )

                # End the run
                mlflow.end_run(status="FINISHED" if result.status.value == "completed" else "FAILED")

                self.logger.info(
                    "Logged evaluation result to MLFlow",
                    run_id=result.mlflow_run_id,
                    evaluation_id=str(result.evaluation_id),
                    status=result.status.value
                )

        except MlflowException as e:
            self.logger.error(
                "Failed to log evaluation result to MLFlow",
                run_id=result.mlflow_run_id,
                error=str(e)
            )

    async def get_experiment_url(self, experiment_id: str) -> str:
        """Get the URL for viewing an experiment in the MLFlow UI."""
        base_url = self.settings.mlflow_tracking_uri.rstrip("/")
        return f"{base_url}/#/experiments/{experiment_id}"

    async def get_run_url(self, run_id: str) -> str:
        """Get the URL for viewing a run in the MLFlow UI."""
        base_url = self.settings.mlflow_tracking_uri.rstrip("/")
        return f"{base_url}/#/experiments/0/runs/{run_id}"

    async def search_runs(
        self,
        experiment_id: str,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Run]:
        """Search for runs in an experiment."""
        try:
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                output_format="list"
            )
            return runs
        except MlflowException as e:
            raise MLFlowError(f"Failed to search runs: {e}")

    async def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics for a specific run."""
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            return run.data.metrics
        except MlflowException as e:
            raise MLFlowError(f"Failed to get run metrics: {e}")

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment."""
        try:
            mlflow.delete_experiment(experiment_id)
            self.logger.info(
                "Deleted MLFlow experiment",
                experiment_id=experiment_id
            )
        except MlflowException as e:
            raise MLFlowError(f"Failed to delete experiment: {e}")

    def _generate_experiment_name(self, request: EvaluationRequest) -> str:
        """Generate a unique experiment name for the request."""
        if request.experiment_name:
            base_name = request.experiment_name
        else:
            # Generate name based on timestamp and request content
            timestamp = request.created_at.strftime("%Y%m%d_%H%M%S")
            model_identifiers = list(set(f"{eval.model_server_id}::{eval.model_name}" for eval in request.evaluations))
            if len(model_identifiers) == 1:
                base_name = f"{model_identifiers[0]}_{timestamp}"
            else:
                base_name = f"multi_model_{timestamp}"

        # Add prefix
        return f"{self.settings.mlflow_experiment_prefix}_{base_name}"

    async def aggregate_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Aggregate metrics across all runs in an experiment."""
        try:
            runs = await self.search_runs(experiment_id)

            if not runs:
                return {}

            # Aggregate metrics
            all_metrics = {}
            metric_counts = {}

            for run in runs:
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            # Calculate aggregated statistics
            aggregated = {}
            for metric_name, values in all_metrics.items():
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)
                aggregated[f"{metric_name}_count"] = len(values)

            return aggregated

        except MlflowException as e:
            self.logger.error(
                "Failed to aggregate experiment metrics",
                experiment_id=experiment_id,
                error=str(e)
            )
            return {}