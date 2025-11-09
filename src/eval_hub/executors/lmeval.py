"""LM Evaluation Harness executor for running evaluations."""

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from ..core.exceptions import BackendError
from ..core.logging import get_logger
from ..models.evaluation import EvaluationResult, EvaluationStatus
from .base import ExecutionContext, Executor


class LMEvalExecutor(Executor):
    """Executor for running evaluations using lm-evaluation-harness."""

    @staticmethod
    def _sanitize_k8s_name(name: str, max_length: int = 63) -> str:
        """Sanitize a string to be a valid Kubernetes resource name.

        Kubernetes resource names must follow RFC 1123 subdomain naming:
        - Lowercase alphanumeric characters, '-' or '.'
        - Must start and end with an alphanumeric character
        - Maximum 63 characters

        Args:
            name: The name to sanitize
            max_length: Maximum length (default: 63 for subdomains)

        Returns:
            A sanitized name that meets Kubernetes naming requirements
        """
        # Convert to lowercase
        sanitized = name.lower()

        # Replace underscores and other invalid characters with hyphens
        sanitized = re.sub(r"[^a-z0-9.-]", "-", sanitized)

        # Remove consecutive hyphens
        sanitized = re.sub(r"-+", "-", sanitized)

        # Remove leading and trailing hyphens and dots
        sanitized = sanitized.strip("-.")

        # Ensure it starts with alphanumeric
        if sanitized and not sanitized[0].isalnum():
            sanitized = "a" + sanitized

        # Ensure it ends with alphanumeric
        if sanitized and not sanitized[-1].isalnum():
            sanitized = sanitized + "0"

        # Truncate to max length if needed
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            # Ensure it still ends with alphanumeric after truncation
            sanitized = sanitized.rstrip("-.")

        # Final check: if empty or doesn't start/end with alphanumeric, add prefix/suffix
        if not sanitized:
            sanitized = "evaljob"
        if not sanitized[0].isalnum():
            sanitized = "a" + sanitized[1:]
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "0"

        return sanitized

    def __init__(self, backend_config: dict[str, Any]):
        self.logger = get_logger(__name__)
        # Set model before super().__init__() to avoid validation error
        # Model will be taken from context during execution if not in config
        self.model = backend_config.get("model", None)
        super().__init__(backend_config)

        # Default configuration
        self.model_args = backend_config.get("model_args", "")
        self.batch_size = backend_config.get("batch_size", "1")
        self.device = backend_config.get("device", "cuda:0")
        self.output_path = backend_config.get("output_path", "/tmp/lmeval_results")
        self.limit = backend_config.get("limit", None)
        self.num_fewshot = backend_config.get("num_fewshot", 0)
        self.lm_eval_path = backend_config.get("lm_eval_path", "lm_eval")
        self.timeout_seconds = backend_config.get("timeout_seconds", 3600)
        self.namespace = backend_config.get("namespace", "test")
        self.log_samples = backend_config.get("log_samples", True)
        self.deploy_crs = backend_config.get("deploy_crs", True)

        # Kubernetes client initialization
        self.k8s_client = None
        if self.deploy_crs:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                try:
                    config.load_kube_config()
                except config.ConfigException:
                    self.logger.warning(
                        "Could not load Kubernetes config. CR deployment will be disabled."
                    )
                    self.deploy_crs = False

            if self.deploy_crs:
                self.k8s_client = client.CustomObjectsApi()

        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate LM Evaluation Harness configuration."""
        # Model can be provided in config or will be taken from context during execution
        pass

    @classmethod
    def get_backend_type(cls) -> str:
        """Get the backend type identifier."""
        return "lm-evaluation-harness"

    async def health_check(self) -> bool:
        """Check if lm-evaluation-harness is available."""
        return True

    async def execute_benchmark(
        self,
        context: ExecutionContext,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> EvaluationResult:
        """Execute a benchmark evaluation using lm-evaluation-harness."""

        # Get model from context if not in config
        model = self.model or context.model_name
        if not model:
            raise BackendError("Model name is required (either in backend config or context)")

        self.logger.info(
            "Starting LM Evaluation Harness execution",
            evaluation_id=str(context.evaluation_id),
            benchmark=context.benchmark_spec.name,
            model=model,
        )

        try:
            # Report progress start
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    0.0,
                    f"Preparing {context.benchmark_spec.name} for LM Evaluation Harness",
                )

            # Build command arguments
            task_name = context.benchmark_spec.name
            tasks = context.benchmark_spec.tasks or [task_name]

            # Construct LMEval job CR
            job_cr = self._build_lmeval_job_cr(context, tasks, model)
            job_cr_yaml = yaml.dump(job_cr, default_flow_style=False, sort_keys=False)
            self.logger.info(
                "LMEval Job YAML CR",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
            )
            self.logger.info(f"LMEval Job YAML CR:\n{job_cr_yaml}")

            # Report progress: deploying CR
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    10.0,
                    f"Deploying LMEvalJob CR for {task_name}",
                )

            # Deploy the CR to Kubernetes
            if self.deploy_crs and self.k8s_client:
                await self._deploy_lmeval_job_cr(job_cr, context)
                self.logger.info(
                    "LMEvalJob CR deployed successfully",
                    evaluation_id=str(context.evaluation_id),
                    benchmark=context.benchmark_spec.name,
                    namespace=self.namespace,
                )

                # Return a result indicating the job was deployed
                # The actual evaluation will be handled by the operator
                return EvaluationResult(
                    evaluation_id=context.evaluation_id,
                    backend_name="lm-evaluation-harness",
                    benchmark_name=context.benchmark_spec.name,
                    status=EvaluationStatus.PENDING,
                    started_at=context.started_at,
                    completed_at=None,
                    duration_seconds=None,
                    metadata={
                        "cr_name": job_cr["metadata"]["name"],
                        "namespace": self.namespace,
                        "deployed": True,
                    },
                )

            # Fallback to local execution if CR deployment is disabled
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    20.0,
                    f"Running evaluation locally for {task_name}",
                )

            # Execute the evaluation locally
            result = await self._run_lm_eval(
                tasks=tasks,
                context=context,
                model=model,
                progress_callback=progress_callback,
            )

            # Report completion
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    100.0,
                    f"Completed {context.benchmark_spec.name} on LM Evaluation Harness",
                )

            # Convert result to eval-hub format
            eval_result = await self._convert_lmeval_result_to_eval_hub(result, context)

            self.logger.info(
                "LM Evaluation Harness execution completed",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
                status=eval_result.status,
            )

            return eval_result

        except Exception as e:
            self.logger.error(
                "LM Evaluation Harness execution failed",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
                error=str(e),
            )

            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                backend_name="lm-evaluation-harness",
                benchmark_name=context.benchmark_spec.name,
                status=EvaluationStatus.FAILED,
                error_message=str(e),
                started_at=context.started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=(
                    datetime.utcnow() - context.started_at
                ).total_seconds(),
            )

    async def cleanup(self) -> None:
        """Perform post-evaluation cleanup."""
        # Cleanup is handled by the subprocess
        pass

    def _build_lmeval_job_cr(
        self, context: ExecutionContext, tasks: list[str], model: str
    ) -> dict[str, Any]:
        """Build an LMEval job Kubernetes Custom Resource matching the user's structure."""
        benchmark_config = context.benchmark_spec.config or {}

        # Determine limit
        limit = benchmark_config.get("limit") or self.limit
        num_fewshot = benchmark_config.get("num_fewshot") or self.num_fewshot

        # Build model args as list of name/value pairs
        model_args_list: list[dict[str, str]] = []

        # Get model args from backend config
        backend_model_args = self.backend_config.get("model_args", {})
        if isinstance(backend_model_args, dict):
            for name, value in backend_model_args.items():
                model_args_list.append({"name": name, "value": str(value)})
        elif isinstance(backend_model_args, str) and backend_model_args:
            # Try to parse as JSON
            try:
                parsed = json.loads(backend_model_args)
                if isinstance(parsed, dict):
                    for name, value in parsed.items():
                        model_args_list.append({"name": name, "value": str(value)})
            except json.JSONDecodeError:
                pass

        # Add default model args if not present
        if not any(arg["name"] == "model" for arg in model_args_list):
            model_args_list.append({"name": "model", "value": model})

        # Get base_url from context or config and always add it
        if not any(arg["name"] == "base_url" for arg in model_args_list):
            # Try to get base_url from context first (from model server)
            base_url = context.model_server_base_url
            self.logger.info(
                "Retrieving base_url for CR",
                evaluation_id=str(context.evaluation_id),
                model_server_id=context.model_server_id,
                model_name=context.model_name,
                context_base_url=context.model_server_base_url,
            )

            if not base_url:
                # Fallback to backend config
                base_url = self.backend_config.get("base_url", "")
                self.logger.info(
                    "Using base_url from backend config",
                    evaluation_id=str(context.evaluation_id),
                    backend_config_base_url=base_url,
                )

            # Construct base_url with /v1/completions suffix if we have a base URL
            if base_url:
                # Remove trailing slash if present
                base_url = base_url.rstrip("/")
                # Add /v1/completions if not already present
                if not base_url.endswith("/v1/completions"):
                    base_url = f"{base_url}/v1/completions"
                self.logger.info(
                    "Constructed base_url with /v1/completions suffix",
                    evaluation_id=str(context.evaluation_id),
                    final_base_url=base_url,
                )
            else:
                # Log warning if we don't have a base_url
                self.logger.warning(
                    "No base_url found in context or config, base_url will be empty in CR",
                    evaluation_id=str(context.evaluation_id),
                    model_server_id=context.model_server_id,
                    model_name=context.model_name,
                    model_server_base_url=context.model_server_base_url,
                    backend_config_base_url=self.backend_config.get("base_url"),
                )
                # Use empty string as fallback (operator may set it)
                base_url = ""

            # Always add base_url (even if empty, the operator might set it)
            # Ensure value is always a string, never None
            model_args_list.append({"name": "base_url", "value": str(base_url) if base_url else ""})

            self.logger.info(
                "Added base_url to modelArgs",
                evaluation_id=str(context.evaluation_id),
                base_url=base_url,
                base_url_in_model_args=any(
                    arg["name"] == "base_url" and arg["value"] == base_url
                    for arg in model_args_list
                ),
            )

        # Add other default model args (always add these if not present)
        default_model_args = {
            "num_concurrent": self.backend_config.get("num_concurrent", "1"),
            "max_retries": self.backend_config.get("max_retries", "3"),
            "tokenized_requests": self.backend_config.get("tokenized_requests", "False"),
            "tokenizer": self.backend_config.get("tokenizer", "google/flan-t5-base"),
        }

        for name, default_value in default_model_args.items():
            if not any(arg["name"] == name for arg in model_args_list):
                model_args_list.append({"name": name, "value": str(default_value)})

        # Build pod container env
        pod_env = []
        secret_name = self.backend_config.get("secret_name")
        secret_key = self.backend_config.get("secret_key", "token")
        if secret_name:
            pod_env.append({
                "name": "OPENAI_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": secret_name,
                        "key": secret_key,
                    }
                }
            })

        # Construct the CR matching user's structure
        # Generate a UUID-based name for the CR
        job_uuid = uuid.uuid4()
        job_name = f"lmeval-job-{job_uuid}"

        job_cr = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "evaluation-id": str(context.evaluation_id),
                    "benchmark": context.benchmark_spec.name,
                    "model": model,
                    "backend": "lm-evaluation-harness",
                },
            },
            "spec": {
                "model": self.backend_config.get("model_name", "local-completions"),
                "taskList": {
                    "taskNames": tasks,
                },
                "logSamples": self.log_samples,
                "batchSize": str(self.batch_size),
                "allowCodeExecution": True,
                "allowOnline": True,
                "modelArgs": model_args_list,
            },
        }

        # Add pod configuration if env vars are present
        if pod_env:
            job_cr["spec"]["pod"] = {
                "container": {
                    "env": pod_env,
                }
            }

        # Add optional fields
        if limit is not None:
            job_cr["spec"]["limit"] = str(limit)
        if num_fewshot is not None:
            job_cr["spec"]["numFewshot"] = num_fewshot

        return job_cr

    async def _deploy_lmeval_job_cr(
        self, job_cr: dict[str, Any], context: ExecutionContext
    ) -> None:
        """Deploy LMEvalJob CR to Kubernetes."""
        if not self.k8s_client:
            raise BackendError("Kubernetes client not initialized")

        group = "trustyai.opendatahub.io"
        version = "v1alpha1"
        plural = "lmevaljobs"
        namespace = job_cr["metadata"]["namespace"]
        name = job_cr["metadata"]["name"]

        try:
            # Try to get existing CR
            try:
                self.k8s_client.get_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name,
                )
                # Update if exists
                self.logger.info(
                    "LMEvalJob CR already exists, updating",
                    name=name,
                    namespace=namespace,
                )
                self.k8s_client.patch_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name,
                    body=job_cr,
                )
            except ApiException as e:
                if e.status == 404:
                    # Create if doesn't exist
                    self.logger.info(
                        "Creating new LMEvalJob CR",
                        name=name,
                        namespace=namespace,
                    )
                    self.k8s_client.create_namespaced_custom_object(
                        group=group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        body=job_cr,
                    )
                else:
                    raise BackendError(f"Failed to deploy LMEvalJob CR: {e}") from e

        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.reason} (status: {e.status})"
            if e.body:
                try:
                    error_body = json.loads(e.body)
                    error_msg += f" - {error_body.get('message', '')}"
                except (json.JSONDecodeError, KeyError):
                    pass
            raise BackendError(error_msg) from e

    async def _run_lm_eval(
        self,
        tasks: list[str],
        context: ExecutionContext,
        model: str,
        progress_callback: Callable[[str, float, str], None] | None,
    ) -> dict[str, Any]:
        """Run lm-evaluation-harness command and return results."""

        # Build command
        cmd = [
            self.lm_eval_path,
            "--model",
            model,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(self.batch_size),
            "--device",
            self.device,
            "--num_fewshot",
            str(self.num_fewshot),
            "--output_path",
            self.output_path,
        ]

        # Add model args if provided
        if self.model_args:
            cmd.extend(["--model_args", self.model_args])

        # Add limit if provided
        if self.limit:
            cmd.extend(["--limit", str(self.limit)])

        # Add benchmark-specific config
        benchmark_config = context.benchmark_spec.config or {}
        if "limit" in benchmark_config:
            cmd.extend(["--limit", str(benchmark_config["limit"])])
        if "num_fewshot" in benchmark_config:
            cmd.extend(["--num_fewshot", str(benchmark_config["num_fewshot"])])

        # Add output file path
        output_file = f"{self.output_path}/results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        cmd.extend(["--output_path", output_file])

        self.logger.debug(
            "Running LM Evaluation Harness command",
            evaluation_id=str(context.evaluation_id),
            command=" ".join(cmd),
        )

        # Report progress: executing
        if progress_callback:
            progress_callback(
                str(context.evaluation_id),
                30.0,
                "Running LM Evaluation Harness evaluation",
            )

        # Run the command with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Monitor progress (simplified - in real implementation, parse stdout)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise BackendError(
                    f"LM Evaluation Harness failed with return code {process.returncode}: {error_msg}"
                )

            # Report progress: processing results
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    90.0,
                    "Processing LM Evaluation Harness results",
                )

            # Load results from output file
            if Path(output_file).exists():
                with open(output_file) as f:
                    result_data = json.load(f)
                return result_data
            else:
                # Try to parse stdout as JSON
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError as e:
                    raise BackendError(
                        "Failed to parse LM Evaluation Harness output as JSON"
                    ) from e

        except TimeoutError as e:
            raise BackendError(
                f"LM Evaluation Harness execution timed out after {self.timeout_seconds} seconds"
            ) from e

    async def _convert_lmeval_result_to_eval_hub(
        self, lmeval_result: dict[str, Any], context: ExecutionContext
    ) -> EvaluationResult:
        """Convert LM Evaluation Harness result to eval-hub EvaluationResult format."""

        metrics = {}
        artifacts = {}

        # Extract metrics from results
        # LM Evaluation Harness results structure: {task_name: {metric_name: value}}
        for task_name, task_results in lmeval_result.items():
            if isinstance(task_results, dict):
                for metric_name, metric_value in task_results.items():
                    # Flatten metric names
                    full_metric_name = (
                        f"{task_name}_{metric_name}"
                        if len(lmeval_result) > 1
                        else metric_name
                    )
                    metrics[full_metric_name] = metric_value

        # Add artifacts
        output_file = f"{self.output_path}/results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        artifacts["lmeval_results"] = output_file

        # Save full result for debugging
        full_result_file = f"{self.output_path}/full_results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        with open(full_result_file, "w") as f:
            json.dump(lmeval_result, f, indent=2, default=str)
        artifacts["lmeval_full_results"] = full_result_file

        return EvaluationResult(
            evaluation_id=context.evaluation_id,
            backend_name="lm-evaluation-harness",
            benchmark_name=context.benchmark_spec.name,
            status=EvaluationStatus.COMPLETED,
            metrics=metrics,
            artifacts=artifacts,
            started_at=context.started_at,
            completed_at=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - context.started_at).total_seconds(),
        )

    def get_recommended_timeout_minutes(self) -> int:
        """Get the recommended timeout for LM Evaluation Harness."""
        return self.timeout_seconds // 60

    def get_max_retry_attempts(self) -> int:
        """Get the maximum retry attempts for LM Evaluation Harness."""
        return self.backend_config.get("max_retries", 2)

