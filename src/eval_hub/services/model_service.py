"""Model service for managing model server registration and runtime configuration."""

import os
import re

from ..core.config import Settings
from ..core.logging import get_logger
from ..models.model import (
    ListModelServersResponse,
    ListModelsResponse,
    Model,
    ModelRegistrationRequest,
    ModelServer,
    ModelServerRegistrationRequest,
    ModelServerSummary,
    ModelServerUpdateRequest,
    ModelStatus,
    ModelType,
    ModelUpdateRequest,
    ServerModel,
)
from ..utils import utcnow

logger = get_logger(__name__)


class ModelService:
    """Service for managing model servers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._registered_servers: dict[str, ModelServer] = {}
        self._runtime_servers: dict[str, ModelServer] = {}
        self._registered_models: dict[str, Model] = {}
        self._runtime_models: dict[str, Model] = {}
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize the model service by loading runtime servers from environment variables."""
        if self._initialized:
            return

        self._load_runtime_servers()
        self._load_runtime_models_from_servers()
        self._initialized = True
        logger.info(
            "Model service initialized",
            registered_servers=len(self._registered_servers),
            runtime_servers=len(self._runtime_servers),
            registered_models=len(self._registered_models),
            runtime_models=len(self._runtime_models),
        )

    def _load_runtime_servers(self) -> None:
        """Load model servers specified via environment variables."""
        # Simple pattern: MODEL_SERVER_URL and MODEL_SERVER_TYPE (creates server with ID "default")
        # Optional: MODEL_SERVER_ID=<id> (defaults to "default")
        # Pattern: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_URL=<url>
        # Optional: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_ID=<id> (defaults to derived from env var name)
        # Optional: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_TYPE=<type>
        # Optional: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_MODELS=<comma-separated model names>
        # For backward compatibility, also support: EVAL_HUB_MODEL_<ID>_URL (creates server with single model)
        # Optional: EVAL_HUB_MODEL_<ID>_ID=<id> (defaults to derived from env var name)

        runtime_servers = {}

        # Simple pattern: MODEL_SERVER_URL and MODEL_SERVER_TYPE
        model_server_url = os.getenv("MODEL_SERVER_URL")
        if model_server_url:
            model_server_url = model_server_url.strip()
            if model_server_url:
                server_id = os.getenv("MODEL_SERVER_ID", "default")
                model_type_str = os.getenv("MODEL_SERVER_TYPE", "openai-compatible")
                models_str = os.getenv("MODEL_SERVER_MODELS", "")

                try:
                    server_type = ModelType(model_type_str.lower())
                except ValueError:
                    logger.warning(
                        f"Invalid server type '{model_type_str}' for MODEL_SERVER_TYPE, "
                        f"using default 'openai-compatible'"
                    )
                    server_type = ModelType.OPENAI_COMPATIBLE

                model_names = (
                    [m.strip() for m in models_str.split(",") if m.strip()]
                    if models_str
                    else []
                )
                if not model_names:
                    model_names = [server_id]

                server_models = []
                for model_name in model_names:
                    server_models.append(
                        ServerModel(
                            model_name=model_name,
                            description=None,
                            status=ModelStatus.ACTIVE,
                            tags=["runtime"],
                        )
                    )

                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=model_server_url,
                    api_key_required=True,
                    models=server_models,
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from MODEL_SERVER_URL environment variable",
                    server_id=server_id,
                    server_type=server_type.value,
                    base_url=model_server_url,
                    model_count=len(server_models),
                )

        # New pattern: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_URL
        for env_var, env_value in os.environ.items():
            if env_var.startswith("EVAL_HUB_MODEL_SERVER_") and env_var.endswith(
                "_URL"
            ):
                match = re.match(r"EVAL_HUB_MODEL_SERVER_(.+)_URL", env_var)
                if not match:
                    continue

                server_id = match.group(1).lower()
                base_url = env_value.strip()

                if not base_url:
                    logger.warning(
                        f"Empty URL for runtime server {server_id}, skipping"
                    )
                    continue

                # Get optional configuration
                type_var = f"EVAL_HUB_MODEL_SERVER_{match.group(1)}_TYPE"
                models_var = f"EVAL_HUB_MODEL_SERVER_{match.group(1)}_MODELS"
                id_var = f"EVAL_HUB_MODEL_SERVER_{match.group(1)}_ID"

                # Allow overriding server_id via env var, but default to derived value
                server_id = os.getenv(id_var, server_id)
                model_type_str = os.getenv(type_var, "openai-compatible")
                models_str = os.getenv(models_var, "")

                # Validate model type
                try:
                    server_type = ModelType(model_type_str.lower())
                except ValueError:
                    logger.warning(
                        f"Invalid server type '{model_type_str}' for runtime server {server_id}, "
                        f"using default 'openai-compatible'"
                    )
                    server_type = ModelType.OPENAI_COMPATIBLE

                # Parse model names (comma-separated)
                model_names = (
                    [m.strip() for m in models_str.split(",") if m.strip()]
                    if models_str
                    else []
                )

                # If no models specified, create a default model with the server_id
                if not model_names:
                    model_names = [server_id]

                # Create ServerModel objects
                server_models = []
                for model_name in model_names:
                    server_models.append(
                        ServerModel(
                            model_name=model_name,
                            description=None,
                            status=ModelStatus.ACTIVE,
                            tags=["runtime"],
                        )
                    )

                # Create runtime server
                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=base_url,
                    api_key_required=True,
                    models=server_models,
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from environment",
                    server_id=server_id,
                    server_type=server_type.value,
                    base_url=base_url,
                    model_count=len(server_models),
                )

        # Backward compatibility: EVAL_HUB_MODEL_<ID>_URL creates a server with a single model
        for env_var, env_value in os.environ.items():
            if (
                env_var.startswith("EVAL_HUB_MODEL_")
                and env_var.endswith("_URL")
                and "SERVER" not in env_var
            ):
                match = re.match(r"EVAL_HUB_MODEL_(.+)_URL", env_var)
                if not match:
                    continue

                server_id = match.group(1).lower()

                # Skip if already processed as a server
                if server_id in runtime_servers:
                    continue

                base_url = env_value.strip()
                if not base_url:
                    continue

                type_var = f"EVAL_HUB_MODEL_{match.group(1)}_TYPE"
                id_var = f"EVAL_HUB_MODEL_{match.group(1)}_ID"
                name_var = f"EVAL_HUB_MODEL_{match.group(1)}_NAME"
                path_var = f"EVAL_HUB_MODEL_{match.group(1)}_PATH"

                # Allow overriding server_id via env var, but default to derived value
                server_id = os.getenv(id_var, server_id)
                model_type_str = os.getenv(type_var, "openai-compatible")
                model_name = os.getenv(name_var)
                model_path = os.getenv(path_var)

                try:
                    server_type = ModelType(model_type_str.lower())
                except ValueError:
                    server_type = ModelType.OPENAI_COMPATIBLE

                # Use provided name or default to "Runtime Model <ID>"
                if not model_name:
                    model_name = f"Runtime Model {match.group(1).upper()}"

                # Create server with single model
                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=base_url,
                    api_key_required=True,
                    models=[
                        ServerModel(
                            model_name=model_name,
                            description=None,
                            status=ModelStatus.ACTIVE,
                            tags=["runtime"],
                        )
                    ],
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )

                # Store model_path in server metadata for later use
                runtime_server.model_path = model_path

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from legacy environment variable",
                    server_id=server_id,
                    base_url=base_url,
                )

        self._runtime_servers = runtime_servers

    def _load_runtime_models_from_servers(self) -> None:
        """Load runtime models from runtime servers."""
        from ..models.model import Model, ModelCapabilities, ModelConfig

        self._runtime_models.clear()

        for server in self._runtime_servers.values():
            for server_model in server.models:
                # Use server_id as model_id if only one model, otherwise use model_name
                model_id = (
                    server.server_id
                    if len(server.models) == 1
                    else server_model.model_name
                )

                # Get model_path from server metadata if available (set during _load_runtime_servers)
                # If not set, use None (don't default to model_name)
                model_path = getattr(server, "model_path", None)

                model = Model(
                    model_id=model_id,
                    model_name=server_model.model_name,
                    description=server_model.description or "",
                    model_type=server.server_type,
                    base_url=server.base_url,
                    api_key_required=server.api_key_required,
                    model_path=model_path,
                    capabilities=ModelCapabilities(),
                    config=ModelConfig(),
                    status=server_model.status,
                    tags=server_model.tags,
                    created_at=server.created_at,
                    updated_at=server.updated_at,
                )

                self._runtime_models[model_id] = model

    def register_server(self, request: ModelServerRegistrationRequest) -> ModelServer:
        """Register a new model server."""
        self._initialize()

        # Check if server ID already exists
        if request.server_id in self._registered_servers:
            raise ValueError(f"Server with ID '{request.server_id}' already exists")

        if request.server_id in self._runtime_servers:
            raise ValueError(
                f"Server with ID '{request.server_id}' is specified as runtime server via environment variable"
            )

        # Create the server
        now = utcnow()
        server = ModelServer(
            server_id=request.server_id,
            server_type=request.server_type,
            base_url=request.base_url,
            api_key_required=request.api_key_required,
            models=request.models or [],
            server_config=request.server_config,
            status=request.status,
            tags=request.tags,
            created_at=now,
            updated_at=now,
        )

        self._registered_servers[request.server_id] = server

        logger.info(
            "Model server registered successfully",
            server_id=request.server_id,
            server_type=request.server_type.value,
            model_count=len(server.models),
        )

        return server

    def get_server_by_id(self, server_id: str) -> ModelServer | None:
        """Get a server by ID (from either registered or runtime servers)."""
        self._initialize()

        # Check registered servers first
        if server_id in self._registered_servers:
            return self._registered_servers[server_id]

        # Check runtime servers
        if server_id in self._runtime_servers:
            return self._runtime_servers[server_id]

        return None

    def get_model_on_server(
        self, server_id: str, model_name: str
    ) -> tuple[ModelServer, ServerModel] | None:
        """Get a specific model on a server. Returns (server, model) tuple if found."""
        self._initialize()

        server = self.get_server_by_id(server_id)
        if not server:
            return None

        # Find the model on the server
        for model in server.models:
            if model.model_name == model_name:
                return (server, model)

        return None

    def get_all_servers(
        self, include_inactive: bool = True
    ) -> ListModelServersResponse:
        """Get all model servers (registered and runtime)."""
        self._initialize()

        # Convert registered servers to summaries
        registered_summaries = []
        for server in self._registered_servers.values():
            if include_inactive or server.status == ModelStatus.ACTIVE:
                summary = ModelServerSummary(
                    server_id=server.server_id,
                    server_type=server.server_type,
                    base_url=server.base_url,
                    model_count=len(server.models),
                    status=server.status,
                    tags=server.tags,
                    created_at=server.created_at,
                )
                registered_summaries.append(summary)

        # Convert runtime servers to summaries
        runtime_summaries = []
        for server in self._runtime_servers.values():
            summary = ModelServerSummary(
                server_id=server.server_id,
                server_type=server.server_type,
                base_url=server.base_url,
                model_count=len(server.models),
                status=server.status,
                tags=server.tags,
                created_at=server.created_at,
            )
            runtime_summaries.append(summary)

        # Combine all servers
        all_summaries = registered_summaries + runtime_summaries

        return ListModelServersResponse(
            servers=all_summaries,
            total_servers=len(all_summaries),
            runtime_servers=runtime_summaries,
        )

    def update_server(
        self, server_id: str, request: ModelServerUpdateRequest
    ) -> ModelServer | None:
        """Update an existing registered server."""
        self._initialize()

        if server_id in self._runtime_servers:
            raise ValueError(
                "Cannot update runtime servers specified via environment variables"
            )

        if server_id not in self._registered_servers:
            return None

        server = self._registered_servers[server_id]

        # Update fields that are provided
        if request.base_url is not None:
            server.base_url = request.base_url
        if request.api_key_required is not None:
            server.api_key_required = request.api_key_required
        if request.models is not None:
            server.models = request.models
        if request.server_config is not None:
            server.server_config = request.server_config
        if request.status is not None:
            server.status = request.status
        if request.tags is not None:
            server.tags = request.tags

        server.updated_at = utcnow()

        logger.info("Server updated successfully", server_id=server_id)

        return server

    def delete_server(self, server_id: str) -> bool:
        """Delete a registered server."""
        self._initialize()

        if server_id in self._runtime_servers:
            raise ValueError(
                "Cannot delete runtime servers specified via environment variables"
            )

        if server_id in self._registered_servers:
            del self._registered_servers[server_id]
            logger.info("Server deleted successfully", server_id=server_id)
            return True

        return False

    def reload_runtime_servers(self) -> None:
        """Reload runtime servers from environment variables."""
        self._runtime_servers.clear()
        self._load_runtime_servers()
        logger.info("Runtime servers reloaded from environment variables")

    # Model-level methods (direct storage)
    def get_all_models(self, include_inactive: bool = True) -> "ListModelsResponse":
        """Get all models."""
        from ..models.model import ListModelsResponse, ModelSummary

        self._initialize()

        models: list[ModelSummary] = []
        runtime_models: list[ModelSummary] = []

        # Collect registered models
        for model in self._registered_models.values():
            if not include_inactive and model.status != ModelStatus.ACTIVE:
                continue
            models.append(
                ModelSummary(
                    model_id=model.model_id,
                    model_name=model.model_name,
                    description=model.description,
                    model_type=model.model_type,
                    base_url=model.base_url,
                    status=model.status,
                    tags=model.tags,
                    created_at=model.created_at,
                )
            )

        # Collect runtime models
        for model in self._runtime_models.values():
            if not include_inactive and model.status != ModelStatus.ACTIVE:
                continue
            runtime_models.append(
                ModelSummary(
                    model_id=model.model_id,
                    model_name=model.model_name,
                    description=model.description,
                    model_type=model.model_type,
                    base_url=model.base_url,
                    status=model.status,
                    tags=model.tags,
                    created_at=model.created_at,
                )
            )

        all_models = models + runtime_models
        return ListModelsResponse(
            models=all_models,
            total_models=len(all_models),
            runtime_models=runtime_models,
        )

    def get_model_by_id(self, model_id: str) -> "Model | None":
        """Get a model by ID."""
        self._initialize()

        # Check registered models first
        if model_id in self._registered_models:
            return self._registered_models[model_id]

        # Check runtime models
        if model_id in self._runtime_models:
            return self._runtime_models[model_id]

        return None

    def register_model(self, request: "ModelRegistrationRequest") -> "Model":
        """Register a new model."""
        from ..models.model import Model, ModelCapabilities, ModelConfig

        self._initialize()

        # Check if model ID already exists
        if request.model_id in self._registered_models:
            raise ValueError(f"Model with ID '{request.model_id}' already exists")

        if request.model_id in self._runtime_models:
            raise ValueError(
                f"Model with ID '{request.model_id}' is specified as runtime model"
            )

        # Create the model
        now = utcnow()
        model = Model(
            model_id=request.model_id,
            model_name=request.model_name,
            description=request.description,
            model_type=request.model_type,
            base_url=request.base_url,
            api_key_required=request.api_key_required,
            model_path=request.model_path,
            capabilities=request.capabilities or ModelCapabilities(),
            config=request.config or ModelConfig(),
            status=request.status,
            tags=request.tags,
            created_at=now,
            updated_at=now,
        )

        self._registered_models[request.model_id] = model

        logger.info(
            "Model registered successfully",
            model_id=request.model_id,
            model_name=request.model_name,
        )

        return model

    def update_model(
        self, model_id: str, request: "ModelUpdateRequest"
    ) -> "Model | None":
        """Update an existing model."""
        self._initialize()

        # Check if it's a runtime model
        if model_id in self._runtime_models:
            raise ValueError("Cannot update runtime models")

        # Get the model
        if model_id not in self._registered_models:
            return None

        model = self._registered_models[model_id]

        # Update fields
        if request.model_name is not None:
            model.model_name = request.model_name
        if request.description is not None:
            model.description = request.description
        if request.model_type is not None:
            model.model_type = request.model_type
        if request.base_url is not None:
            model.base_url = request.base_url
        if request.api_key_required is not None:
            model.api_key_required = request.api_key_required
        if request.model_path is not None:
            model.model_path = request.model_path
        if request.capabilities is not None:
            model.capabilities = request.capabilities
        if request.config is not None:
            model.config = request.config
        if request.status is not None:
            model.status = request.status
        if request.tags is not None:
            model.tags = request.tags

        model.updated_at = utcnow()

        logger.info("Model updated successfully", model_id=model_id)
        return model

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        self._initialize()

        # Check if it's a runtime model
        if model_id in self._runtime_models:
            raise ValueError("Cannot delete runtime models")

        if model_id in self._registered_models:
            del self._registered_models[model_id]
            logger.info("Model deleted successfully", model_id=model_id)
            return True

        return False

    def search_models(
        self,
        model_type: "ModelType | None" = None,
        status: "ModelStatus | None" = None,
        tags: list[str] | None = None,
    ) -> list["Model"]:
        """Search models by type, status, or tags."""
        self._initialize()

        results: list[Model] = []

        # Search in both registered and runtime models
        for model in list(self._registered_models.values()) + list(
            self._runtime_models.values()
        ):
            # Filter by type
            if model_type is not None and model.model_type != model_type:
                continue

            # Filter by status
            if status is not None and model.status != status:
                continue

            # Filter by tags
            if tags is not None:
                if not any(tag in model.tags for tag in tags):
                    continue

            results.append(model)

        return results

    def get_active_models(self) -> list["Model"]:
        """Get all active models."""
        return self.search_models(status=ModelStatus.ACTIVE)

    def reload_runtime_models(self) -> None:
        """Reload runtime models from environment variables."""
        # Reload runtime servers
        self._load_runtime_servers()
        # Reload runtime models from servers
        self._load_runtime_models_from_servers()
        logger.info("Runtime models reloaded from environment variables")
