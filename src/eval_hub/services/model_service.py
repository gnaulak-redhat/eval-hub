"""Model service for managing model server registration and runtime configuration."""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from ..core.config import Settings
from ..core.logging import get_logger
from ..models.model import (
    Model,
    ModelSummary,
    ModelType,
    ModelStatus,
    ModelCapabilities,
    ModelConfig,
    ModelRegistrationRequest,
    ModelUpdateRequest,
    ListModelsResponse,
    RuntimeModelConfig,
    ModelsData,
    ModelServer,
    ModelServerSummary,
    ModelServerRegistrationRequest,
    ModelServerUpdateRequest,
    ListModelServersResponse,
    ServerModel,
)

logger = get_logger(__name__)


class ModelService:
    """Service for managing model servers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._registered_servers: Dict[str, ModelServer] = {}
        self._runtime_servers: Dict[str, ModelServer] = {}
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize the model service by loading runtime servers from environment variables."""
        if self._initialized:
            return

        self._load_runtime_servers()
        self._initialized = True
        logger.info(
            "Model service initialized",
            registered_servers=len(self._registered_servers),
            runtime_servers=len(self._runtime_servers)
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

                model_names = [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []
                if not model_names:
                    model_names = [server_id]

                server_models = []
                for model_name in model_names:
                    server_models.append(ServerModel(
                        model_name=model_name,
                        description=None,
                        status=ModelStatus.ACTIVE,
                        tags=["runtime"]
                    ))

                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=model_server_url,
                    api_key_required=True,
                    models=server_models,
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from MODEL_SERVER_URL environment variable",
                    server_id=server_id,
                    server_type=server_type.value,
                    base_url=model_server_url,
                    model_count=len(server_models)
                )

        # New pattern: EVAL_HUB_MODEL_SERVER_<SERVER_ID>_URL
        for env_var, env_value in os.environ.items():
            if env_var.startswith("EVAL_HUB_MODEL_SERVER_") and env_var.endswith("_URL"):
                match = re.match(r'EVAL_HUB_MODEL_SERVER_(.+)_URL', env_var)
                if not match:
                    continue

                server_id = match.group(1).lower()
                base_url = env_value.strip()

                if not base_url:
                    logger.warning(f"Empty URL for runtime server {server_id}, skipping")
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
                model_names = [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []
                
                # If no models specified, create a default model with the server_id
                if not model_names:
                    model_names = [server_id]

                # Create ServerModel objects
                server_models = []
                for model_name in model_names:
                    server_models.append(ServerModel(
                        model_name=model_name,
                        description=None,
                        status=ModelStatus.ACTIVE,
                        tags=["runtime"]
                    ))

                # Create runtime server
                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=base_url,
                    api_key_required=True,
                    models=server_models,
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from environment",
                    server_id=server_id,
                    server_type=server_type.value,
                    base_url=base_url,
                    model_count=len(server_models)
                )

        # Backward compatibility: EVAL_HUB_MODEL_<ID>_URL creates a server with a single model
        for env_var, env_value in os.environ.items():
            if env_var.startswith("EVAL_HUB_MODEL_") and env_var.endswith("_URL") and "SERVER" not in env_var:
                match = re.match(r'EVAL_HUB_MODEL_(.+)_URL', env_var)
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

                # Allow overriding server_id via env var, but default to derived value
                server_id = os.getenv(id_var, server_id)
                model_type_str = os.getenv(type_var, "openai-compatible")

                try:
                    server_type = ModelType(model_type_str.lower())
                except ValueError:
                    server_type = ModelType.OPENAI_COMPATIBLE

                # Create server with single model (using server_id as model name)
                runtime_server = ModelServer(
                    server_id=server_id,
                    server_type=server_type,
                    base_url=base_url,
                    api_key_required=True,
                    models=[ServerModel(
                        model_name=server_id,
                        description=None,
                        status=ModelStatus.ACTIVE,
                        tags=["runtime"]
                    )],
                    status=ModelStatus.ACTIVE,
                    tags=["runtime"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )

                runtime_servers[server_id] = runtime_server
                logger.info(
                    "Loaded runtime server from legacy environment variable",
                    server_id=server_id,
                    base_url=base_url
                )

        self._runtime_servers = runtime_servers

    def register_server(self, request: ModelServerRegistrationRequest) -> ModelServer:
        """Register a new model server."""
        self._initialize()

        # Check if server ID already exists
        if request.server_id in self._registered_servers:
            raise ValueError(f"Server with ID '{request.server_id}' already exists")

        if request.server_id in self._runtime_servers:
            raise ValueError(f"Server with ID '{request.server_id}' is specified as runtime server via environment variable")

        # Create the server
        now = datetime.utcnow()
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
            updated_at=now
        )

        self._registered_servers[request.server_id] = server

        logger.info(
            "Model server registered successfully",
            server_id=request.server_id,
            server_type=request.server_type.value,
            model_count=len(server.models)
        )

        return server

    def get_server_by_id(self, server_id: str) -> Optional[ModelServer]:
        """Get a server by ID (from either registered or runtime servers)."""
        self._initialize()

        # Check registered servers first
        if server_id in self._registered_servers:
            return self._registered_servers[server_id]

        # Check runtime servers
        if server_id in self._runtime_servers:
            return self._runtime_servers[server_id]

        return None

    def get_model_on_server(self, server_id: str, model_name: str) -> Optional[Tuple[ModelServer, ServerModel]]:
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

    def get_all_servers(self, include_inactive: bool = True) -> ListModelServersResponse:
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
                    created_at=server.created_at
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
                created_at=server.created_at
            )
            runtime_summaries.append(summary)

        # Combine all servers
        all_summaries = registered_summaries + runtime_summaries

        return ListModelServersResponse(
            servers=all_summaries,
            total_servers=len(all_summaries),
            runtime_servers=runtime_summaries
        )

    def update_server(self, server_id: str, request: ModelServerUpdateRequest) -> Optional[ModelServer]:
        """Update an existing registered server."""
        self._initialize()

        if server_id in self._runtime_servers:
            raise ValueError("Cannot update runtime servers specified via environment variables")

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

        server.updated_at = datetime.utcnow()

        logger.info(
            "Server updated successfully",
            server_id=server_id
        )

        return server

    def delete_server(self, server_id: str) -> bool:
        """Delete a registered server."""
        self._initialize()

        if server_id in self._runtime_servers:
            raise ValueError("Cannot delete runtime servers specified via environment variables")

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