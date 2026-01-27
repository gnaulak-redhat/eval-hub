"""Logging configuration for the evaluation service."""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger

from .config import Settings


def setup_logging(settings: Settings) -> None:
    """Set up structured logging configuration."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            (
                structlog.dev.ConsoleRenderer()
                if settings.debug
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def log_request_start(
    logger: FilteringBoundLogger,
    request_id: str,
    endpoint: str,
    method: str,
    **kwargs: Any,
) -> None:
    """Log the start of a request."""
    logger.info(
        "Request started",
        request_id=request_id,
        endpoint=endpoint,
        method=method,
        **kwargs,
    )


def log_request_end(
    logger: FilteringBoundLogger,
    request_id: str,
    status_code: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log the end of a request."""
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_evaluation_start(
    logger: FilteringBoundLogger,
    evaluation_id: str,
    model_name: str,
    backend_count: int,
    **kwargs: Any,
) -> None:
    """Log the start of an evaluation."""
    logger.info(
        "Evaluation started",
        evaluation_id=evaluation_id,
        model_name=model_name,
        backend_count=backend_count,
        **kwargs,
    )


def log_evaluation_complete(
    logger: FilteringBoundLogger,
    evaluation_id: str,
    status: str,
    duration_seconds: float,
    **kwargs: Any,
) -> None:
    """Log the completion of an evaluation."""
    logger.info(
        "Evaluation completed",
        evaluation_id=evaluation_id,
        status=status,
        duration_seconds=duration_seconds,
        **kwargs,
    )
