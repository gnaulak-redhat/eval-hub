# Eval Hub

[![CI](https://github.com/eval-hub/eval-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/eval-hub/eval-hub/actions/workflows/ci.yml)

A Go API REST server built with net/http that serves as a routing and orchestration layer for evaluation backends. Supports local development and Podman containers.

## Overview

The Evaluation Hub is designed to:

- Parse requests containing lists of evaluations for each backend
- Route and orchestrate evaluation execution across multiple backends
- Store results and aggregate responses to clients
- Handle requests concurrently and asynchronously
- Deploy locally for development or as Podman containers

## Features

- **Multi-Backend Support**: Orchestrates evaluations across different backends (lm-evaluation-harness, Lighteval, RAGAS, Garak, custom backends)
- **Collection Management**: Create, manage, and execute curated collections of benchmarks via the API
- **Provider & Benchmark Discovery**: Comprehensive API for discovering evaluation providers and their available benchmarks
- **Async Execution**: Handles requests concurrently with progress tracking
- **Monitoring**: Prometheus metrics and health checks
- **Structured Logging**: Request-scoped logging with zap (request ID, method, URI, and more)
- **ExecutionContext**: Evaluation-related handlers receive an execution context with logger, config, and request metadata

## Architecture

### Core Components

1. **Handlers**: HTTP request handlers for evaluations, collections, providers, benchmarks, health, status, and OpenAPI
2. **ExecutionContext**: Request-scoped context with logger, configuration, and evaluation metadata
3. **Configuration**: Viper-based config loading from `config/config.yaml` with environment and secrets mapping
4. **Metrics**: Prometheus middleware for request duration and status codes
5. **Storage**: Pluggable storage abstraction (SQLite in-memory or PostgreSQL via config)

## Quick Start

### Prerequisites

- Go 1.25 or higher
- Podman (for container builds)

### Running the Service

#### Using Make (Recommended)

1. Install dependencies:

   ```bash
   make install-deps
   ```

2. Run the server:

   ```bash
   make start-service
   ```

   The server starts on port **8080** by default. Override with the `PORT` environment variable:

   ```bash
   PORT=3000 make start-service
   ```

3. View logs:

   ```bash
   tail -f bin/service.log
   ```

4. Stop the server:

   ```bash
   make stop-service
   ```

#### Using Go directly

1. Install dependencies:

   ```bash
   go mod download
   ```

2. Run the server:

   ```bash
   go run cmd/eval_hub/main.go
   ```

   Default port is **8080**. Override with:

   ```bash
   PORT=3000 go run cmd/eval_hub/main.go
   ```

3. Access the API:

   - API documentation (Swagger UI): http://localhost:8080/docs
   - OpenAPI spec: http://localhost:8080/openapi.yaml
   - Health check: http://localhost:8080/api/v1/health
   - Metrics: http://localhost:8080/metrics

## API Endpoints

### Evaluations

- `POST /api/v1/evaluations/jobs` - Create Evaluation
- `GET /api/v1/evaluations/jobs` - List Evaluations
- `GET /api/v1/evaluations/jobs/{id}` - Get Evaluation Status
- `DELETE /api/v1/evaluations/jobs/{id}` - Cancel Evaluation
- `GET /api/v1/evaluations/jobs/{id}/summary` - Get Evaluation Summary

### Benchmarks

- `GET /api/v1/evaluations/benchmarks` - List All Benchmarks

### Collections

- `GET /api/v1/evaluations/collections` - List Collections
- `POST /api/v1/evaluations/collections` - Create Collection
- `GET /api/v1/evaluations/collections/{collection_id}` - Get Collection
- `PUT /api/v1/evaluations/collections/{collection_id}` - Update Collection
- `PATCH /api/v1/evaluations/collections/{collection_id}` - Patch Collection
- `DELETE /api/v1/evaluations/collections/{collection_id}` - Delete Collection

### Providers

- `GET /api/v1/evaluations/providers` - List Providers
- `GET /api/v1/evaluations/providers/{provider_id}` - Get Provider

### Health & Status

- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/status` - Service status endpoint

### Metrics

- `GET /api/v1/metrics/system` - Get System Metrics
- `GET /metrics` - Prometheus metrics endpoint

### Documentation

- `GET /openapi.yaml` - OpenAPI 3.1.0 specification
- `GET /docs` - Interactive API documentation (Swagger UI)

## API Documentation

For comprehensive API documentation including request/response formats and examples, see **[API.md](./API.md)**.

Key API capabilities:

- **Evaluation Management**: Create, monitor, and manage evaluation jobs
- **Provider Integration**: Support for LM-Evaluation-Harness, RAGAS, Garak, Lighteval, and custom providers
- **Collection Management**: Curated benchmark collections for domain-specific evaluation
- **Real-time Monitoring**: Health checks, metrics, and system status endpoints

## Configuration

Configuration is loaded from `config/config.yaml` using Viper. Values can be overridden by environment variables and optional secrets files.

### Environment Mappings

Common mappings (defined in config):

- `PORT` → `service.port` (default 8080)
- `DB_URL` → `database.url`

### Database Setup (Optional)

For PostgreSQL-backed storage, use the Makefile targets:

```bash
make install-postgres   # Install PostgreSQL (macOS/Linux)
make start-postgres    # Start PostgreSQL service
make create-database   # Create eval_hub database
make create-user       # Create eval_hub user
make grant-permissions # Grant permissions to user
```

See `config/config.yaml` and `config/providers/` for provider-specific configuration (e.g., lm_evaluation_harness, ragas, garak, lighteval).

## Building

### Using Make

Build the binary:

```bash
make build
```

Run the binary:

```bash
./bin/eval-hub
```

### Using Go directly

Build the binary:

```bash
go build -o bin/eval-hub ./cmd/eval_hub
```

Run the binary:

```bash
./bin/eval-hub
```

## Container Build and Run

Build the container image:

```bash
podman build -t eval-hub:latest \
  --build-arg BUILD_NUMBER=0.0.1 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -f Containerfile .
```

This builds the image with:

- Go 1.25 toolchain (UBI9 base)
- Build metadata (version and timestamp)
- Multi-stage build for minimal final image

Run the container locally:

```bash
podman run --rm -p 8080:8080 eval-hub:latest
```

The service is available at http://localhost:8080.

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Display all available targets |
| `make clean` | Remove build artifacts and coverage files |
| `make build` | Build the binary to bin/eval-hub |
| `make start-service` | Start the application in background (port 8080) |
| `make stop-service` | Stop the application |
| `make lint` | Lint the code (go vet) |
| `make fmt` | Format code with go fmt |
| `make vet` | Run go vet |
| `make test` | Run unit tests (internal/...) |
| `make test-fvt` | Run FVT tests using godog (tests/features/...) |
| `make test-all` | Run all tests (unit + FVT) |
| `make test-coverage` | Run unit tests with coverage report (bin/coverage.html) |
| `make test-fvt-coverage` | Run FVT tests with coverage |
| `make test-all-coverage` | Run all tests with coverage |
| `make install-deps` | Download and tidy dependencies |
| `make update-deps` | Update all dependencies to latest |
| `make get-deps` | Get all dependencies |
| `make pre-commit` | Install/update pre-commit hooks |
| `make install-postgres` | Install PostgreSQL (macOS/Linux) |
| `make start-postgres` | Start PostgreSQL service |
| `make stop-postgres` | Stop PostgreSQL service |
| `make create-database` | Create eval_hub database |
| `make create-user` | Create eval_hub user |
| `make grant-permissions` | Grant database permissions to user |

## Project Structure

This project follows the [standard Go project layout](https://github.com/golang-standards/project-layout):

```
eval-hub/
├── cmd/
│   └── eval_hub/              # Main application entry point
│       ├── main.go
│       └── server/            # Server setup and routing
│           ├── server.go
│           ├── execution_context.go
│           ├── middleware.go
│           ├── middleware_test.go
│           └── server_test.go
├── internal/                   # Private application code
│   ├── config/                # Configuration loading (Viper)
│   ├── constants/             # Shared constants (log fields, env vars)
│   ├── executioncontext/      # ExecutionContext pattern
│   ├── handlers/              # HTTP request handlers
│   │   ├── handlers.go        # Basic handlers
│   │   ├── evaluations.go
│   │   ├── collections.go
│   │   ├── providers.go
│   │   ├── benchmarks.go
│   │   ├── health.go
│   │   ├── status.go
│   │   ├── openapi.go
│   │   ├── system_metrics.go
│   │   └── *_test.go
│   ├── logging/               # Logger creation and request enhancement
│   ├── metrics/               # Prometheus metrics
│   ├── serialization/
│   ├── storage/               # Storage abstraction (SQLite, PostgreSQL)
│   ├── validation/
│   └── abstractions/
├── pkg/
│   └── api/                   # Shared API types
├── api/
│   └── openapi.yaml           # OpenAPI 3.1.0 specification
├── config/
│   ├── config.yaml            # Main configuration
│   └── providers/             # Provider-specific config (e.g. ragas, garak, lighteval)
├── scripts/                   # start_server.sh, stop_server.sh
├── tests/
│   └── features/              # BDD-style FVT tests (godog)
│       ├── health.feature
│       ├── status.feature
│       ├── metrics.feature
│       ├── evaluations.feature
│       ├── step_definitions_test.go
│       └── suite_test.go
├── Makefile
├── go.mod
├── go.sum
└── Containerfile
```

## Testing

### Unit Tests

Unit tests are in `*_test.go` files alongside the code:

- `internal/handlers/*_test.go` - Handler and OpenAPI tests
- `cmd/eval_hub/server/server_test.go` - Server tests
- `cmd/eval_hub/server/middleware_test.go` - Metrics middleware tests

Run unit tests:

```bash
make test
```

### FVT (Functional Verification Tests)

FVT tests use [godog](https://github.com/cucumber/godog) for BDD-style testing:

- Feature files: `tests/features/*.feature` (health, status, metrics, evaluations)
- Step definitions: `tests/features/step_definitions_test.go`

Run FVT tests:

```bash
make test-fvt
```

Run all tests:

```bash
make test-all
```

Generate coverage report:

```bash
make test-coverage
# Open bin/coverage.html
```

## Implementation Details

### Structured Logging

The service uses [zap](https://github.com/uber-go/zap) for structured JSON logging. Each request is enriched with:

- **Request ID**: From `X-Global-Transaction-Id` header or auto-generated UUID
- **HTTP Method**: Request method (GET, POST, etc.)
- **URI**: Request path
- **User Agent**: Client user agent
- **Remote Address**: Client IP
- **Remote User**: Authenticated user (if available)
- **Referer**: HTTP referer (if present)

### Execution Context

Evaluation-related handlers receive an `ExecutionContext` that includes:

- Logger with request-specific fields
- Service configuration (timeouts, retries, etc.)
- Model and benchmark specifications
- Metadata and experiment information

### Dependencies

Key dependencies:

- **zap** (`go.uber.org/zap`) - Structured logging
- **Prometheus** (`github.com/prometheus/client_golang`) - Metrics
- **Viper** (`github.com/spf13/viper`) - Configuration
- **godog** (`github.com/cucumber/godog`) - BDD testing
- **uuid** (`github.com/google/uuid`) - UUID generation
- **validator** (`github.com/go-playground/validator/v10`) - Request validation

## Monitoring

- **Health**: `GET /api/v1/health` for liveness/readiness
- **Status**: `GET /api/v1/status` for service status
- **Metrics**: `GET /metrics` for Prometheus (request counts, duration, status codes)
- **Logs**: Structured JSON with request IDs for correlation

## Troubleshooting

### Common Issues

1. **Port in use**: Change port with `PORT=3000 make start-service` or set in config.
2. **Database errors**: For SQLite (default), no setup is needed. For PostgreSQL, run `make create-database`, `make create-user`, `make grant-permissions` and set `DB_URL` in config.
3. **Logs**: Check `bin/service.log` when running via `make start-service`, or stdout when using `go run cmd/eval_hub/main.go`.

### Logs and Debugging

Use the metrics endpoint for request volume and latency, and the evaluation job endpoints for per-job status. Request IDs in logs help correlate requests across the service.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for development setup, coding standards, and the contribution process.

Quick links:

- [Development Setup](CONTRIBUTING.md#development-setup)
- [Code Standards](CONTRIBUTING.md#code-standards)
- [Pull Request Process](CONTRIBUTING.md#pull-request-process)
- [Issue Reporting](CONTRIBUTING.md#issue-reporting)

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.
