# Contributing to Eval Hub

Thank you for your interest in contributing to Eval Hub! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

Eval Hub is an API REST server that serves as a routing and orchestration layer for evaluation backends. It supports flexible deployment options from local development to production Kubernetes/OpenShift clusters. Before contributing, familiarize yourself with:

- **Architecture**: Read the [README.md](README.md) for project overview
- **API Documentation**: Check [API.md](API.md) for endpoint specifications
- **Deployment Options**: Understand local development, Podman, and Kubernetes/OpenShift deployment models

### Prerequisites

**Required for All Development:**
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Git

**Optional for Container Testing:**
- Podman (for containerization testing)

**Optional for Cluster Integration Testing:**
- Access to a Kubernetes/OpenShift cluster
- kubectl or oc CLI tools

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/eval-hub.git
   cd eval-hub
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install development dependencies
   uv pip install -e ".[dev]"
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Setup**
   ```bash
   # Run tests to verify everything works
   pytest

   # Start the development server
   python -m eval_hub.main
   ```

## How to Contribute

We welcome contributions in various forms:

### Types of Contributions

- **Bug Fixes**: Fix issues in existing functionality
- **Features**: Add new evaluation backends, API endpoints, or capabilities
- **Documentation**: Improve README, API docs, or add examples
- **Testing**: Add test coverage or improve test infrastructure
- **Performance**: Optimize existing code or reduce resource usage
- **DevOps**: Improve CI/CD, deployment, or monitoring

### Contribution Areas

1. **Backend Executors**: Add support for new evaluation frameworks
2. **API Endpoints**: Extend the REST API with new functionality
3. **Deployment Integration**: Improve local, Podman, or Kubernetes deployment and orchestration
4. **MLFlow Integration**: Enhance experiment tracking capabilities
5. **Monitoring**: Add metrics, logging, or health checks
6. **Documentation**: User guides, API documentation, examples

## Development Workflow

### 1. Create an Issue

Before starting work, create an issue to discuss:
- **Bug Reports**: Describe the problem with reproduction steps
- **Feature Requests**: Explain the use case and proposed solution
- **Architectural Changes**: See special requirements below
- **Questions**: Ask for clarification or guidance

#### Architectural Changes

**Definition**: Changes that affect system design, component interactions, or technology choices, including:
- New backend executors or evaluation frameworks
- API endpoint additions or modifications
- Database schema changes
- Deployment architecture updates
- New dependencies or technology stack changes
- Performance or security architectural decisions

**Required Process**:
1. **Create Issue**: Use `kind/architecture` label
2. **Discussion**: Allow community input and maintainer feedback in the issue
3. **Approval**: Maintainers add `status/accepted` label after discussion
4. **Implementation**: Only proceed with implementation after approval
5. **Closure**: Issues without approval will be closed with explanation

**Note**: Implementation PRs for architectural changes will only be accepted if the corresponding issue has `status/accepted` label.

### 2. Branch Strategy

```bash
# Create a feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 3. Development Process

1. **Write Tests First**: For new features, write tests before implementation
2. **Implement Changes**: Write code following our standards
3. **Test Locally**: Run full test suite and verify functionality
4. **Document Changes**: Update relevant documentation

### 4. Commit Guidelines

Use conventional commits:
```bash
# Format: type(scope): description
git commit -m "feat(api): add collection-based evaluation endpoint"
git commit -m "fix(executor): handle timeout errors in NeMo evaluator"
git commit -m "docs(readme): update deployment instructions"
git commit -m "test(integration): add MLFlow integration tests"
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

## Code Standards

### Code Quality Tools

We use automated tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Python Standards

- **Python Version**: Support 3.12+
- **Code Style**: Follow PEP 8 (enforced by Black)
- **Type Hints**: Add type annotations to all new code
- **Docstrings**: Use Google-style docstrings for public APIs
- **Import Order**: Follow isort conventions (handled by Ruff)

### Code Organization

- **Modules**: Keep modules focused and cohesive
- **Dependencies**: Add new dependencies judiciously
- **Error Handling**: Use appropriate exceptions with clear messages
- **Logging**: Use structured logging with appropriate levels
- **Configuration**: Use Pydantic models for configuration

### Example Code Structure

```python
"""Module for handling evaluation requests."""

from typing import List, Optional
from pydantic import BaseModel
from structlog import get_logger

logger = get_logger(__name__)

class EvaluationRequest(BaseModel):
    """Evaluation request model."""

    model: str
    benchmarks: List[str]
    experiment_name: Optional[str] = None

async def process_evaluation(request: EvaluationRequest) -> dict:
    """Process evaluation request.

    Args:
        request: The evaluation request to process

    Returns:
        Evaluation results dictionary

    Raises:
        ValidationError: If request is invalid
        ExecutionError: If evaluation fails
    """
    logger.info("Processing evaluation", model=request.model)
    # Implementation here
```

## Testing

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test HTTP endpoints end-to-end

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/eval_hub

# Run specific test categories
pytest -m unit
pytest -m integration

# Run tests for specific modules
pytest tests/unit/services/
```

### Test Requirements

1. **New Features**: Must include unit and integration tests
2. **Bug Fixes**: Must include regression tests
3. **Coverage**: Maintain >80% test coverage
4. **Performance**: Include performance tests for critical paths

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch
from eval_hub.services.executor import ExecutionService

class TestExecutionService:
    """Test cases for ExecutionService."""

    @pytest.fixture
    def service(self):
        return ExecutionService()

    async def test_execute_evaluation_success(self, service):
        """Test successful evaluation execution."""
        # Test implementation

    async def test_execute_evaluation_timeout(self, service):
        """Test evaluation timeout handling."""
        # Test implementation
```

## Pull Request Process

### Before Submitting

1. **Rebase on Main**: Ensure your branch is up-to-date
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run Full Test Suite**
   ```bash
   pytest
   pre-commit run --all-files
   ```

3. **Update Documentation**: Include relevant documentation updates

### PR Template

When creating a pull request, include:

**Description**
- Brief summary of changes
- Link to related issue(s)

**Type of Change**
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] New tests added for new functionality

**Checklist**
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No new warnings introduced

### Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **OWNERS Assignment**: Project maintainers automatically assigned as reviewers ([OWNERS Guide](.github/OWNERS.md))
3. **Code Review**: Component experts and maintainer approval required
4. **Testing**: Reviewers may test functionality manually
5. **Documentation**: Ensure documentation is clear and complete

## Issue Reporting

We use a structured labeling system with `kind/*` prefixes to categorize issues. See our [Label Guide](.github/LABELS.md) for the complete labeling scheme and setup instructions.

### Bug Reports

When reporting bugs, include:

```markdown
**Description**: Clear description of the issue

**To Reproduce**: Steps to reproduce the behavior
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**: What you expected to happen

**Environment**:
- OS: [e.g. Ubuntu 22.04]
- Python Version: [e.g. 3.12]
- eval-hub Version: [e.g. 0.1.1]
- Kubernetes Version: [e.g. 1.28]

**Additional Context**: Any additional information
```

### Feature Requests

For feature requests, include:

```markdown
**Problem Statement**: What problem does this solve?

**Proposed Solution**: Describe your proposed solution

**Alternatives**: Any alternative solutions considered

**Use Case**: Real-world scenario where this would be useful

**Implementation Notes**: Technical considerations or constraints
```

## Documentation

### Types of Documentation

1. **API Documentation**: OpenAPI specs and endpoint documentation
2. **User Guides**: How-to guides for common tasks
3. **Developer Docs**: Architecture and implementation details
4. **Deployment Guides**: Kubernetes/OpenShift deployment instructions

### Documentation Standards

- **Clarity**: Write for your intended audience
- **Examples**: Include practical examples
- **Accuracy**: Keep documentation in sync with code
- **Structure**: Use consistent formatting and organization

### Building Documentation

```bash
# Generate OpenAPI documentation
python -m eval_hub.main --generate-openapi

# Build documentation locally (if using docs framework)
cd docs/
make html
```

## Community

### Communication Channels

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for general questions
- **Pull Requests**: GitHub PRs for code contributions

### Getting Help

1. **Check Existing Issues**: Search for similar problems
2. **Read Documentation**: Review README and API docs
3. **Ask Questions**: Create a GitHub Discussion
4. **Join Community**: Engage with other contributors

### Recognition

Contributors are recognized in:
- **Release Notes**: Major contributions highlighted
- **Contributors**: GitHub automatically tracks contributors
- **Acknowledgments**: Special recognition for significant contributions

## License

By contributing to Eval Hub, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Eval Hub! Your efforts help improve ML evaluation capabilities for the entire community.
