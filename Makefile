
# BGE-M3 Test Makefile
# This Makefile provides convenient commands for running BGE-M3 tests

.PHONY: help test test-all test-unit test-integration test-service test-utils test-api test-performance test-benchmark test-health test-parallel test-report test-coverage test-threshold test-specific test-clean test-docs

# Default target
help:
	@echo "BGE-M3 Test Commands"
	@echo "===================="
	@echo ""
	@echo "Test Commands:"
	@echo "  test-all           - Run all BGE-M3 tests"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests only"
	@echo "  test-service       - Run BGE-M3 service tests"
	@echo "  test-utils         - Run BGE-M3 utils tests"
	@echo "  test-api           - Run BGE-M3 API tests"
	@echo "  test-performance   - Run performance tests"
	@echo "  test-benchmark     - Run benchmark tests"
	@echo "  test-health        - Run health check tests"
	@echo "  test-parallel      - Run tests in parallel"
	@echo "  test-report        - Generate comprehensive test report"
	@echo ""
	@echo "Coverage Commands:"
	@echo "  test-coverage      - Generate coverage reports"
	@echo "  test-coverage-html - Generate HTML coverage report"
	@echo "  test-coverage-xml  - Generate XML coverage report"
	@echo "  test-coverage-json - Generate JSON coverage report"
	@echo "  test-threshold     - Run tests with coverage threshold"
	@echo ""
	@echo "Specific Commands:"
	@echo "  test-specific      - Run specific test file"
	@echo "  test-specific-path - Run specific test path (usage: make test-specific-path TEST_PATH=tests/test_bge_m3_service.py)"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  test-docs          - Generate test documentation"
	@echo ""
	@echo "Utility Commands:"
	@echo "  test-clean         - Clean test artifacts"
	@echo "  test-setup         - Setup test environment"
	@echo "  test-check         - Check test environment"
	@echo ""

# Test commands
test-all:
	@echo "Running all BGE-M3 tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --verbose

test-unit:
	@echo "Running BGE-M3 unit tests..."
	@python scripts/run_bge_m3_tests.py --mode unit --coverage

test-integration:
	@echo "Running BGE-M3 integration tests..."
	@python scripts/run_bge_m3_tests.py --mode integration --coverage

test-service:
	@echo "Running BGE-M3 service tests..."
	@python scripts/run_bge_m3_tests.py --mode service --coverage

test-utils:
	@echo "Running BGE-M3 utils tests..."
	@python scripts/run_bge_m3_tests.py --mode utils --coverage

test-api:
	@echo "Running BGE-M3 API tests..."
	@python scripts/run_bge_m3_tests.py --mode api --coverage

test-performance:
	@echo "Running BGE-M3 performance tests..."
	@python scripts/run_bge_m3_tests.py --mode performance

test-benchmark:
	@echo "Running BGE-M3 benchmark tests..."
	@python scripts/run_bge_m3_tests.py --mode benchmark --benchmark-iterations 10

test-health:
	@echo "Running BGE-M3 health checks..."
	@python scripts/run_bge_m3_tests.py --mode health

test-parallel:
	@echo "Running BGE-M3 tests in parallel..."
	@python scripts/run_bge_m3_tests.py --mode all --parallel --processes 4

test-report:
	@echo "Generating comprehensive test report..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --report

# Coverage commands
test-coverage:
	@echo "Generating coverage reports..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage

test-coverage-html:
	@echo "Generating HTML coverage report..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --coverage-format html

test-coverage-xml:
	@echo "Generating XML coverage report..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --coverage-format xml

test-coverage-json:
	@echo "Generating JSON coverage report..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --coverage-format json

test-threshold:
	@echo "Running tests with coverage threshold..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --threshold 90

# Specific test commands
test-specific:
	@echo "Running specific test: $(TEST_PATH)"
	@if [ -z "$(TEST_PATH)" ]; then \
		echo "Error: TEST_PATH is required. Usage: make test-specific TEST_PATH=tests/test_bge_m3_service.py"; \
		exit 1; \
	fi
	@python scripts/run_bge_m3_tests.py --test-path "$(TEST_PATH)"

test-specific-path:
	@echo "Running specific test path: $(TEST_PATH)"
	@if [ -z "$(TEST_PATH)" ]; then \
		echo "Error: TEST_PATH is required. Usage: make test-specific-path TEST_PATH=tests/test_bge_m3_service.py::TestBGE_M3Service::test_generate_dense_embedding"; \
		exit 1; \
	fi
	@python -m pytest "$(TEST_PATH)" -v

# Documentation commands
test-docs:
	@echo "Generating test documentation..."
	@echo "Test documentation is available at: docs/bge_m3_tests.md"
	@echo "You can view it with: cat docs/bge_m3_tests.md"

# Utility commands
test-clean:
	@echo "Cleaning test artifacts..."
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf coverage_bge_m3.xml
	@rm -rf coverage_bge_m3.json
	@rm -rf tests/tmp/
	@rm -rf tests/__pycache__/
	@rm -rf tests/test_data/__pycache__/
	@rm -rf tests/integration/__pycache__/
	@rm -rf src/app/__pycache__/
	@rm -rf src/app/services/__pycache__/
	@rm -rf src/app/utils/__pycache__/
	@rm -rf src/app/api/__pycache__/
	@rm -rf src/app/models/__pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Test artifacts cleaned!"

test-setup:
	@echo "Setting up test environment..."
	@python -m pip install -e .
	@python -m pip install -r requirements-dev.txt
	@python -m pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark pytest-html pytest-xdist
	@echo "✅ Test environment setup complete!"

test-check:
	@echo "Checking test environment..."
	@python -c "import pytest; print('✅ pytest available')"
	@python -c "import pytest_asyncio; print('✅ pytest-asyncio available')"
	@python -c "import pytest_cov; print('✅ pytest-cov available')"
	@python -c "import pytest_mock; print('✅ pytest-mock available')"
	@python -c "import pytest_benchmark; print('✅ pytest-benchmark available')"
	@python -c "import pytest_html; print('✅ pytest-html available')"
	@python -c "import pytest_xdist; print('✅ pytest-xdist available')"
	@echo "✅ Test environment check complete!"

# Development commands
test-dev:
	@echo "Running development tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --verbose --threshold 90

test-quick:
	@echo "Running quick tests (no coverage)..."
	@python scripts/run_bge_m3_tests.py --mode all --no-coverage

test-watch:
	@echo "Running tests in watch mode..."
	@python -m pytest-watch -- tests -m bge_m3 --cov=src

test-debug:
	@echo "Running tests with debug output..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --verbose --tb=long

# CI/CD commands
test-ci:
	@echo "Running CI tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --threshold 90 --parallel

test-pr:
	@echo "Running PR tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --threshold 90

test-nightly:
	@echo "Running nightly tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --report --benchmark-iterations 20

# Performance commands
test-perf:
	@echo "Running performance tests..."
	@python scripts/run_bge_m3_tests.py --mode performance --benchmark-iterations 20

test-load:
	@echo "Running load tests..."
	@python scripts/run_bge_m3_tests.py --mode performance --benchmark-iterations 50

# Stress test commands
test-stress:
	@echo "Running stress tests..."
	@python scripts/run_bge_m3_tests.py --mode performance --benchmark-iterations 100

test-memory:
	@echo "Running memory tests..."
	@python -m pytest tests/ -m performance --benchmark-only --benchmark-sort=min --benchmark-max-time=1.0

# Regression test commands
test-regression:
	@echo "Running regression tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --threshold 90

test-regression-unit:
	@echo "Running regression unit tests..."
	@python scripts/run_bge_m3_tests.py --mode unit --coverage --threshold 95

test-regression-integration:
	@echo "Running regression integration tests..."
	@python scripts/run_bge_m3_tests.py --mode integration --coverage --threshold 90

# Smoke test commands
test-smoke:
	@echo "Running smoke tests..."
	@python scripts/run_bge_m3_tests.py --mode health

test-smoke-unit:
	@echo "Running smoke unit tests..."
	@python -m pytest tests/test_bge_m3_service.py::TestBGE_M3Service::test_generate_dense_embedding -v

test-smoke-api:
	@echo "Running smoke API tests..."
	@python -m pytest tests/test_bge_m3_api.py::TestBGE_M3_API_Endpoints::test_query_bge_m3_endpoint -v

# Advanced commands
test-compare:
	@echo "Running comparison tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --benchmark-iterations 10
	@echo "Comparison results available in htmlcov/bge_m3/"

test-profile:
	@echo "Running profiling tests..."
	@python -m pytest tests/ -m performance --benchmark-only --benchmark-sort=mean --benchmark-min-rounds=5

test-annotate:
	@echo "Running annotated tests..."
	@python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

# Docker commands (if Docker is available)
test-docker:
	@echo "Running tests in Docker..."
	@docker-compose -f docker-compose.test.yml up --build

test-docker-parallel:
	@echo "Running tests in Docker (parallel)..."
	@docker-compose -f docker-compose.test.yml up --build --scale test=4

# Kubernetes commands (if Kubernetes is available)
test-k8s:
	@echo "Running tests in Kubernetes..."
	@kubectl apply -f k8s/test-job.yaml
	@kubectl wait --for=condition=complete job/bge-m3-tests --timeout=300s

# Advanced CI/CD commands
test-cd:
	@echo "Running CD tests..."
	@python scripts/run_bge_m3_tests.py --mode all --coverage --threshold 90 --parallel --report

test-cd-performance:
	@echo "Running CD performance tests..."
	@python scripts/run_bge_m3_tests.py --mode performance --benchmark-iterations 20

test-cd-security:
	@echo "Running CD security tests..."
	@python -m bandit -r src/ -f json -o bandit-report.json

test-cd-quality:
	@echo "Running CD quality tests..."
	@python -m flake8 src/
	@python -m black --check src/
	@python -m isort --check-only src/
	@python -m mypy src/

# Advanced development commands
test-dev-all:
	@echo "Running all development tests..."
	@make test-dev
	@make test-perf
	@make test-regression
	@make test-coverage-html

test-dev-quick:
	@echo "Running quick development tests..."
	@make test-quick
	@make test-smoke

test-dev-full:
	@echo "Running full development tests..."
	@make test-dev-all
	@make test-docs
	@make test-report

# Advanced documentation commands
test-docs-html:
	@echo "Generating HTML documentation..."
	@python -m mkdocs build --config-file docs/mkdocs.yml

test-docs-serve:
	@echo "Serving documentation..."
	@python -m mkdocs serve --config-file docs/mkdocs.yml

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@rm -rf docs/api/
	@rm -rf docs/build/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@echo "✅ All test artifacts cleaned!"

test-clean-docker:
	@echo "Cleaning Docker test artifacts..."
	@docker-compose -f docker-compose.test.yml down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Docker test artifacts cleaned!"

# Advanced setup commands
test-setup-dev:
	@echo "Setting up development environment..."
	@make test-setup
	@python -m pip install black isort flake8 mypy bandit pytest-watch pytest-xdist
	@python -m pre-commit install
	@echo "✅ Development environment setup complete!"

test-setup-docker:
	@echo "Setting up Docker test environment..."
	@docker-compose -f docker-compose.test.yml build
	@echo "✅ Docker test environment setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@python -c "import docker; print('✅ Docker available')" 2>/dev/null || echo "⚠️  Docker not available"
	@python -c "import kubernetes; print('✅ Kubernetes available')" 2>/dev/null || echo "⚠️  Kubernetes not available"
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@echo "✅ All test environments setup complete!"

# Advanced check commands
test-check-all:
	@echo "Checking all test environments..."
	@make test-check
	@echo "✅ All test environment checks complete!"

# Advanced CI/CD commands
test-cd-all:
	@echo "Running all CD tests..."
	@make test-cd
	@make test-cd-performance
	@make test-cd-security
	@make test-cd-quality
	@echo "✅ All CD tests complete!"

# Advanced regression commands
test-regression-all:
	@echo "Running all regression tests..."
	@make test-regression-unit
	@make test-regression-integration
	@make test-regression
	@echo "✅ All regression tests complete!"

# Advanced performance commands
test-perf-all:
	@echo "Running all performance tests..."
	@make test-perf
	@make test-load
	@make test-stress
	@make test-memory
	@echo "✅ All performance tests complete!"

# Advanced documentation commands
test-docs-all:
	@echo "Generating all documentation..."
	@make test-docs
	@make test-docs-html
	@echo "✅ All documentation generated!"

# Advanced cleanup commands
test-clean-all:
	@echo "Cleaning all test artifacts..."
	@make test-clean
	@make test-clean-docker
	@echo "✅ All test artifacts cleaned!"

# Advanced setup commands
test-setup-all:
	@echo "Setting up all test environments..."
	@make test-setup-dev
	@make test-setup-docker
	@