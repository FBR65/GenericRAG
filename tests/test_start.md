# Install Test dependecies
uv add --group test

# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py coverage

# Run specific tests
python run_tests.py specific tests/test_config.py

# Run code quality checks
python run_tests.py lint

# Direct pytest usage
python -m pytest tests/ --cov=src --cov-report=html