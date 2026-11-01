# Contributing to SLM Project

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [issue tracker](https://github.com/bharqav/emotional-slm/issues) to see if the bug has already been reported
2. Collect relevant information (environment, Python version, error messages, steps to reproduce)

When filing a bug report, include:
- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior** vs **actual behavior**
- **Environment details**: Python version, OS, PyTorch version
- **Logs and error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:
1. Use a clear title and description
2. Provide a specific example of how the enhancement would work
3. Explain why this enhancement would be useful

### Pull Requests

1. **Fork the repo** and create a new branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Ensure all tests pass**: `make test` or `python -m pytest tests/ -v`
5. **Run linting**: `make lint` or `ruff check .`
6. **Run type checking**: `make typecheck` or `mypy .`
7. **Update documentation** if needed
8. **Submit a pull request** with a clear description of the changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/slm-comparison.git
cd slm-comparison

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev,server,qlora]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

# Run full pipeline
python run_all.py
```

## Code Style

- **Python**: Follow PEP 8, use type hints, max line length 88
- **Formatting**: `ruff format .`
- **Linting**: `ruff check .`
- **Type checking**: `mypy .` (strict mode)
- **Commits**: Use clear, descriptive commit messages

## Testing

- All new features should include tests
- Tests should be deterministic (use `--seed` for generation tests)
- Aim for >80% coverage on new code
- Run `make test` before submitting PR

## Documentation

- Update `README.md` for user-facing changes
- Update docstrings for API changes
- Add entries to `CHANGELOG.md` for significant changes

## Review Process

1. Maintainers will review your PR as soon as possible
2. Address any requested changes
3. Once approved, a maintainer will merge your PR

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.
