# Contributing to PyZaplinePlus

Thank you for your interest in contributing to PyZaplinePlus! This document provides guidelines and information for contributors.

## 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a development environment**
4. **Make your changes**
5. **Test your changes**
6. **Submit a pull request**

## 🛠 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Setting Up the Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/PyZapline_plus.git
cd PyZapline_plus

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import pyzaplineplus; print('Installation successful!')"
```

### Development Dependencies

The `[dev]` extra includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `ruff` - Linting and code quality
- `mkdocs` - Documentation
- `mkdocs-material` - Documentation theme

## 🧪 Testing

We maintain high test coverage and all contributions should include appropriate tests.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=pyzaplineplus

# Run specific test file
pytest tests/test_core.py

# Run tests with verbose output
pytest -v

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test function names: `test_feature_behavior_condition`
- Include both unit tests and integration tests
- Test edge cases and error conditions
- Add performance tests for computationally intensive features

Example test structure:
```python
def test_zapline_plus_basic_functionality():
    """Test basic line noise removal functionality."""
    # Arrange
    data, fs = create_sample_data_with_noise()
    
    # Act
    clean_data, config, analytics, plots = zapline_plus(data, fs)
    
    # Assert
    assert clean_data.shape == data.shape
    assert analytics['proportion_removed_noise'] > 0.1
```

## 🎨 Code Style

We use automated code formatting and linting tools.

### Code Formatting

```bash
# Format code with black
black .

# Check formatting
black --check .
```

### Linting

```bash
# Run ruff linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Code Style Guidelines

- Follow PEP 8 conventions
- Use type hints for function signatures
- Write docstrings in NumPy style
- Keep line length to 88 characters (Black default)
- Use meaningful variable and function names

Example function with proper style:
```python
def detect_noise_frequency(
    pxx: np.ndarray, 
    f: np.ndarray, 
    threshold: float = 5.0
) -> Optional[float]:
    """
    Detect line noise frequency in power spectrum.
    
    Parameters
    ----------
    pxx : np.ndarray
        Power spectral density array.
    f : np.ndarray  
        Frequency vector.
    threshold : float, optional
        Detection threshold (default: 5.0).
        
    Returns
    -------
    Optional[float]
        Detected frequency or None if not found.
    """
    # Implementation here
    pass
```

## 📚 Documentation

Documentation is built with MkDocs and hosted on GitHub Pages.

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[dev]"

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add cross-references to related functions
- Update relevant documentation when changing APIs

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment information**:
   - Python version
   - PyZaplinePlus version
   - Operating system
   - Relevant package versions
5. **Minimal code example** that reproduces the issue

Use our bug report template:

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- Python version: 
- PyZaplinePlus version:
- OS:
- Other relevant packages:

## Code Example
```python
# Minimal code that reproduces the issue
```

## 💡 Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Provide examples** of how the feature would be used
4. **Consider backwards compatibility**

## 🔄 Pull Request Process

### Before Submitting

1. **Check tests pass**: `pytest`
2. **Check code style**: `black --check .` and `ruff check .`
3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** if appropriate

### Pull Request Guidelines

1. **Create a feature branch** from `main`
2. **Use descriptive commit messages**
3. **Keep changes focused** - one feature per PR
4. **Include tests** for new functionality
5. **Update documentation** as needed
6. **Ensure CI passes** before requesting review

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add support for multiple noise frequencies
fix(detection): handle edge case in frequency detection
docs(api): update zapline_plus function documentation
```

## 🏗 Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for integration
- `feature/name`: Feature development branches
- `fix/name`: Bug fix branches

### Typical Workflow

1. **Create feature branch** from `main`
2. **Develop and test** your changes
3. **Submit pull request** to `main`
4. **Address review feedback**
5. **Merge after approval**

## 📋 Release Process

For maintainers releasing new versions:

1. **Update version** in `pyzaplineplus/_version.py`
2. **Update CHANGELOG.md** with release notes
3. **Create release tag**: `git tag v1.0.0`
4. **Push tag**: `git push --tags`
5. **Create GitHub release** - this triggers PyPI publication

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what's best for the community
- Show empathy towards others

## ❓ Questions?

- **Documentation**: Check our [docs](https://pyzaplineplus.readthedocs.io/)
- **Issues**: Open a [GitHub issue](https://github.com/SinaEsmaeili/PyZaplinePlus/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/SinaEsmaeili/PyZaplinePlus/discussions)
- **Email**: Contact sina.esmaeili@umontreal.ca

## 🙏 Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes
- Documentation acknowledgments

Thank you for contributing to PyZaplinePlus! 🎉
