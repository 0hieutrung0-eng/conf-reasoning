# CLAUDE.md - Coding Guidelines for verl

## Build/Test/Lint Commands
- **Install**: `pip install -e .` or `pip install -e '.[test]'` with test dependencies
- **Format code**: `bash scripts/format.sh` (uses yapf with Google style)
- **Run tests**: `pytest tests/` or specific test `pytest tests/path/to/test_file.py::test_function`
- **Check imports**: `python tests/sanity/test_import.py`
- **Build package**: `python setup.py bdist_wheel`

## Code Style Guidelines
- **Formatting**: Google style (yapf) with 120 column limit, 4-space indentation
- **Imports**: Standard library first, then third-party, then local modules (alphabetically)
- **Typing**: Use type hints for function parameters and return values
- **Docstrings**: Include for all public functions/classes ("""Description...""")
- **Error handling**: Use specific exceptions, include context messages
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **License**: Include Apache 2.0 license at top of each file
- **Unit tests**: Write tests for new functionality, aim for high coverage
- **Ray usage**: Follow Ray conventions for remote functions and actors