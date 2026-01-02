# AGENTS.md

Guidelines for agentic coding agents working in the Athene repository.

## Project Overview

Athene is a Python project using Manim for mathematical animations. Python 3.13+ with uv for dependency management.

## Build/Lint/Test Commands

### Environment Setup
```bash
uv sync                      # Install dependencies
source .venv/bin/activate    # Activate venv (optional, uv run handles this)
```

### Running Manim Scenes
```bash
# Using Makefile (edit SCENE and SOURCE vars as needed)
make preview                 # Quick preview (low quality)
make render                  # High quality render
make clean                   # Remove media output

# Direct commands
uv run manim -pql script.py SceneName    # Preview low quality
uv run manim -qh script.py SceneName     # Render high quality
uv run manim -qk script.py SceneName     # Render 4K quality
```

### Testing
```bash
uv run pytest                                    # Run all tests
uv run pytest tests/test_file.py                 # Run specific file
uv run pytest tests/test_file.py::test_func      # Run single test
uv run pytest tests/test_file.py::TestClass      # Run test class
uv run pytest -k "pattern"                       # Run tests matching pattern
uv run pytest -x                                 # Stop on first failure
uv run pytest -v                                 # Verbose output
uv run pytest --cov=athene                       # With coverage
```

### Linting and Formatting
```bash
uv run ruff check .          # Check for lint errors
uv run ruff check --fix .    # Auto-fix lint errors
uv run ruff format .         # Format code
uv run ruff format --check . # Check formatting without changes
uv run mypy .                # Type checking
```

### Git Operations
```bash
cz commit                    # Commit with conventional commits (pre-commit configured)
cz check                     # Validate commit message
```

## Code Style Guidelines

### Import Organization
```python
# 1. Standard library
import os
from typing import Optional

# 2. Third-party
import numpy as np
from manim import *

# 3. Local
from athene.utils import helper
```

### Type Hints
- Use type hints for all function parameters and return values
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (Python 3.10+)
- Use `from typing import` only for complex types (TypeVar, Protocol, etc.)

### Naming Conventions
| Element     | Convention  | Example              |
|-------------|-------------|----------------------|
| Files       | snake_case  | `animation_scene.py` |
| Classes     | PascalCase  | `TowerOfHanoi`       |
| Functions   | snake_case  | `create_animation`   |
| Variables   | snake_case  | `disk_count`         |
| Constants   | UPPER_CASE  | `DEFAULT_COLOR`      |

### Formatting
- Line length: 88 characters (ruff/black default)
- Use f-strings for string formatting
- Use trailing commas in multi-line collections

### Error Handling
- Use specific exception types, not bare `except:`
- Validate inputs in public functions
- Use logging over print statements for debugging

### Documentation
- Use docstrings for all public functions and classes
- Follow Google docstring style:
```python
def move_disk(source: int, target: int) -> None:
    """Move a disk from source peg to target peg.

    Args:
        source: Index of the source peg (0-2).
        target: Index of the target peg (0-2).

    Raises:
        ValueError: If peg indices are out of range.
    """
```

### Manim-Specific Guidelines
- Import: `from manim import *`
- Scene classes inherit from `Scene` or specialized scene types
- Use Manim color constants: `RED`, `BLUE`, `GREEN`, etc.
- Use `self.play()` for animations, `self.wait()` for pauses
- Organize complex scenes into methods for readability

## Project Structure
```
athene/
├── athene/           # Main package
│   ├── __init__.py
│   ├── scenes/       # Manim scene files
│   └── utils/        # Helper utilities
├── tests/            # Test files (mirror src structure)
├── media/            # Manim output (gitignored)
├── pyproject.toml    # Project config and dependencies
└── Makefile          # Common commands
```

## Dependencies

```bash
uv add package_name          # Add runtime dependency
uv add --dev package_name    # Add dev dependency
uv remove package_name       # Remove dependency
uv sync                      # Sync from lock file
```

## Git Workflow

- Use conventional commits (commitizen configured via pre-commit)
- Commit types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Branch naming: `feature/description`, `fix/description`

## Quick Reference

| Task                    | Command                              |
|-------------------------|--------------------------------------|
| Install deps            | `uv sync`                            |
| Preview scene           | `make preview`                       |
| Render HD               | `make render`                        |
| Run all tests           | `uv run pytest`                      |
| Run single test         | `uv run pytest path::test_name`      |
| Lint                    | `uv run ruff check .`                |
| Format                  | `uv run ruff format .`               |
| Type check              | `uv run mypy .`                      |
| Commit                  | `cz commit`                          |
