# Contributing to LOLA OS

Thank you for your interest in contributing to LOLA OS! This guide outlines how to contribute to the project, ensuring alignment with our goals of Developer Sovereignty, EVM-Native capabilities, and Radical Reliability.

## Getting Started

1. **Fork the Repository**:
   - Fork `https://github.com/Levi-Chinecherem/lola-os` on GitHub.
   - Clone your fork:
     ```bash
     git clone git@github.com:<your-username>/lola-os.git
     cd lola-os
     ```

2. **Set Up the Environment**:
   - Install Poetry: `pip install poetry`
   - Install dependencies:
     ```bash
     poetry install
     ```
   - Set PYTHONPATH:
     ```bash
     export PYTHONPATH=$(pwd)/python:$PYTHONPATH
     ```

3. **Run Tests**:
   - Verify the setup:
     ```bash
     poetry run pytest tests/ -v
     ```

## Contribution Process

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make Changes**:
   - Follow the project structure in `README.md`.
   - Add tests in `tests/` for new features.
   - Update documentation in `docs/` if needed.

3. **Commit Changes**:
   - Use clear commit messages:
     ```bash
     git commit -m "Add <feature>: <description>"
     ```

4. **Push and Create Pull Request**:
   ```bash
   git push origin feature/your-feature
   ```
   - Open a pull request on `https://github.com/Levi-Chinecherem/lola-os`.

## Code Style

- Follow PEP 8 for Python code.
- Use type hints where applicable.
- Include docstrings in Google/NumPy format.

## Reporting Issues

- Open an issue on GitHub with a clear description, steps to reproduce, and expected behavior.

## Contact

Email: Levi Chinecherem Chidi <lchinecherem2018@gmail.com>