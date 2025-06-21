# Contributing to the LLM Evaluation Framework

We welcome contributions from the community! Whether you're fixing a bug, adding a new benchmark, or improving documentation, your help is appreciated.

## Setting Up a Development Environment

1.  **Fork and Clone:** Fork the repository to your own GitHub account and then clone it to your local machine.
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:** Install the required packages, including development tools.
    ```bash
    pip install -r requirements.txt
    ```

## How to Add a New Benchmark

Adding a new benchmark is designed to be a simple "drop-in" process:

1.  Create a new file in the `benchmarks/` directory (e.g., `my_awesome_benchmark.py`).
2.  Inside this file, create a class that inherits from `BaseBenchmark`.
3.  Implement the required `get_questions()` and `evaluate()` methods.
4.  The framework will automatically discover and make your new benchmark available to be enabled in `config.yaml`.

## Submitting a Pull Request

1.  Create a new branch for your feature or bug fix.
2.  Make your changes.
3.  If adding a new feature, consider adding a test for it in the `tests/` directory.
4.  Ensure your code passes basic checks by running `pytest`.
5.  Push your branch to your fork and open a pull request against our `main` branch.
6.  Provide a clear description of your changes in the pull request.

Thank you for contributing!
