# Local LLM Evaluation Framework

A flexible and extensible Python framework for benchmarking the performance, accuracy, and resource consumption of local language models running via the Ollama API.

This framework allows you to easily evaluate multiple models against various benchmarks, track performance metrics, and compare results in multiple formats.

## Key Features

- **Ollama Integration**: Directly connects to your local Ollama instance.
- **External Configuration**: Easily configure models, benchmarks, and reporters via a central `config.yaml` file.
- **Command-Line Interface**: Override configurations and run specific experiments directly from the CLI.
- **Automatic Discovery**: New benchmarks and reporters are automatically discovered—just drop them in the right directory.
- **Modular Benchmarks**: Add new benchmarks by inheriting from a simple base class.
- **Resource Monitoring**: Tracks average CPU, RAM, and GPU utilization for each model's evaluation run.
- **Energy Tracking**: Measures total GPU energy consumption (in Watt-hours) for NVIDIA GPUs.
- **Multiple Reporters**: Get results as a console table and a cumulative HTML report out-of-the-box.
- **CI/CD Ready**: Includes a GitHub Actions workflow for automated linting and testing.

## Getting Started

### Prerequisites

1.  **Python**: Python 3.9 or newer.
2.  **Ollama**: [Ollama installed](https://ollama.com/download) and running.
3.  **Ollama Models**: Pull the models you wish to evaluate (e.g., `ollama pull llama3:8b`).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Configure Your Run

Edit the `config.yaml` file to define which models, benchmarks, and reporters you want to use.

```yaml
# example config.yaml
models_to_evaluate:
  - "llama3:8b"
  - "qwen2:7b"

benchmarks:
  mmlu_pro:
    enabled: true
    percentage_per_subject: 1.0 # Use 1% of questions for a quick test
```
