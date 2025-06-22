# Local LLM Evaluation Framework

A flexible and extensible Python framework for benchmarking the performance, accuracy, and resource consumption of local language models running via the Ollama API.

This framework allows you to easily evaluate multiple models against various benchmarks on **local GPUs**, track performance metrics, and compare results in multiple formats.

## Key Features

- **Ollama Integration**: Directly connects to any local Ollama instance to run evaluations on all available models.
- **Extensive Resource Monitoring**: Captures detailed, per-model performance data, including:

  - Average CPU and RAM Utilization (%)
  - Average GPU Utilization and VRAM Usage (%) for NVIDIA GPUs
  - Total GPU Energy Consumption (in Watt-hours)

- **External & Centralized Configuration**: Easily configure models, benchmarks, and reporters via a central `config.yaml` file.
- **Command-Line Interface**: Override configurations (like the list of models to test) directly from the command line for quick experiments and scripting.
- **Automatic Module Discovery**: Add new benchmarks or reporters simply by dropping a file into the correct directory. No code changes are needed in the main application.
- **Modular Benchmarks**: Add new benchmarks by inheriting from a simple base class.
- **Deterministic & Reproducible Results**: Control model generation with parameters like temperature and seed to ensure consistent and reproducible outputs.
- **Advanced Logging** :
  - Clean, informative console output for high-level progress.
  - Verbose DEBUG level logs saved to evaluation.log for detailed troubleshooting.
  - Automatic log rotation to prevent log files from growing indefinitely
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
    git clone https://github.com/kvt-bk/LLMPCBench.git
    cd LLMPCBench
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
