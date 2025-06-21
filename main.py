import os
import sys
import yaml
import inspect
import logging
import argparse
import importlib.util
import re


from evaluator import run_evaluation
from ollama_client import check_ollama_connection
from benchmarks.base_benchmark import BaseBenchmark
from reporters.base_reporter import BaseReporter



def setup_logging():
    """Configures logging to file and console."""
    # Check if handlers are already configured to avoid duplicates
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler("evaluation.log", mode='w',encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create a console handler to print logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on console
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

def load_modules_from_path(path, base_class):
    """Dynamically loads modules from a path and finds classes inheriting from a base class."""
    loaded_classes = {}
    for filename in os.listdir(path):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            module_path = os.path.join(path, filename)
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, base_class) and obj is not base_class:
                       loaded_classes[name] = obj
    return loaded_classes

def main():
    parser = argparse.ArgumentParser(description="A framework for benchmarking local LLMs via Ollama.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--models', nargs='+', help='Override models from config file. e.g., --models llama3:8b qwen2:7b')
    args = parser.parse_args()
    setup_logging()

    if not check_ollama_connection():
        sys.exit(1)
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    # --- Load Models ---
    models_to_evaluate = args.models if args.models else config.get('models_to_evaluate', [])
    if not models_to_evaluate:
        logging.error("No models specified in config or via CLI. Exiting.")
        sys.exit(1)

    # --- Discover and Load Benchmarks ---
    available_benchmarks = load_modules_from_path('benchmarks', BaseBenchmark)
    benchmarks_to_run = []
    for name, params in config.get('benchmarks', {}).items():
        if params.get('enabled') and name in available_benchmarks:
            cls = available_benchmarks[name]
            # Pass only the relevant params to the constructor
            instance_params = {k: v for k, v in params.items() if k != 'enabled'}
            benchmarks_to_run.append(cls(**instance_params))
            logging.info(f"Loaded benchmark: {name}")

    # --- Discover and Load Reporters ---
    available_reporters = load_modules_from_path('reporters', BaseReporter)
    reporters_to_run = []
    for name, params in config.get('reporters', {}).items():
        if params.get('enabled') and name in available_reporters:
            cls = available_reporters[name]
            reporters_to_run.append(cls(params))
            logging.info(f"Loaded reporter: {name}")
    
    # --- Run Evaluation ---
    logging.info(f"Starting evaluation for models: {', '.join(models_to_evaluate)}")
    results = run_evaluation(models_to_evaluate, benchmarks_to_run)


    # Present the results
    # --- Report Results ---
    if results:
        for reporter in reporters_to_run:
            reporter.report(results)
    else:
        logging.warning("Evaluation finished but produced no results.")

    logging.info("LLM Evaluation Project finished.")
 

if __name__ == "__main__":
    main()