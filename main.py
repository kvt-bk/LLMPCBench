import logging
import sys
import ollama_client
from evaluator import run_evaluation
from utils.presentation import print_results_table, save_results_to_html

# Import benchmark classes
from benchmarks.example_benchmark import ExampleBenchmark
from benchmarks.mmlu_pro_adapter import MMLUProAdapter 
#from benchmarks.hle_adapter import HLEAdapter
#from benchmarks.math_500_adapter import Math500Adapter
#from benchmarks.live_code_bench_adapter import LiveCodeBenchAdapter

# --- Configuration for MMLU-Pro ---
# Specify a few subjects (Hugging Face configurations) to run for quicker testing, or None for all.
# These should match the configuration names on Hugging Face for TIGER-Lab/MMLU-Pro.
""" MMLU_PRO_SUBJECTS_TO_RUN = [
    "professional_psychology", 
    "moral_scenarios", 
    "us_foreign_policy",
    "abstract_algebra" # Example list
]   """
# Set to None to attempt loading all available subjects for MMLU-Pro
MMLU_PRO_SUBJECTS_TO_RUN = None 
MMLU_PRO_DATA_SPLIT = "test" # Can be "test", "validation", or "dev"
# New: Set the maximum number of questions per MMLU-Pro subject
MMLU_PRO_PERCENTAGE_PER_SUBJECT = 1.0 # Example: Load 1% of questions per subject for a quick test
# --- End MMLU-Pro Configuration ---

models_to_evaluate = ["hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:IQ2_S","qwen3:14b","deepseek-r1:8b","qwen3:8b","llama3:8b"]
# Instantiate the benchmarks
benchmarks = [
        #ExampleBenchmark(),
        # Initialize MMLUProAdapter (no data_dir needed now)
        MMLUProAdapter(subjects=MMLU_PRO_SUBJECTS_TO_RUN, 
                       data_split=MMLU_PRO_DATA_SPLIT,
                       percentage_per_subject=MMLU_PRO_PERCENTAGE_PER_SUBJECT)
        #HLEAdapter(),
        #Math500Adapter(),
        #LiveCodeBenchAdapter()
    ]
# --- End of Configuration ---

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

def main():
    setup_logging()
    logging.info("Starting LLM Evaluation...")

    if not models_to_evaluate:
        logging.error("No models selected or available for evaluation.")
        return

    logging.info(f"\nSelected models for evaluation: {', '.join(models_to_evaluate)}")
    logging.info(f"Selected benchmarks: {', '.join([b.get_name() for b in benchmarks])}")

    # Run the evaluation
    results = run_evaluation(models_to_evaluate, benchmarks)

    # Present the results
    if results:
        print_results_table(results) 
        html_output_filename = "evaluation_results.html"
        save_results_to_html(results, html_output_filename)
    else:
        logging.warning("\nEvaluation completed, but no results were generated.")
        logging.warning("This might be due to errors, no models/benchmarks, or issues with questions.")
        save_results_to_html([], "evaluation_results_empty.html")

    logging.info("\nLLM Evaluation Project finished.")
    

if __name__ == "__main__":
    main()