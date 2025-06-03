# ollama_eval_project/main.py
import ollama_client
from evaluator import run_evaluation
from utils.presentation import print_results_table, save_results_to_html

# Import benchmark classes
from benchmarks.example_benchmark import ExampleBenchmark
from benchmarks.mmlu_pro_adapter import MMLUProAdapter # Ensure this is imported
from benchmarks.hle_adapter import HLEAdapter
from benchmarks.math_500_adapter import Math500Adapter
from benchmarks.live_code_bench_adapter import LiveCodeBenchAdapter

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
# Set to None to load all questions for the selected subjects.
MMLU_PRO_MAX_QUESTIONS_PER_SUBJECT = 10 # Example: Load only the first 10 questions per subject
# MMLU_PRO_MAX_QUESTIONS_PER_SUBJECT = None # To load all
# --- End MMLU-Pro Configuration ---

# --- End MMLU-Pro Configuration ---


def main():
    print("Starting LLM Evaluation Project...")

    # ... (Ollama model discovery logic) ...
    print("\nChecking available Ollama models...")
    available_models, error = ollama_client.list_ollama_models()
    if error:
        print(f"Error fetching Ollama models: {error}")
        return
    if not available_models:
        print("No local Ollama models found.")
        return
    print("Available models:", ", ".join(available_models))

    models_to_evaluate = ["llama3:8b", "hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:IQ2_S","qwen3:14b","deepseek-r1:8b","qwen3:8b"] # Or select a subset
    # models_to_evaluate = ["llama3:latest"] # For faster testing

    if not models_to_evaluate:
        print("No models selected or available for evaluation.")
        return

    # Instantiate the benchmarks
    benchmarks = [
        #ExampleBenchmark(),
        # Initialize MMLUProAdapter (no data_dir needed now)
        MMLUProAdapter(subjects=MMLU_PRO_SUBJECTS_TO_RUN, 
                       data_split=MMLU_PRO_DATA_SPLIT,
                       max_questions_per_subject=MMLU_PRO_MAX_QUESTIONS_PER_SUBJECT),
        #HLEAdapter(),
        #Math500Adapter(),
        #LiveCodeBenchAdapter()
    ]

    print(f"\nSelected models for evaluation: {', '.join(models_to_evaluate)}")
    print(f"Selected benchmarks: {', '.join([b.get_name() for b in benchmarks])}")

    # Run the evaluation
    results = run_evaluation(models_to_evaluate, benchmarks)

    # Present the results
    if results:
        print_results_table(results) 
        html_output_filename = "evaluation_results.html"
        save_results_to_html(results, html_output_filename)
    else:
        print("\nEvaluation completed, but no results were generated.")
        print("This might be due to errors, no models/benchmarks, or issues with questions.")
        save_results_to_html([], "evaluation_results_empty.html")

    print("\nLLM Evaluation Project finished.")
    # ... (rest of your IMPORTANT NOTICE)

if __name__ == "__main__":
    main()