# ollama_eval_project/main.py
import ollama_client
from evaluator import run_evaluation
from utils.presentation import print_results_table

# Import benchmark classes
from benchmarks.example_benchmark import ExampleBenchmark
from benchmarks.mmlu_pro_adapter import MMLUProAdapter
from benchmarks.hle_adapter import HLEAdapter
from benchmarks.math_500_adapter import Math500Adapter
from benchmarks.live_code_bench_adapter import LiveCodeBenchAdapter

def main():
    print("Starting LLM Evaluation Project...")

    # 1. Discover available Ollama models
    print("\nChecking available Ollama models...")
    available_models, error = ollama_client.list_ollama_models()
    if error:
        print(f"Error fetching Ollama models: {error}")
        print("Please ensure Ollama is running and accessible.")
        return
    if not available_models:
        print("No local Ollama models found. Pull some models first (e.g., 'ollama pull llama3').")
        return
    
    print("Available models:", ", ".join(available_models))

    # --- Configuration ---
    # Specify which models you want to test from the available ones
    models_to_evaluate = ["llama3:8b", "hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:IQ2_S","qwen3:14b","deepseek-r1:8b","qwen3:8b"]
    # Or, to test all available models:
    # models_to_evaluate = available_models # Or select a subset

    # If you want to test specific models, filter them:
    # desired_models = ["llama2:latest", "phi3:latest"] # example
    # models_to_evaluate = [m for m in available_models if m in desired_models]
    # if not models_to_evaluate:
    #     print(f"None of the desired models ({', '.join(desired_models)}) are available locally.")
    #     print(f"Available models are: {', '.join(available_models)}")
    #     return


    # Instantiate the benchmarks you want to run
    benchmarks = [
        ExampleBenchmark(),
        #MMLUProAdapter(),       # Uses dummy data / logic
        #HLEAdapter(),           # Uses dummy data / logic
        #Math500Adapter(),       # Uses dummy data / logic
        #LiveCodeBenchAdapter()  # Uses dummy data / logic
    ]
    # --- End Configuration ---

    if not models_to_evaluate:
        print("No models selected or available for evaluation.")
        return

    print(f"\nSelected models for evaluation: {', '.join(models_to_evaluate)}")
    print(f"Selected benchmarks: {', '.join([b.get_name() for b in benchmarks])}")

    # 2. Run the evaluation
    results = run_evaluation(models_to_evaluate, benchmarks)

    # 3. Present the results
    if results:
        print_results_table(results)
    else:
        print("\nEvaluation completed, but no results were generated.")
        print("This might be due to errors, no models/benchmarks, or issues with questions.")

    print("\nLLM Evaluation Project finished.")
    print("\nIMPORTANT NOTICE:")
    print("The adapters for MMLU-Pro, HLE, MATH-500, and LiveCodeBench are stubs.")
    print("To use them effectively, you need to:")
    print("  1. Download their respective datasets.")
    print("  2. Implement the data loading logic in `get_questions()` for each adapter.")
    print("  3. Implement the specific evaluation logic in `evaluate()` for each adapter,")
    print("     often by calling official evaluation scripts or using robust answer parsing.")
    print("  4. For LiveCodeBench, setting up a secure code execution environment is crucial and complex.")

if __name__ == "__main__":
    main()