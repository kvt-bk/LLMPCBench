# ollama_eval_project/evaluator.py
import time
from ollama_client import get_ollama_response
from benchmarks.base_benchmark import BaseBenchmark

def run_evaluation(models_to_test: list[str], benchmarks_to_run: list[BaseBenchmark]):
    """
    Runs the specified benchmarks on the specified Ollama models.

    Args:
        models_to_test (list[str]): A list of Ollama model names.
        benchmarks_to_run (list[BaseBenchmark]): A list of benchmark objects.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the results for a model-benchmark pair.
              {'model': str, 'benchmark': str, 'score': float, 'avg_tokens_s': float | None}
    """
    all_results = []

    if not models_to_test:
        print("No models specified for evaluation.")
        return []
    if not benchmarks_to_run:
        print("No benchmarks specified for evaluation.")
        return []

    for benchmark in benchmarks_to_run:
        benchmark_name = benchmark.get_name()
        print(f"  Running Benchmark: {benchmark_name}...")
        questions = benchmark.get_questions()
        if not questions:
            print(f"    No questions found for benchmark {benchmark_name}. Skipping.")
            continue
        for model_name in models_to_test:
            print(f"\n--- Evaluating Model: {model_name} ---")
            total_score = 0
            num_questions = len(questions)
            successful_evals = 0
            
            all_tps = [] # To store tokens/second for each question

            for i, q_data in enumerate(questions):
                prompt = q_data.get("prompt")
                if not prompt:
                    print(f"    Question {i+1} has no prompt. Skipping.")
                    num_questions -=1 # Adjust count of valid questions
                    continue

                print(f"    Querying model for question {i+1}/{len(questions)}...")
                print("Prompt is "+prompt)
                response_text, tps, error = get_ollama_response(model_name, prompt)

                if error:
                    print(f"      Error getting response for question {q_data.get('id', i+1)}: {error}")
                    # Optionally decide if this question should count as 0 or be skipped for scoring
                    # For now, let's skip it from score calculation if there's an error fetching response
                    num_questions -=1 
                    continue
                
                if tps is not None:
                    all_tps.append(tps)

                question_score = benchmark.evaluate(response_text, q_data)
                
                if question_score is not None:
                    total_score += question_score
                    successful_evals += 1
                    print(f"      Question {q_data.get('id', i+1)} - Score: {question_score:.2f}" + (f", TPS: {tps:.2f}" if tps else ""))
                else:
                    print(f"      Question {q_data.get('id', i+1)} - Could not be evaluated.")
                    # num_questions -=1 # If unevaluable questions shouldn't count towards the average

            avg_score_percent = (total_score / successful_evals) * 100 if successful_evals > 0 else 0.0
            avg_tps = sum(all_tps) / len(all_tps) if all_tps else None

            print(f"  Benchmark {benchmark_name} Summary for {model_name}:")
            print(f"    Average Score: {avg_score_percent:.2f}% (over {successful_evals} evaluated questions)")
            if avg_tps is not None:
                print(f"    Average Tokens/Second: {avg_tps:.2f}")
            else:
                print(f"    Average Tokens/Second: N/A")

            all_results.append({
                "model": model_name,
                "benchmark": benchmark_name,
                "score": avg_score_percent if successful_evals > 0 else None, # Use None if no questions could be scored
                "avg_tokens_s": avg_tps
            })
            time.sleep(1) # Small delay between benchmarks

    return all_results