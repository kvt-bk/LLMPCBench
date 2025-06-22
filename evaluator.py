# ollama_eval_project/evaluator.py
import time
import logging
from ollama_client import get_ollama_response
from benchmarks.base_benchmark import BaseBenchmark
from utils.monitoring import SystemMonitor 


logger = logging.getLogger(__name__)

def run_evaluation(models_to_test: list[str], benchmarks_to_run: list[BaseBenchmark], model_options: dict ):
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
        logger.warning("No models specified for evaluation.")
        return []
    if not benchmarks_to_run:
        logger.warning("No benchmarks specified for evaluation.")
        return []

    for benchmark in benchmarks_to_run:
        benchmark_name = benchmark.get_name()
        logger.info(f"Running Benchmark: {benchmark_name}...")
        questions = benchmark.get_questions()
        if not questions:
            logger.warning(f"    No questions found for benchmark {benchmark_name}. Skipping.")
            continue
        for model_name in models_to_test:
            logger.info(f"\n--- Evaluating Model: {model_name} on {benchmark_name} ---")
            # <<< START monitor >>>
            monitor = SystemMonitor(interval=1)
            monitor.start()
            
            total_score = 0
            num_questions = len(questions)
            successful_evals = 0
            
            all_tps = [] # To store tokens/second for each question

            for i, q_data in enumerate(questions):
                prompt = q_data.get("prompt")
                if not prompt:
                    logger.warning(f"Question {i+1} has no prompt. Skipping.")
                    num_questions -=1 # Adjust count of valid questions
                    continue

                logger.debug(f"Querying model for question {i+1}/{len(questions)}...")
                logger.debug("Prompt : "+prompt)
                response_text, tps, error = get_ollama_response(model_name, prompt, model_options)
                logger.debug(f"Response received for question {response_text}.")

                if error:
                    logger.error(f"      Error getting response for question {q_data.get('id', i+1)}: {error}")
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
                    logger.debug(f"Question {q_data.get('id', i+1)} - Score: {question_score:.2f}" + (f", TPS: {tps:.2f}" if tps else "")+"\n")
                else:
                    logger.warning(f"Question {q_data.get('id', i+1)} - Could not be evaluated.")
                    # num_questions -=1 # If unevaluable questions shouldn't count towards the average

            monitoring_results = monitor.stop() # End monitoring
            avg_score_percent = (total_score / successful_evals) * 100 if successful_evals > 0 else 0.0
            avg_tps = sum(all_tps) / len(all_tps) if all_tps else None

            result_entry = { 
                "model": model_name, 
                "benchmark": benchmark_name, 
                "score": avg_score_percent, 
                "avg_tokens_s": avg_tps, 
                "static_info": monitor.static_info,
            }
            result_entry.update(monitoring_results) 
            all_results.append(result_entry) 

            logger.info(f"Summary for {model_name} on Benchmark {benchmark_name} :")
            logger.info(f"    Average Score: {avg_score_percent:.2f}% (over {successful_evals} evaluated questions)")
            if avg_tps is not None:
                logger.info(f"    Average Tokens/Second: {avg_tps:.2f}")
            else:
                logger.warning(f"    Average Tokens/Second: N/A")
            
            if monitoring_results:
                logger.info(" System Usage (Avg):")
                logger.info(f"    CPU: {monitoring_results.get('avg_cpu_percent', 0):.2f}% | RAM: {monitoring_results.get('avg_ram_percent', 0):.2f}%")
                if 'avg_gpu_util_percent' in monitoring_results:
                     logger.info(f"   GPU Util: {monitoring_results.get('avg_gpu_util_percent', 0):.2f}% | GPU Mem: {monitoring_results.get('avg_gpu_mem_percent', 0):.2f}%")
                     logger.info(f"   Total GPU Energy: {monitoring_results.get('total_gpu_energy_wh', 0):.6f} Wh")

            time.sleep(5) # Small delay between benchmarks

    return all_results