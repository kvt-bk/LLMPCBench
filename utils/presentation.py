# ollama_eval_project/utils/presentation.py
from tabulate import tabulate

def print_results_table(results_data):
    """
    Prints evaluation results in a formatted table.

    Args:
        results_data (list of dicts): Each dict should contain:
            'model': Name of the LLM
            'benchmark': Name of the benchmark
            'score': Score achieved (e.g., percentage)
            'avg_tokens_s': Average tokens per second for this benchmark
    """
    if not results_data:
        print("No results to display.")
        return

    # Process data for tabulate: group by model, then list benchmark scores
    # Header: Model | Benchmark 1 Score | Benchmark 1 TPS | Benchmark 2 Score | Benchmark 2 TPS | ...
    
    models = sorted(list(set(r['model'] for r in results_data)))
    benchmarks = sorted(list(set(r['benchmark'] for r in results_data)))

    headers = ["Model"]
    for bench_name in benchmarks:
        headers.append(f"{bench_name} Score (%)")
        headers.append(f"{bench_name} (Tokens/s)")

    table_data = []
    for model_name in models:
        row = [model_name]
        for bench_name in benchmarks:
            score_found = False
            tps_found = False
            for res in results_data:
                if res['model'] == model_name and res['benchmark'] == bench_name:
                    score_str = f"{res['score']:.2f}" if res['score'] is not None else "N/A"
                    tps_str = f"{res['avg_tokens_s']:.2f}" if res['avg_tokens_s'] is not None else "N/A"
                    row.extend([score_str, tps_str])
                    score_found = True
                    tps_found = True
                    break
            if not score_found: # Should not happen if data is complete
                 row.extend(["N/A", "N/A"])
        table_data.append(row)

    print("\n--- Evaluation Results ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))