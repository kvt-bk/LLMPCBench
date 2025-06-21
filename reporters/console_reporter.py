import logging
from datetime import datetime
from tabulate import tabulate
from reporters.base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class ConsoleReporter(BaseReporter):
    """Prints evaluation results to the console."""
    def report(self, results_data: list[dict]):
        if not results_data:
            logger.info("No results to display in console table.")
            return

        static_info = results_data[0].get('static_info', {})
        cpu_model = static_info.get('cpu_model', 'N/A')
        gpu_models = static_info.get('gpu_models', 'N/A')

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- CONSOLE EVALUATION RESULTS ({current_time}) ---")
        print(f"CPU: {cpu_model} | GPU: {gpu_models}")

        headers = [
            "Model", "Benchmark", "Score (%)", "Tokens/s", "Avg CPU %",
            "Avg RAM %", "Avg GPU %", "GPU Energy (Wh)"
        ]
        
        table_data = []
        for res in results_data:
            row = [
                res.get('model', 'N/A'),
                res.get('benchmark', 'N/A'),
                f"{res.get('score', 0):.2f}",
                f"{res.get('avg_tokens_s', 0):.2f}" if res.get('avg_tokens_s') else "N/A",
                f"{res.get('avg_cpu_percent', 0):.2f}",
                f"{res.get('avg_ram_percent', 0):.2f}",
                f"{res.get('avg_gpu_util_percent', 0):.2f}" if 'avg_gpu_util_percent' in res else "N/A",
                f"{res.get('total_gpu_energy_wh', 0):.6f}" if 'total_gpu_energy_wh' in res else "N/A",
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("--- END OF CONSOLE REPORT ---")