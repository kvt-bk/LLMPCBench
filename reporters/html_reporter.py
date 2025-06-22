import os
import logging
from datetime import datetime
from reporters.base_reporter import BaseReporter

logger = logging.getLogger(__name__)

# Define the HTML templates directly within the reporter file
HTML_BASE_TEMPLATE = """
<html>
<head>
    <title>LLM Evaluation Results Log</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
        h1 {{ text-align: center; color: #343a40; }}
        .run-container {{ 
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 30px; 
            padding: 20px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}
        .run-header h2 {{ margin-top: 0; color: #495057; }}
        .run-header p {{ font-size: 0.9em; color: #6c757d; font-family: 'Courier New', Courier, monospace; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
        th, td {{ border: 1px solid #e9ecef; text-align: left; padding: 10px; }}
        th {{ background-color: #007bff; color: white; font-weight: bold; white-space: nowrap; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .na-value {{ color: #999; font-style: italic; }}
    </style>
</head>
<body>
    <h1>LLM Evaluation Results Log</h1>
    </body>
</html>
"""

RUN_TEMPLATE = """
<div class="run-container">
    <div class="run-header">
        <h2>Evaluation Run: {datetime}</h2>
        <p>CPU: {cpu_model}<br>GPU: {gpu_models}</p>
    </div>
    <table>
        <thead>
            {header_row}
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</div>
"""

class HTMLReporter(BaseReporter):
    """Saves evaluation results to a cumulative HTML file."""
    config_key = "html" # Define config key for auto-discovery

    def __init__(self, config: dict):
        super().__init__(config)
        self.output_filename = self.config.get('output_file', 'evaluation_results.html')

    def report(self, results_data: list[dict]):
        if not results_data:
            logger.warning("No results were generated, skipping HTML report generation.")
            return

        # --- START OF THE CORRECTED LOGIC ---
        
        static_info = results_data[0].get('static_info', {})
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        headers = [
            "Model", "Benchmark", "Score (%)", "Tokens/s", "Avg CPU %", 
            "Avg RAM %", "Avg GPU %", "GPU Energy (Wh)"
        ]
        header_html = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

        rows_html_list = []
        # Loop through each result and build the HTML row
        for res in results_data:
            row_html = "<tr>"
            row_html += f"<td>{res.get('model', 'N/A')}</td>"
            row_html += f"<td>{res.get('benchmark', 'N/A')}</td>"

            # --- Simplified and Corrected Formatting Logic ---

            # Format Score
            score_val = res.get('score')
            score_str = f"{score_val:.2f}" if score_val is not None else '<span class="na-value">N/A</span>'
            row_html += f"<td>{score_str}</td>"

            # Format Tokens/Second
            tps_val = res.get('avg_tokens_s')
            tps_str = f"{tps_val:.2f}" if tps_val is not None else '<span class="na-value">N/A</span>'
            row_html += f"<td>{tps_str}</td>"

            # Format required system metrics
            row_html += f"<td>{res.get('avg_cpu_percent', 0):.2f}</td>"
            row_html += f"<td>{res.get('avg_ram_percent', 0):.2f}</td>"

            # Format optional GPU metrics
            if 'avg_gpu_util_percent' in res:
                gpu_util_str = f"{res.get('avg_gpu_util_percent', 0):.2f}"
                gpu_energy_str = f"{res.get('total_gpu_energy_wh', 0):.6f}"
                row_html += f"<td>{gpu_util_str}</td>"
                row_html += f"<td>{gpu_energy_str}</td>"
            else:
                row_html += '<td><span class="na-value">N/A</span></td>'
                row_html += '<td><span class="na-value">N/A</span></td>'

            row_html += "</tr>"
            rows_html_list.append(row_html)
        
        table_rows_html = "\n".join(rows_html_list)

        # Create the complete HTML block for this new run
        new_run_html = RUN_TEMPLATE.format(
            datetime=current_time_str,
            cpu_model=static_info.get('cpu_model', 'N/A'),
            gpu_models=static_info.get('gpu_models', 'N/A'),
            header_row=header_html,
            table_rows=table_rows_html
        )

        # Read existing file or create a new one and insert the new block
        file_content = ""
        if os.path.exists(self.output_filename):
            with open(self.output_filename, 'r', encoding='utf-8') as f:
                file_content = f.read()
        
        if not file_content:
            file_content = HTML_BASE_TEMPLATE

        insertion_point = "</h1>"
        if insertion_point in file_content:
            parts = file_content.split(insertion_point)
            final_html = parts[0] + insertion_point + "\n" + new_run_html + "\n".join(parts[1:])
        else:
             final_html = file_content.replace("", new_run_html)

        # Write the updated content back to the file
        try:
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                f.write(final_html)
            logger.info(f"HTML report successfully generated and saved to {self.output_filename}")
        except IOError as e:
            logger.error(f"Error writing HTML file {self.output_filename}: {e}")