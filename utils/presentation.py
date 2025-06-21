# ollama_eval_project/utils/presentation.py
import os
from datetime import datetime
from tabulate import tabulate

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

def print_results_table(results_data):
    """Prints evaluation results in a formatted console table."""
    if not results_data:
        print("No results to display.")
        return

    # All entries in a single run should have the same static info
    static_info = results_data[0].get('static_info', {})
    cpu_model = static_info.get('cpu_model', 'N/A')
    gpu_models = static_info.get('gpu_models', 'N/A')

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- Console Evaluation Results ({current_time}) ---")
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
            f"{res.get('avg_gpu_util_percent', 0):.2f}" if res.get('total_gpu_energy_wh') is not None else "N/A",
            f"{res.get('total_gpu_energy_wh', 0):.6f}" if res.get('total_gpu_energy_wh') is not None else "N/A",
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_results_to_html(results_data, output_filename="evaluation_results.html"):
    """Saves evaluation results to an HTML file, appending new results."""
    if not results_data:
        print("No results were generated, skipping HTML save.")
        return

    # 1. Gather static info (should be same for all results in this run)
    static_info = results_data[0].get('static_info', {})
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. Create HTML table header
    headers = [
        "Model", "Benchmark", "Score (%)", "Tokens/s", "Avg CPU %", 
        "Avg RAM %", "Avg GPU %", "GPU Energy (Wh)"
    ]
    header_html = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

    # 3. Create HTML table rows
    rows_html_list = []
    for res in results_data:
        row_html = "<tr>"
        row_html += f"<td>{res.get('model', 'N/A')}</td>"
        row_html += f"<td>{res.get('benchmark', 'N/A')}</td>"
        row_html += f"<td>{res.get('score', 0):.2f}</td>"
        row_html += f"<td>{res.get('avg_tokens_s', 0):.2f}</td>"
        row_html += f"<td>{res.get('avg_cpu_percent', 0):.2f}</td>"
        row_html += f"<td>{res.get('avg_ram_percent', 0):.2f}</td>"
        # Handle optional GPU metrics gracefully
        if res.get('total_gpu_energy_wh') is not None:
             row_html += f"<td>{res.get('avg_gpu_util_percent', 0):.2f}</td>"
             row_html += f"<td>{res.get('total_gpu_energy_wh', 0):.6f}</td>"
        else:
            row_html += '<td><span class="na-value">N/A</span></td>' * 2
        row_html += "</tr>"
        rows_html_list.append(row_html)
    table_rows_html = "\n".join(rows_html_list)

    # 4. Create the complete HTML block for this new run
    new_run_html = RUN_TEMPLATE.format(
        datetime=current_time_str,
        cpu_model=static_info.get('cpu_model', 'N/A'),
        gpu_models=static_info.get('gpu_models', 'N/A'),
        header_row=header_html,
        table_rows=table_rows_html
    )

    # 5. Read existing file or create a new one and insert the new block
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            file_content = f.read()
    else:
        file_content = HTML_BASE_TEMPLATE

    insertion_point = "</h1>"
    parts = file_content.split(insertion_point)
    final_html = parts[0] + insertion_point + "\n" + new_run_html + "\n".join(parts[1:])
    
    if "" in final_html:
         final_html = final_html.replace("", new_run_html)

    # 6. Write the updated content back to the file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"\nResults successfully saved to {output_filename}")
    except IOError as e:
        print(f"Error writing HTML file {output_filename}: {e}")