# ollama_eval_project/utils/presentation.py

from tabulate import tabulate # We can keep this if you still want console output as an option

HTML_TEMPLATE = """
<html>
<head>
    <title>LLM Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ text-align: center; color: #333; }}
        table {{ 
            border-collapse: collapse; 
            width: 90%; 
            margin: 20px auto; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            text-align: left; 
            padding: 10px; 
        }}
        th {{ 
            background-color: #4CAF50; 
            color: white; 
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .na-value {{ color: #999; font-style: italic; }}
    </style>
</head>
<body>
    <h1>LLM Evaluation Results</h1>
    <table>
        <thead>
            {header_row}
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""

def print_results_table(results_data):
    """
    Prints evaluation results in a formatted console table.
    (This is the existing function, kept for console output option)
    """
    if not results_data:
        print("No results to display.")
        return

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
            for res in results_data:
                if res['model'] == model_name and res['benchmark'] == bench_name:
                    score_str = f"{res['score']:.2f}" if res['score'] is not None else "N/A"
                    tps_str = f"{res['avg_tokens_s']:.2f}" if res['avg_tokens_s'] is not None else "N/A"
                    row.extend([score_str, tps_str])
                    score_found = True
                    break
            if not score_found:
                 row.extend(["N/A", "N/A"])
        table_data.append(row)

    print("\n--- Console Evaluation Results ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_results_to_html(results_data, output_filename="evaluation_results.html"):
    """
    Saves evaluation results to an HTML file with a formatted table.

    Args:
        results_data (list of dicts): Each dict should contain:
            'model': Name of the LLM
            'benchmark': Name of the benchmark
            'score': Score achieved (e.g., percentage)
            'avg_tokens_s': Average tokens per second for this benchmark
        output_filename (str): The name of the HTML file to save.
    """
    if not results_data:
        print(f"No results to save to {output_filename}.")
        # Create an empty HTML file or one with a message
        with open(output_filename, 'w') as f:
            f.write(HTML_TEMPLATE.format(header_row="<tr><th>No Data</th></tr>", table_rows="<tr><td>No results were generated.</td></tr>"))
        return

    models = sorted(list(set(r['model'] for r in results_data)))
    benchmarks = sorted(list(set(r['benchmark'] for r in results_data)))

    # Create header row for HTML
    header_html = "<tr>\n                <th>Model</th>\n"
    for bench_name in benchmarks:
        header_html += f"                <th>{bench_name} Score (%)</th>\n"
        header_html += f"                <th>{bench_name} (Tokens/s)</th>\n"
    header_html += "            </tr>"

    # Create table rows for HTML
    rows_html_list = []
    for model_name in models:
        row_html = f"            <tr>\n                <td>{model_name}</td>\n"
        for bench_name in benchmarks:
            score_val = "N/A"
            tps_val = "N/A"
            score_class = "na-value"
            tps_class = "na-value"

            for res in results_data:
                if res['model'] == model_name and res['benchmark'] == bench_name:
                    if res['score'] is not None:
                        score_val = f"{res['score']:.2f}"
                        score_class = ""
                    if res['avg_tokens_s'] is not None:
                        tps_val = f"{res['avg_tokens_s']:.2f}"
                        tps_class = ""
                    break
            
            row_html += f'                <td class="{score_class}">{score_val}</td>\n'
            row_html += f'                <td class="{tps_class}">{tps_val}</td>\n'
        row_html += "            </tr>"
        rows_html_list.append(row_html)
    
    table_rows_html = "\n".join(rows_html_list)

    # Populate the HTML template
    final_html = HTML_TEMPLATE.format(header_row=header_html, table_rows=table_rows_html)

    try:
        with open(output_filename, 'w') as f:
            f.write(final_html)
        print(f"\nResults successfully saved to {output_filename}")
    except IOError as e:
        print(f"Error writing HTML file {output_filename}: {e}")