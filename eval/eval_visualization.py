import json
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

LABEL_FONTSIZE = 30
TITLE_FONTSIZE = 36
TICK_FONTSIZE = 26

def visualize_benchmark_results(json_file_path: str):
    """
    Create comprehensive visualizations for benchmark evaluation results.

    Args:
        json_file_path: Path to the JSON results file
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    eval_timestamp = data['timestamp']
    eval_date = datetime.fromisoformat(eval_timestamp).strftime("%Y-%m-%d %H:%M:%S")

    first_query = next(iter(data['queries'].values()))
    if first_query and 'batches' in first_query:
        first_batch = next(iter(first_query['batches'].values()))
        k_values = []
        for metric_key in first_batch['metrics'].keys():
            if '@' in metric_key:
                k = int(metric_key.split('@')[1])
                if k not in k_values:
                    k_values.append(k)
        k_values.sort()
    else:
        k_values = [1, 3, 5]

    batch_metrics = ['precision', 'recall', 'hit_rate', 'f1']
    standard_metrics = ['hit_rate', 'precision', 'recall', 'f1']

    def get_metric_colors():
        """Create color mapping for metrics."""
        base_colors = {
            'precision': '#EB3434',  # Red
            'recall': '#07C3F2',  # Cyan/Teal
            'hit_rate': '#05FF0D',  # Green
            'f1': '#DB0FD8'  # Purple
        }

        colors = {}
        for metric in ['precision', 'recall', 'hit_rate', 'f1']:
            base_color = base_colors[metric]
            rgb = tuple(int(base_color[i:i + 2], 16) for i in (1, 3, 5))

            for j, k in enumerate(k_values):
                intensity = 0.7 + (j * 0.1)
                colors[f'{metric}@{k}'] = tuple(int(c * intensity) for c in rgb)

        hex_colors = {}
        for key, rgb in colors.items():
            hex_colors[key] = tuple(c / 255.0 for c in rgb)

        return hex_colors

    metric_colors = get_metric_colors()

    # 1. Batch-level plots (one plot per query per batch)
    for query_num, query_data in data['queries'].items():
        query_text = query_data['query_text'][:50] + "..." if len(query_data['query_text']) > 50 else query_data[
            'query_text']

        batches = list(query_data['batches'].keys())
        n_batches = len(batches)

        if n_batches == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Query {query_num} - Batch-level Metrics\nEval: {eval_date}\n"{query_text}"', fontsize=14)
        axes = axes.flatten()

        for i, metric in enumerate(batch_metrics):
            ax = axes[i]

            q_batch_labels = []
            metric_data = {f'@{k}': [] for k in k_values}

            for ts_key, ts_data in query_data['batches'].items():
                q_batch_labels.append(ts_key.split('_')[1])
                for k in k_values:
                    metric_data[f'@{k}'].append(ts_data['metrics'][f'{metric}@{k}'])

            x = np.arange(len(q_batch_labels))
            width = 0.25

            for j, k in enumerate(k_values):
                offset = (j - len(k_values) / 2 + 0.5) * width
                color = metric_colors[f'{metric}@{k}']
                ax.bar(x + offset, metric_data[f'@{k}'], width, label=f'@{k}', color=color)

            ax.set_xlabel('Query Batch')
            ax.set_ylabel(f'{metric.title()}')
            ax.set_title(f'{metric.title()}@k by Query Batch')
            ax.set_xticks(x)
            ax.set_xticklabels(q_batch_labels)
            ax.legend()
            ax.grid(False)
            ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

    # 2. Query-level averages plot
    queries = list(data['queries'].keys())
    if queries:
        fig, ax = plt.subplots(1, 1, figsize=(28, 14))

        query_data_dict = {}
        for metric in standard_metrics:
            query_data_dict[metric] = {}
            for k in k_values:
                query_data_dict[metric][f'@{k}'] = [data['queries'][q]['query_averages'].get(f'avg_{metric}@{k}', 0)
                                                    for q in queries]

        n_queries = len(queries)
        x = np.arange(n_queries)

        total_bars = len(k_values) * len(standard_metrics)
        width = 0.8 / total_bars

        bar_offset = 0

        for metric in standard_metrics:
            for j, k in enumerate(k_values):
                color = metric_colors[f'{metric}@{k}']
                ax.bar(x + bar_offset * width, query_data_dict[metric][f'@{k}'], width,
                       label=f'{metric.title()}@{k}', color=color)
                bar_offset += 1

        ax.set_xlabel('Query Number', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Average Score', fontsize=LABEL_FONTSIZE)
        ax.set_title(f'Query-level Average Metrics', fontsize=TITLE_FONTSIZE)
        ax.set_xticks(x + (total_bars - 1) * width / 2)
        ax.set_xticklabels(range(1, len(queries) + 1), fontsize=TICK_FONTSIZE)  # Changed to incremental integers
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=TICK_FONTSIZE)
        ax.grid(False)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        save_dir = os.path.dirname(json_file_path)
        plt.savefig(os.path.join(save_dir, 'query_averages_standard.pdf'), dpi=300, bbox_inches='tight', pad_inches=1.0)
        plt.show()

    # 3. Global averages plot
    if data['global_averages']:
        fig, ax = plt.subplots(1, 1, figsize=(28, 14))

        global_metrics_std = []
        global_values_std = []
        global_colors_std = []

        for metric in standard_metrics:
            for k in k_values:
                global_metrics_std.append(f'{metric.title()}@{k}')
                global_values_std.append(data['global_averages'].get(f'global_avg_{metric}@{k}', 0))
                global_colors_std.append(metric_colors[f'{metric}@{k}'])

        bars = ax.bar(range(len(global_metrics_std)), global_values_std, color=global_colors_std)

        ax.set_xlabel('Metrics', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Global Average Score', fontsize=LABEL_FONTSIZE)
        ax.set_title(f'Global Average Metrics Across All Queries', fontsize=TITLE_FONTSIZE)
        ax.set_xticks(range(len(global_metrics_std)))
        ax.set_xticklabels(global_metrics_std, rotation=45, ha='right', fontsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.grid(False)
        ax.set_ylim(0, 1.0)

        for bar, value in zip(bars, global_values_std):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=TICK_FONTSIZE)

        plt.tight_layout()
        save_dir = os.path.dirname(json_file_path)
        plt.savefig(os.path.join(save_dir, 'global_averages_standard.pdf'), dpi=300, bbox_inches='tight', pad_inches=1.0)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize benchmark evaluation results from JSON file')
    parser.add_argument('json_file', type=str, nargs='?',
                       help='Path to the benchmark results JSON file')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest results file from eval/results/retrieval/')

    args = parser.parse_args()

    if args.latest:
        # Find the latest results directory
        results_base = './eval/results/retrieval'
        if os.path.exists(results_base):
            subdirs = [d for d in os.listdir(results_base)
                      if os.path.isdir(os.path.join(results_base, d))]
            if subdirs:
                latest_dir = max(subdirs)
                json_path = os.path.join(results_base, latest_dir, 'benchmark_results.json')
                if os.path.exists(json_path):
                    print(f"Using latest results: {json_path}")
                    visualize_benchmark_results(json_path)
                else:
                    print(f"No benchmark_results.json found in {os.path.join(results_base, latest_dir)}")
            else:
                print("No result directories found in eval/results/retrieval/")
        else:
            print("eval/results/retrieval/ directory not found")
    elif args.json_file:
        if os.path.exists(args.json_file):
            print(f"Visualizing results from: {args.json_file}")
            visualize_benchmark_results(args.json_file)
        else:
            print(f"Error: File not found: {args.json_file}")
    else:
        # Interactive mode - prompt user for file path
        print("\n=== Benchmark Results Visualization ===\n")
        json_path = input("Enter path to benchmark results JSON file: ").strip()

        if json_path.startswith('"') and json_path.endswith('"'):
            json_path = json_path[1:-1]
        if json_path.startswith("'") and json_path.endswith("'"):
            json_path = json_path[1:-1]

        if os.path.exists(json_path):
            print(f"\nVisualizing results from: {json_path}\n")
            visualize_benchmark_results(json_path)
        else:
            print(f"Error: File not found: {json_path}")
            print("\nExample usage:")
            print("  python eval_visualization.py ./eval/results/retrieval/20240115_143022/benchmark_results.json")
            print("  python eval_visualization.py --latest")
