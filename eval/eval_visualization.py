import json
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


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

    # Extract k_values from the first query's metrics instead
    first_query = next(iter(data['queries'].values()))
    if first_query and 'timespans' in first_query:
        first_timespan = next(iter(first_query['timespans'].values()))
        k_values = []
        for metric_key in first_timespan['metrics'].keys():
            if '@' in metric_key:
                k = int(metric_key.split('@')[1])
                if k not in k_values:
                    k_values.append(k)
        k_values.sort()
    else:
        k_values = [1, 3, 5]  # fallback default

    # Extract metrics for plotting - now includes normalized metrics
    timespan_metrics = ['precision', 'recall', 'normalized_recall', 'hit_rate', 'f1', 'normalized_f1']

    # Separate metric groups for cleaner visualization
    standard_metrics = ['hit_rate', 'precision', 'recall', 'f1']
    normalized_metrics = ['hit_rate', 'precision', 'normalized_recall', 'normalized_f1']

    # Define color mapping for metrics (base colors for standard metrics, darker shades for normalized)
    def get_metric_colors():
        """Create color mapping for metrics with darker shades for normalized versions."""
        base_colors = {
            'precision': '#EB3434',  # Red
            'recall': '#07C3F2',  # Cyan/Teal
            'hit_rate': '#05FF0D',  # Green
            'f1': '#DB0FD8'  # Purple
        }

        colors = {}
        for metric in ['precision', 'recall', 'hit_rate', 'f1']:
            base_color = base_colors[metric]
            # Convert hex to RGB for manipulation
            rgb = tuple(int(base_color[i:i + 2], 16) for i in (1, 3, 5))

            # Create intensity variations for k values (lighter to darker)
            for j, k in enumerate(k_values):
                intensity = 0.7 + (j * 0.1)  # 0.6, 0.7, 0.8, 0.9
                colors[f'{metric}@{k}'] = tuple(int(c * intensity) for c in rgb)

                # Darker version for normalized metrics
                normalized_intensity = intensity * 0.65
                colors[f'normalized_{metric}@{k}'] = tuple(int(c * normalized_intensity) for c in rgb)

        # Convert RGB tuples back to hex/normalized RGB for matplotlib
        hex_colors = {}
        for key, rgb in colors.items():
            hex_colors[key] = tuple(c / 255.0 for c in rgb)  # Normalize to 0-1 for matplotlib

        return hex_colors

    metric_colors = get_metric_colors()

    # 1. Timespan-level plots (one plot per query per timespan)
    for query_num, query_data in data['queries'].items():
        query_text = query_data['query_text'][:50] + "..." if len(query_data['query_text']) > 50 else query_data[
            'query_text']

        timespans = list(query_data['timespans'].keys())
        n_timespans = len(timespans)

        if n_timespans == 0:
            continue

        # Create subplot grid for timespan metrics - now 2x3 to accommodate normalized metrics
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Query {query_num} - Timespan-level Metrics\nEval: {eval_date}\n"{query_text}"', fontsize=14)
        axes = axes.flatten()

        for i, metric in enumerate(timespan_metrics):
            ax = axes[i]

            # Prepare data for this metric
            timespan_labels = []
            metric_data = {f'@{k}': [] for k in k_values}

            for ts_key, ts_data in query_data['timespans'].items():
                timespan_labels.append(ts_key.split('_')[1])  # Extract timespan number
                for k in k_values:
                    metric_data[f'@{k}'].append(ts_data['metrics'][f'{metric}@{k}'])

            # Create grouped bar plot with custom colors
            x = np.arange(len(timespan_labels))
            width = 0.25

            for j, k in enumerate(k_values):
                offset = (j - len(k_values) / 2 + 0.5) * width
                color = metric_colors[f'{metric}@{k}']
                ax.bar(x + offset, metric_data[f'@{k}'], width, label=f'@{k}', color=color)

            ax.set_xlabel('Timespan')
            ax.set_ylabel(f'{metric.title()}')
            ax.set_title(f'{metric.title()}@k by Timespan')
            ax.set_xticks(x)
            ax.set_xticklabels(timespan_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

    # 2. Query-level averages plot - SPLIT INTO TWO PLOTS
    queries = list(data['queries'].keys())
    if queries:
        # 2a. Standard metrics plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Prepare data for standard metrics
        query_data_dict = {}
        for metric in standard_metrics:
            query_data_dict[metric] = {}
            for k in k_values:
                query_data_dict[metric][f'@{k}'] = [data['queries'][q]['query_averages'].get(f'avg_{metric}@{k}', 0)
                                                    for q in queries]

        # Create grouped bar plot with custom colors
        n_queries = len(queries)
        x = np.arange(n_queries)

        # Calculate total number of bars per query
        total_bars = len(k_values) * len(standard_metrics)
        width = 0.8 / total_bars

        bar_offset = 0

        for metric in standard_metrics:
            for j, k in enumerate(k_values):
                color = metric_colors[f'{metric}@{k}']
                ax.bar(x + bar_offset * width, query_data_dict[metric][f'@{k}'], width,
                       label=f'{metric.title()}@{k}', color=color)
                bar_offset += 1

        ax.set_xlabel('Query Number')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Query-level Average Metrics (Standard)\nEval: {eval_date}')
        ax.set_xticks(x + (total_bars - 1) * width / 2)
        ax.set_xticklabels(queries)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

        # 2b. Normalized metrics plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Prepare data for normalized metrics
        query_data_dict_norm = {}
        for metric in normalized_metrics:
            query_data_dict_norm[metric] = {}
            for k in k_values:
                query_data_dict_norm[metric][f'@{k}'] = [
                    data['queries'][q]['query_averages'].get(f'avg_{metric}@{k}', 0)
                    for q in queries]

        # Create grouped bar plot with custom colors
        bar_offset = 0

        for metric in normalized_metrics:
            for j, k in enumerate(k_values):
                color = metric_colors[f'{metric}@{k}']
                ax.bar(x + bar_offset * width, query_data_dict_norm[metric][f'@{k}'], width,
                       label=f'{metric.title()}@{k}', color=color)
                bar_offset += 1

        ax.set_xlabel('Query Number')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Query-level Average Metrics (Normalized)\nEval: {eval_date}')
        ax.set_xticks(x + (total_bars - 1) * width / 2)
        ax.set_xticklabels(queries)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

    # 3. Global averages plot - SPLIT INTO TWO PLOTS
    if data['global_averages']:
        # 3a. Standard metrics plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Prepare global standard metrics data
        global_metrics_std = []
        global_values_std = []
        global_colors_std = []

        for metric in standard_metrics:
            for k in k_values:
                global_metrics_std.append(f'{metric.title()}@{k}')
                global_values_std.append(data['global_averages'].get(f'global_avg_{metric}@{k}', 0))
                global_colors_std.append(metric_colors[f'{metric}@{k}'])

        # Create bar plot with custom colors
        bars = ax.bar(range(len(global_metrics_std)), global_values_std, color=global_colors_std)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Global Average Score')
        ax.set_title(f'Global Average Metrics (Standard) Across All Queries\nEval: {eval_date}')
        ax.set_xticks(range(len(global_metrics_std)))
        ax.set_xticklabels(global_metrics_std, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, value in zip(bars, global_values_std):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        save_dir = os.path.dirname(json_file_path)
        plt.savefig(os.path.join(save_dir, 'global_averages_standard.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 3b. Normalized metrics plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Prepare global normalized metrics data
        global_metrics_norm = []
        global_values_norm = []
        global_colors_norm = []

        for metric in normalized_metrics:
            for k in k_values:
                global_metrics_norm.append(f'{metric.title()}@{k}')
                global_values_norm.append(data['global_averages'].get(f'global_avg_{metric}@{k}', 0))
                global_colors_norm.append(metric_colors[f'{metric}@{k}'])

        # Create bar plot with custom colors
        bars = ax.bar(range(len(global_metrics_norm)), global_values_norm, color=global_colors_norm)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Global Average Score')
        ax.set_title(f'Global Average Metrics (Normalized) Across All Queries\nEval: {eval_date}')
        ax.set_xticks(range(len(global_metrics_norm)))
        ax.set_xticklabels(global_metrics_norm, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, value in zip(bars, global_values_norm):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'global_averages_normalized.png'), dpi=300, bbox_inches='tight')
        plt.show()
