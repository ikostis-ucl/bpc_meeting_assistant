import json
from datetime import datetime
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np


def precision_at_k(retrieved_pages: Set[int], relevant_pages: Set[int], k: int) -> float:
    """Calculate Precision@K at page level."""
    if not retrieved_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    relevant_retrieved = sum(1 for page in retrieved_list if page in relevant_pages)
    return relevant_retrieved / min(k, len(retrieved_list))


def recall_at_k(retrieved_pages: Set[int], relevant_pages: Set[int], k: int) -> float:
    """Calculate Recall@K at page level."""
    if not relevant_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    relevant_retrieved = sum(1 for page in retrieved_list if page in relevant_pages)
    return relevant_retrieved / len(relevant_pages)


def average_precision(retrieved_pages: List[int], relevant_pages: Set[int]) -> float:
    """Calculate Average Precision for a single query."""
    if not relevant_pages:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0

    for i, page in enumerate(retrieved_pages, 1):
        if page in relevant_pages:
            relevant_found += 1
            precision_sum += relevant_found / i

    return precision_sum / len(relevant_pages) if relevant_pages else 0.0


def hit_rate_at_k(retrieved_pages: Set[int], relevant_pages: Set[int], k: int) -> float:
    """Calculate Hit Rate@K (binary: 1 if any relevant page found, 0 otherwise)."""
    if not relevant_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    return 1.0 if any(page in relevant_pages for page in retrieved_list) else 0.0


def ndcg_at_k(retrieved_pages: List[int], relevant_pages: Set[int], k: int) -> float:
    """Calculate NDCG@K with binary relevance."""
    if not relevant_pages:
        return 0.0

    # DCG@K
    dcg = 0.0
    for i, page in enumerate(retrieved_pages[:k]):
        if page in relevant_pages:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # IDCG@K (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_pages))))

    return dcg / idcg if idcg > 0 else 0.0


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
    k_values = data['evaluation_summary']['k_values']

    # Extract metrics for plotting
    timespan_metrics = ['precision', 'recall', 'hit_rate', 'ndcg']
    query_level_metrics = ['map', 'precision', 'recall', 'hit_rate', 'ndcg']

    # 1. Timespan-level plots (one plot per query per timespan)
    for query_num, query_data in data['queries'].items():
        query_text = query_data['query_text'][:50] + "..." if len(query_data['query_text']) > 50 else query_data[
            'query_text']

        timespans = list(query_data['timespans'].keys())
        n_timespans = len(timespans)

        if n_timespans == 0:
            continue

        # Create subplot grid for timespan metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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

            # Create grouped bar plot
            x = np.arange(len(timespan_labels))
            width = 0.25

            for j, k in enumerate(k_values):
                offset = (j - len(k_values) / 2 + 0.5) * width
                ax.bar(x + offset, metric_data[f'@{k}'], width, label=f'@{k}')

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

    # 2. Query-level averages plot
    queries = list(data['queries'].keys())
    if queries:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Prepare data for query-level metrics
        query_data_dict = {}
        for metric in query_level_metrics:
            query_data_dict[metric] = {}
            if metric == 'map':
                query_data_dict[metric][''] = [data['queries'][q]['query_averages'].get(f'avg_{metric}', 0) for q in
                                               queries]
            else:
                for k in k_values:
                    query_data_dict[metric][f'@{k}'] = [data['queries'][q]['query_averages'].get(f'avg_{metric}@{k}', 0)
                                                        for q in queries]

        # Create grouped bar plot
        n_queries = len(queries)
        x = np.arange(n_queries)

        # Calculate total number of bars per query
        total_bars = 1 + len(k_values) * (len(query_level_metrics) - 1)  # 1 for MAP + k_values for each other metric
        width = 0.8 / total_bars

        bar_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(query_level_metrics)))

        for i, metric in enumerate(query_level_metrics):
            if metric == 'map':
                ax.bar(x + bar_offset * width, query_data_dict[metric][''], width,
                       label=f'{metric.upper()}', color=colors[i], alpha=0.8)
                bar_offset += 1
            else:
                for j, k in enumerate(k_values):
                    ax.bar(x + bar_offset * width, query_data_dict[metric][f'@{k}'], width,
                           label=f'{metric.title()}@{k}', color=colors[i], alpha=0.6 + 0.1 * j)
                    bar_offset += 1

        ax.set_xlabel('Query Number')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Query-level Average Metrics\nEval: {eval_date}')
        ax.set_xticks(x + (total_bars - 1) * width / 2)
        ax.set_xticklabels(queries)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

    # 3. Global averages plot
    if data['global_averages']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Prepare global metrics data
        global_metrics = []
        global_values = []

        for metric in query_level_metrics:
            if metric == 'map':
                global_metrics.append(f'{metric.upper()}')
                global_values.append(data['global_averages'].get(f'global_avg_{metric}', 0))
            else:
                for k in k_values:
                    global_metrics.append(f'{metric.title()}@{k}')
                    global_values.append(data['global_averages'].get(f'global_avg_{metric}@{k}', 0))

        # Create bar plot
        colors = []
        for i, metric in enumerate(query_level_metrics):
            if metric == 'map':
                colors.append(plt.cm.Set3(i))
            else:
                base_color = plt.cm.Set3(i)
                for j in range(len(k_values)):
                    colors.append(base_color)

        bars = ax.bar(range(len(global_metrics)), global_values, color=colors, alpha=0.7)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Global Average Score')
        ax.set_title(f'Global Average Metrics Across All Queries\nEval: {eval_date}')
        ax.set_xticks(range(len(global_metrics)))
        ax.set_xticklabels(global_metrics, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, value in zip(bars, global_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()
