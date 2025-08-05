from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from eval.eval_utils import precision_at_k, recall_at_k, average_precision, hit_rate_at_k, ndcg_at_k


class BenchmarkEvaluator:
    """
    Page-level evaluation metrics for RAG retrieval benchmark.
    """

    def __init__(self, gt_csv_path: str):
        """
        Initialize with ground truth CSV.

        Args:
            gt_csv_path: Path to ground truth CSV file
        """
        self.gt_df = pd.read_csv(gt_csv_path, sep=';')
        self.queries = self._parse_ground_truth()

    def _parse_ground_truth(self) -> Dict[str, Dict[str, Set[int]]]:
        """Parse ground truth CSV into query -> document -> pages structure."""
        queries = {}

        for _, row in self.gt_df.iterrows():
            query_num = str(row['Query Number'])
            query_text = row['Query']
            queries[query_num] = {'text': query_text, 'documents': {}}

            # Parse each document column
            for col in self.gt_df.columns[2:]:  # Skip Query Number and Query columns
                doc_name = col
                pages_str = str(row[col])

                if pd.isna(pages_str) or pages_str == 'nan' or pages_str == '':
                    continue

                # Parse page numbers (handle formats like "3&5", "4&7", etc.)
                pages = set()
                if '&' in pages_str:
                    for page in pages_str.split('&'):
                        try:
                            pages.add(int(page.strip()))
                        except ValueError:
                            continue
                else:
                    try:
                        pages.add(int(pages_str.strip()))
                    except ValueError:
                        continue

                if pages:
                    queries[query_num]['documents'][doc_name] = pages

        return queries

    @staticmethod
    def _extract_pages_from_results(results: List[Tuple]) -> Set[int]:
        """Extract page numbers from retrieval results."""
        pages = set()

        for _, metadata, _ in results:
            if metadata:
                for node_data in metadata.values():
                    if 'metadata' in node_data and 'page_number' in node_data['metadata']:
                        pages.add(node_data['metadata']['page_number'])

        return pages

    def evaluate_query(self, query_num: str, results: List[Tuple], k_values: List[int]) -> Dict:
        """Evaluate a single query with timespan-level breakdown."""
        if query_num not in self.queries:
            return {}

        # Get all relevant pages for this query across all documents
        relevant_pages = set()
        for doc_pages in self.queries[query_num]['documents'].values():
            relevant_pages.update(doc_pages)

        # Skip evaluation if no ground truth pages exist
        if not relevant_pages:
            print(f"Skipping query {query_num}: No ground truth pages found")
            return {}

        query_results = {
            'query_text': self.queries[query_num]['text'],
            'total_relevant_pages': len(relevant_pages),
            'timespans': {},
            'query_averages': {}
        }

        timespan_metrics = defaultdict(list)

        # Evaluate each timespan separately
        for i, (_, metadata, timespan) in enumerate(results):
            timespan_key = f"timespan_{i + 1}_{timespan[0]}_{timespan[1]}"

            # Extract pages for this specific timespan
            timespan_pages = set()
            if metadata:
                for node_data in metadata.values():
                    if 'metadata' in node_data and 'page_number' in node_data['metadata']:
                        timespan_pages.add(node_data['metadata']['page_number'])

            timespan_pages_list = list(timespan_pages)
            actual_pages_retrieved = len(timespan_pages)

            timespan_result = {
                'timespan': timespan,
                'retrieved_pages_count': actual_pages_retrieved,
                'retrieved_pages': list(timespan_pages),
                'metrics': {}
            }

            # Calculate metrics for this timespan
            for k in k_values:
                effective_k = min(k, actual_pages_retrieved, len(relevant_pages))

                if effective_k > 0:
                    precision = precision_at_k(timespan_pages, relevant_pages, effective_k)
                    recall = recall_at_k(timespan_pages, relevant_pages, effective_k)
                    hit_rate = hit_rate_at_k(timespan_pages, relevant_pages, effective_k)
                    ndcg = ndcg_at_k(timespan_pages_list, relevant_pages, effective_k)
                    map_score = average_precision(timespan_pages_list, relevant_pages)
                else:
                    # Zero because no pages retrieved, not because GT missing
                    precision = recall = hit_rate = ndcg = map_score = 0.0

                timespan_result['metrics'][f'precision@{k}'] = precision
                timespan_result['metrics'][f'recall@{k}'] = recall
                timespan_result['metrics'][f'hit_rate@{k}'] = hit_rate
                timespan_result['metrics'][f'ndcg@{k}'] = ndcg

                # Collect for query-level averaging (include zeros from no retrieval)
                timespan_metrics[f'precision@{k}'].append(precision)
                timespan_metrics[f'recall@{k}'].append(recall)
                timespan_metrics[f'hit_rate@{k}'].append(hit_rate)
                timespan_metrics[f'ndcg@{k}'].append(ndcg)

            timespan_result['metrics']['map'] = map_score
            timespan_metrics['map'].append(map_score)

            query_results['timespans'][timespan_key] = timespan_result

        # Calculate query-level averages (include all timespan results)
        for metric, values in timespan_metrics.items():
            if values:
                query_results['query_averages'][f'avg_{metric}'] = np.mean(values)
                query_results['query_averages'][f'std_{metric}'] = np.std(values)

        return query_results

    def evaluate_all_queries(self, results_dict: Dict[str, List[Tuple]], k_values: List[int]) -> Dict:
        """Evaluate all queries and return hierarchical results."""
        from datetime import datetime

        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {},
            'queries': {},
            'global_averages': {}
        }

        query_level_metrics = defaultdict(list)
        evaluated_queries = 0
        skipped_queries = 0

        # Evaluate each query
        for query_num, results in results_dict.items():
            query_metrics = self.evaluate_query(query_num, results, k_values)

            if query_metrics:  # Only process if metrics were calculated (GT exists)
                evaluated_queries += 1
                evaluation_results['queries'][query_num] = query_metrics

                # Collect query averages for global averaging (exclude GT-missing queries)
                for metric, value in query_metrics['query_averages'].items():
                    if isinstance(value, (int, float)) and metric.startswith('avg_'):
                        query_level_metrics[metric].append(value)
            else:
                skipped_queries += 1
                evaluation_results['queries'][query_num] = {
                    'status': 'skipped',
                    'reason': 'no_ground_truth'
                }

        # Calculate global averages (only from queries that had GT)
        for metric, values in query_level_metrics.items():
            if values:
                evaluation_results['global_averages'][f'global_{metric}'] = np.mean(values)
                evaluation_results['global_averages'][f'global_std_{metric.replace("avg_", "")}'] = np.std(values)

        # Add evaluation summary
        evaluation_results['evaluation_summary'] = {
            'total_queries': len(results_dict),
            'evaluated_queries': evaluated_queries,
            'skipped_queries': skipped_queries,
            'k_values': k_values
        }

        print(f"\nEvaluation Summary:")
        print(f"Total queries: {len(results_dict)}")
        print(f"Evaluated queries: {evaluated_queries}")
        print(f"Skipped queries (no ground truth): {skipped_queries}")

        return evaluation_results
