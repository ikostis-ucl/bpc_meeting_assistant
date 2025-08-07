from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from eval.eval_utils import precision_at_k, recall_at_k, average_precision, hit_rate_at_k, ndcg_at_k


class BenchmarkEvaluator:
    """
    Timespan-aware page-level evaluation metrics for RAG retrieval benchmark.
    """

    def __init__(self, gt_csv_path: str, ts_doc_index: Dict, timespans: List[Tuple[int, int]]):
        """
        Initialize with ground truth CSV and timespan information.

        Args:
            gt_csv_path: Path to ground truth CSV file
            ts_doc_index: Timestamp to document name mapping from BaseInference
            timespans: List of (start_timestamp, end_timestamp) tuples
        """
        self.gt_df = pd.read_csv(gt_csv_path, sep=';')
        self.queries = self._parse_ground_truth()

        self.ts_doc_index = {}
        for timestamp, doc_names in ts_doc_index.items():
            if isinstance(doc_names, list):
                self.ts_doc_index[timestamp] = [doc.removesuffix('.pdf') for doc in doc_names]
            else:
                self.ts_doc_index[timestamp] = doc_names.removesuffix('.pdf')

        self.timespans = timespans

    def _parse_ground_truth(self) -> Dict[str, Dict]:
        """Parse ground truth CSV into query -> document -> pages structure."""
        queries = {}

        for _, row in self.gt_df.iterrows():
            query_num = str(row['Query Number'])
            query_text = row['Query']
            queries[query_num] = {
                'text': query_text,
                'relevant_pages': set()  # Store all document::page IDs for this query
            }

            # Parse each document column
            for col in self.gt_df.columns[2:]:  # Skip Query Number and Query columns
                doc_name = col.removesuffix('.pdf')
                pages_str = str(row[col])

                if pd.isna(pages_str) or pages_str == 'nan' or pages_str == '':
                    continue

                # Parse page numbers (handle formats like "3&5", "4&7", etc.)
                if '&' in pages_str:
                    for page in pages_str.split('&'):
                        try:
                            page_num = int(float(page.strip()))
                            page_id = f"{doc_name}::{page_num}"
                            queries[query_num]['relevant_pages'].add(page_id)
                        except ValueError:
                            continue
                else:
                    try:
                        page_num = int(float(pages_str.strip()))
                        page_id = f"{doc_name}::{page_num}"
                        queries[query_num]['relevant_pages'].add(page_id)
                    except ValueError:
                        continue

        return queries

    def _get_timespan_documents(self, start_timestamp: int, end_timestamp: int) -> Set[str]:
        """Get documents present in a specific timespan."""
        timespan_documents = set()

        for timestamp, doc_names in self.ts_doc_index.items():
            if start_timestamp <= timestamp <= end_timestamp:
                if isinstance(doc_names, list):
                    timespan_documents.update(doc_names)
                else:
                    timespan_documents.add(doc_names)

        return timespan_documents

    def _extract_retrieved_pages(self, metadata: Dict) -> Set[str]:
        """Extract document::page identifiers from retrieval metadata."""
        pages = set()

        if metadata:
            for node_data in metadata.values():
                if 'metadata' in node_data:
                    node_metadata = node_data['metadata']
                    if 'page_number' in node_metadata and 'file_name' in node_metadata:
                        doc_name = node_metadata['file_name'].removesuffix('.pdf')
                        page_num = node_metadata['page_number']
                        page_id = f"{doc_name}::{page_num}"
                        pages.add(page_id)

        return pages

    def _filter_ground_truth_by_timespan(self, query_num: str, timespan_documents: Set[str]) -> Set[str]:
        """Filter ground truth pages to only include documents present in timespan."""
        if query_num not in self.queries:
            return set()

        all_relevant_pages = self.queries[query_num]['relevant_pages']
        timespan_relevant_pages = set()

        for page_id in all_relevant_pages:
            doc_name = page_id.split('::')[0]
            if doc_name in timespan_documents:
                timespan_relevant_pages.add(page_id)

        return timespan_relevant_pages

    def evaluate_query(self, query_num: str, results: List[Tuple], k_values: List[int]) -> Dict:
        """Evaluate a single query with timespan-level breakdown."""
        if query_num not in self.queries:
            return {}

        query_results = {
            'query_text': self.queries[query_num]['text'],
            'total_relevant_pages': len(self.queries[query_num]['relevant_pages']),
            'timespans': {},
            'query_averages': {}
        }

        timespan_metrics = defaultdict(list)

        # Evaluate each timespan separately
        for i, (_, metadata, timespan) in enumerate(results):
            start_timestamp, end_timestamp = timespan
            timespan_key = f"ts_{i}_{start_timestamp}_{end_timestamp}"

            # Get documents present in this timespan
            timespan_documents = self._get_timespan_documents(start_timestamp, end_timestamp)

            # Get retrieved pages from metadata
            retrieved_pages = self._extract_retrieved_pages(metadata)

            # Filter ground truth to only include pages from documents in this timespan
            relevant_pages = self._filter_ground_truth_by_timespan(query_num, timespan_documents)

            # Skip if no relevant pages in this timespan (at least 1 document with a page needed)
            if not relevant_pages:
                continue

            timespan_result = {
                'timespan': timespan,
                'timespan_documents': list(timespan_documents),
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': list(relevant_pages),
                'metrics': {}
            }

            # Calculate MAP
            retrieved_pages_list = list(retrieved_pages)
            if retrieved_pages_list and relevant_pages:
                map_score = average_precision(retrieved_pages_list, relevant_pages)
            else:
                map_score = 0.0

            timespan_result['metrics']['map'] = map_score
            timespan_metrics['map'].append(map_score)

            # Calculate k-dependent metrics
            for k in k_values:
                if relevant_pages and retrieved_pages:
                    precision = precision_at_k(retrieved_pages, relevant_pages, k)
                    recall = recall_at_k(retrieved_pages, relevant_pages, k)
                    hit_rate = hit_rate_at_k(retrieved_pages, relevant_pages, k)
                    ndcg = ndcg_at_k(retrieved_pages_list, relevant_pages, k)
                else:
                    precision = recall = hit_rate = ndcg = 0.0

                timespan_result['metrics'][f'precision@{k}'] = precision
                timespan_result['metrics'][f'recall@{k}'] = recall
                timespan_result['metrics'][f'hit_rate@{k}'] = hit_rate
                timespan_result['metrics'][f'ndcg@{k}'] = ndcg

                # Collect for query-level averaging
                timespan_metrics[f'precision@{k}'].append(precision)
                timespan_metrics[f'recall@{k}'].append(recall)
                timespan_metrics[f'hit_rate@{k}'].append(hit_rate)
                timespan_metrics[f'ndcg@{k}'].append(ndcg)

            query_results['timespans'][timespan_key] = timespan_result

        # Calculate query-level averages
        for metric, values in timespan_metrics.items():
            if values:
                query_results['query_averages'][f'avg_{metric}'] = np.mean(values)
                query_results['query_averages'][f'std_{metric}'] = np.std(values)

        return query_results

    def evaluate_all_queries(self, results_dict: Dict[str, List[Tuple]], k_values: List[int]) -> Dict:
        """Evaluate all queries and return hierarchical results."""
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'queries': {},
            'global_averages': {}
        }

        query_level_metrics = defaultdict(list)
        evaluated_queries = 0
        skipped_queries = 0

        # Evaluate each query
        for query_num, results in results_dict.items():
            query_result = self.evaluate_query(query_num, results, k_values)

            if query_result and query_result.get('timespans'):
                evaluation_results['queries'][query_num] = query_result
                evaluated_queries += 1

                # Collect query averages for global calculation
                for metric, value in query_result['query_averages'].items():
                    if metric.startswith('avg_'):
                        query_level_metrics[metric].append(value)
            else:
                skipped_queries += 1

        # Calculate global averages
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
