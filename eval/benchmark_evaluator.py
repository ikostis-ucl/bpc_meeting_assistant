from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from eval.eval_utils import precision_at_k, recall_at_k, hit_rate_at_k, f1_at_k


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

    def _parse_ground_truth(self) -> Dict:
        """Parse ground truth CSV into a structured format."""
        queries = {}

        # Skip the first row (header) by using iloc[1:]
        for _, row in self.gt_df.iloc[0:].iterrows():
            # Convert first column to integer then to string to ensure format consistency
            try:
                query_num = str(int(float(row.iloc[0])))  # Handle cases like '1.0' -> '1'
            except (ValueError, TypeError):
                continue  # Skip rows with invalid query numbers

            # Skip if this query number is not in our benchmark index
            if query_num not in BENCHMARK_QUESTIONS_INDEX:
                continue

            if query_num not in queries:
                queries[query_num] = {
                    'text': BENCHMARK_QUESTIONS_INDEX[query_num],  # Get text from benchmark index
                    'relevant_pages': set()
                }

            # Process each document column (starting from column 2, which is index 2)
            for col_idx in range(2, len(row)):
                pages_str = str(row.iloc[col_idx])

                # Skip empty cells
                if pd.isna(row.iloc[col_idx]) or pages_str.strip() == '' or pages_str == 'nan':
                    continue

                # Get document name from column header
                doc_name = self.gt_df.columns[col_idx]

                # Remove .pdf extension if present
                if doc_name.endswith('.pdf'):
                    doc_name = doc_name[:-4]

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

    def _extract_retrieved_pages(self, metadata: Dict) -> List[str]:
        """Extract document::page identifiers from retrieval metadata."""
        pages = []
        seen = set()  # For deduplication while preserving order

        if metadata:
            for node_data in metadata.values():
                if 'metadata' in node_data:
                    node_metadata = node_data['metadata']
                    if 'page_number' in node_metadata and 'file_name' in node_metadata:
                        doc_name = node_metadata['file_name'].removesuffix('.pdf')
                        page_num = node_metadata['page_number']
                        page_id = f"{doc_name}::{page_num}"

                        if page_id not in seen:
                            pages.append(page_id)
                            seen.add(page_id)

        return pages

    def _filter_ground_truth_by_timespan(self, query_num: str, timespan_documents: Set[str]) -> Set[str]:
        # TODO: Make sure that the relevant pages include ALL the relevant document::pages in the timespan.
        #  If this is the case, the recall is capped due to the physical maximum of pages retrieved being
        #  less than the maximum total. Amend the recall calculation to account for this. Mention in the report.

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

            # Calculate k-dependent metrics including F1@k
            for k in k_values:
                if relevant_pages and retrieved_pages:
                    precision = precision_at_k(retrieved_pages, relevant_pages, k)
                    recall = recall_at_k(retrieved_pages, relevant_pages, k)
                    f1 = f1_at_k(retrieved_pages, relevant_pages, k)
                    hit_rate = hit_rate_at_k(retrieved_pages, relevant_pages, k)
                else:
                    precision = recall = hit_rate = f1 = 0.0

                timespan_result['metrics'][f'precision@{k}'] = precision
                timespan_result['metrics'][f'recall@{k}'] = recall
                timespan_result['metrics'][f'f1@{k}'] = f1
                timespan_result['metrics'][f'hit_rate@{k}'] = hit_rate

                # Collect for query-level averaging
                timespan_metrics[f'precision@{k}'].append(precision)
                timespan_metrics[f'recall@{k}'].append(recall)
                timespan_metrics[f'f1@{k}'].append(f1)
                timespan_metrics[f'hit_rate@{k}'].append(hit_rate)

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

        # Evaluate each query
        for query_num, results in results_dict.items():
            query_result = self.evaluate_query(query_num, results, k_values)

            if query_result and query_result.get('timespans'):
                evaluation_results['queries'][query_num] = query_result

                # Collect query averages for global calculation
                for metric, value in query_result['query_averages'].items():
                    if metric.startswith('avg_'):
                        query_level_metrics[metric].append(value)

        # Calculate global averages
        for metric, values in query_level_metrics.items():
            if values:
                evaluation_results['global_averages'][f'global_{metric}'] = np.mean(values)
                evaluation_results['global_averages'][f'global_std_{metric.replace("avg_", "")}'] = np.std(values)

        return evaluation_results
