from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from eval.eval_metrics import calculate_metrics_with_normalization


class BenchmarkEvaluator:
    """
    Timespan-aware page-level evaluation metrics for RAG retrieval benchmark.
    """

    def __init__(self, gt_csv_path: str, ts_doc_index: Dict, document_batches: List[List]):
        """
        Initialize with ground truth CSV and document batch information.

        Args:
            gt_csv_path: Path to ground truth CSV file
            ts_doc_index: Timestamp to document name mapping from BaseInference
            document_batches: List of document batches (each batch is a list of timestamps)
        """
        self.gt_df = pd.read_csv(gt_csv_path, sep=';')
        self.queries = self._parse_ground_truth()

        self.ts_doc_index = {}
        for timestamp, doc_names in ts_doc_index.items():
            if isinstance(doc_names, list):
                self.ts_doc_index[timestamp] = [doc.removesuffix('.pdf') for doc in doc_names]
            else:
                self.ts_doc_index[timestamp] = doc_names.removesuffix('.pdf')

        self.document_batches = document_batches

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

    def _get_batch_documents(self, timestamp_batch: List) -> Set[str]:
        """Get documents present in a specific document batch."""
        batch_documents = set()

        for timestamp in timestamp_batch:
            if timestamp in self.ts_doc_index:
                doc_names = self.ts_doc_index[timestamp]
                if isinstance(doc_names, list):
                    batch_documents.update(doc_names)
                else:
                    batch_documents.add(doc_names)

        return batch_documents

    def _filter_ground_truth_by_batch(self, query_num: str, batch_documents: Set[str]) -> Set[str]:
        """Filter ground truth pages to only include documents present in batch."""
        if query_num not in self.queries:
            return set()

        all_relevant_pages = self.queries[query_num]['relevant_pages']
        batch_relevant_pages = set()

        for page_id in all_relevant_pages:
            doc_name = page_id.split('::')[0]
            if doc_name in batch_documents:
                batch_relevant_pages.add(page_id)

        return batch_relevant_pages

    @staticmethod
    def _extract_retrieved_pages(metadata: Dict) -> List[str]:
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

    def evaluate_query(self, query_num: str, results: List[Tuple], k_values: List[int]) -> Dict:
        """Evaluate a single query with batch-level breakdown."""
        if query_num not in self.queries:
            return {}

        query_results = {
            'query_text': self.queries[query_num]['text'],
            'total_relevant_pages': len(self.queries[query_num]['relevant_pages']),
            'batches': {},
            'query_averages': {}
        }

        batch_metrics = defaultdict(list)

        # Evaluate each batch separately
        for i, (_, metadata, (batch_idx, timestamp_batch)) in enumerate(results):
            batch_key = f"batch_{batch_idx}"

            # Get documents present in this batch
            batch_documents = self._get_batch_documents(timestamp_batch)

            # Get retrieved pages from metadata
            retrieved_pages = self._extract_retrieved_pages(metadata)

            # Filter ground truth to only include pages from documents in this batch
            relevant_pages = self._filter_ground_truth_by_batch(query_num, batch_documents)

            # Skip if no relevant pages in this batch
            if not relevant_pages:
                continue

            batch_result = {
                'batch_idx': batch_idx,
                'timestamp_batch': timestamp_batch,
                'batch_documents': list(batch_documents),
                'retrieved_pages': list(retrieved_pages),
                'relevant_pages': list(relevant_pages),
                'metrics': {}
            }

            # Calculate k-dependent metrics including normalized metrics
            for k in k_values:
                k_metrics = calculate_metrics_with_normalization(retrieved_pages, relevant_pages, k)

                # Store in batch results
                batch_result['metrics'].update(k_metrics)

                # Collect for query-level averaging
                for metric_name, metric_value in k_metrics.items():
                    batch_metrics[metric_name].append(metric_value)

            query_results['batches'][batch_key] = batch_result

        # Calculate query-level averages
        for metric, values in batch_metrics.items():
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

        all_metrics = defaultdict(list)

        # Evaluate each query
        for query_num, results in results_dict.items():
            query_results = self.evaluate_query(query_num, results, k_values)
            evaluation_results['queries'][query_num] = query_results

            # Collect metrics for global averages
            for metric_key, metric_value in query_results['query_averages'].items():
                if 'avg_' in metric_key:
                    all_metrics[metric_key].append(metric_value)

        # Calculate global averages
        for metric_key, values in all_metrics.items():
            global_key = metric_key.replace('avg_', 'global_avg_')
            evaluation_results['global_averages'][global_key] = np.mean(values)

            global_std_key = metric_key.replace('avg_', 'global_std_')
            evaluation_results['global_averages'][global_std_key] = np.std(values)

        return evaluation_results
