import json
import os
from datetime import datetime
from typing import Dict

from app.engine.inference.groq_inference import GroqInference
from app.scripts.demo import Demo
from app.utils.app_utils import pprint_console, pprint_debug
from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX_RAG as Q_SET
# from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX as Q_SET


class BenchmarkRAG(Demo):
    """
    Benchmark class for RAG system evaluation.
    """

    def __init__(self):
        """
        Initialize configuration.
        Sets up paths and defines questions.
        """
        super().__init__()

        self.args.benchmark_mode = True

        self.query_results = []

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            pprint_console("Running in --anon mode.")
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        self.questions = list(Q_SET.values())
        self.agent = GroqInference(args=self.args)

    def run_single_query(self, query_idx: int, query: str):
        """Run a single query and store timing data."""
        result = self.agent.query_llm(query)

        self.query_results.append({
            'query_idx': query_idx,
            'query': query,
            'result': result,
            'timings': self._extract_query_timings()
        })

        return result

    def _extract_query_timings(self) -> Dict:
        """Extract timing data for the last query from agent."""
        if not hasattr(self.agent, 'timing_data'):
            return {}

        num_batches = len(self.agent.document_batches)
        timings = {}

        for operation, data_list in self.agent.timing_data.items():
            if operation == 'total_query_times':
                timings[operation] = data_list[-1]['duration'] if data_list else 0
            else:
                last_n = data_list[-num_batches:] if len(data_list) >= num_batches else data_list
                timings[operation] = [t['duration'] for t in last_n]

        return timings

    def _calculate_comprehensive_metrics(self) -> Dict:
        """Calculate per-query, per-batch, and global averages."""
        metrics = {
            'per_query': {},
            'per_batch_averages': {},
            'global_averages': {},
            'summary_stats': {}
        }

        for query_data in self.query_results:
            query_idx = query_data['query_idx']
            timings = query_data['timings']

            query_metrics = {}
            for operation, times in timings.items():
                if operation == 'total_query_times':
                    query_metrics[operation] = times
                else:
                    query_metrics[operation] = {
                        'per_batch': times,
                        'average': sum(times) / len(times) if times else 0,
                        'total': sum(times) if times else 0
                    }

            metrics['per_query'][f'query_{query_idx}'] = query_metrics

        all_operations = ['retrieval_times', 'reranker_times', 'synthesis_times', 'judge_times', 'total_query_times']

        for operation in all_operations:
            all_times = []
            for query_data in self.query_results:
                timings = query_data['timings'].get(operation, [])
                if operation == 'total_query_times':
                    if timings:
                        all_times.append(timings)
                else:
                    all_times.extend(timings if isinstance(timings, list) else [])

            if all_times:
                metrics['global_averages'][operation] = {
                    'count': len(all_times),
                    'average': sum(all_times) / len(all_times),
                    'total': sum(all_times),
                    'min': min(all_times),
                    'max': max(all_times)
                }

        num_batches = len(self.agent.document_batches) if hasattr(self.agent, 'document_batches') else 0

        for batch_idx in range(num_batches):
            batch_metrics = {}

            for operation in ['retrieval_times', 'reranker_times', 'synthesis_times']:
                batch_times = []
                for query_data in self.query_results:
                    timings = query_data['timings'].get(operation, [])
                    if batch_idx < len(timings):
                        batch_times.append(timings[batch_idx])

                if batch_times:
                    batch_metrics[operation] = {
                        'average': sum(batch_times) / len(batch_times),
                        'count': len(batch_times),
                        'min': min(batch_times),
                        'max': max(batch_times)
                    }

            metrics['per_batch_averages'][f'batch_{batch_idx}'] = batch_metrics

        return metrics

    def save_benchmark_results(self, filename: str = None):
        """Save comprehensive benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'./eval/results/rag/{timestamp}/benchmark_results.json'

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        metrics = self._calculate_comprehensive_metrics()

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(self.query_results),
                'batches_count': len(self.agent.document_batches) if hasattr(self.agent, 'document_batches') else 0
            },
            'queries': [
                {
                    'query_idx': qr['query_idx'],
                    'query': qr['query'],
                    'timings': qr['timings']
                } for qr in self.query_results
            ],
            'metrics': metrics
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        pprint_debug(f"Benchmark results saved to {filename}")
        return filename

    def print_timing_summary(self):
        """Print concise timing summary."""
        metrics = self._calculate_comprehensive_metrics()

        pprint_debug("\n" + "=" * 60)
        pprint_debug("BENCHMARK TIMING SUMMARY")
        pprint_debug("=" * 60)

        global_avg = metrics['global_averages']
        for operation, stats in global_avg.items():
            op_name = operation.replace('_', ' ').title()
            pprint_debug(f"{op_name:<20}: {stats['average']:.3f}s avg ({stats['count']} samples)")

        pprint_debug(f"\nTotal Queries: {len(self.query_results)}")

    def run(self):
        """Execute benchmark with timing collection."""
        print("Running RAG benchmark with timing analysis...")

        for i, question in enumerate(self.questions):
            pprint_console(f"Processing query {i + 1}/{len(self.questions)}: {question}")
            self.run_single_query(i, question)

        self.print_timing_summary()
        results_file = self.save_benchmark_results()

        pprint_console("Benchmark completed successfully!")
        return results_file
