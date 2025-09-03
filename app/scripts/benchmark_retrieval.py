import json
import os
from datetime import datetime
from typing import List, Dict

from app.scripts.demo import Demo
from app.utils.app_utils import pprint_console, pprint_debug
from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from eval.benchmark_evaluator import BenchmarkEvaluator
from eval.eval_inference import EvalInference
from eval.eval_visualization import visualize_benchmark_results


class BenchmarkRetrieval(Demo):
    """
    BenchmarkRetrieval class for evaluating the system with predefined questions.
    """

    def __init__(self):
        super().__init__()

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            pprint_console("Running in --anon mode.")
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        self.questions = list(BENCHMARK_QUESTIONS_INDEX.values())
        self.evaluator = None
        self.agent = EvalInference(self.args, self.questions)

    def benchmark_eval(self, k_values: List[int]) -> Dict:
        """Run benchmark evaluation with k values that match system design."""
        if self.evaluator is None:
            self.evaluator = BenchmarkEvaluator(
                self.args.benchmark_gt_path,
                self.agent.ts_doc_index,
                self.agent.document_batches
            )

        results_dict = {}
        for query_num, question in BENCHMARK_QUESTIONS_INDEX.items():
            pprint_debug(f"Evaluating Query {query_num}: {question}")
            results = self.agent.evaluate_retriever(question)
            results_dict[query_num] = results

        aggregated_metrics = self.evaluator.evaluate_all_queries(results_dict, k_values)

        return aggregated_metrics

    def run(self):
        """Execute evaluation workflow only."""
        pprint_console("Running benchmark evaluation...")
        metrics = self.benchmark_eval(k_values=[2, 3, 4, 5])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./eval/results/retrieval/{timestamp}/benchmark_results.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        visualize_benchmark_results(save_path)

        pprint_console("Benchmark evaluation completed, check results file for details.")

        return metrics
