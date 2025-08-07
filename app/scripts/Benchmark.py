import json
import os
from datetime import datetime
from typing import List, Dict

from app.scripts.Demo import Demo
from app.utils.app_utils import pprint_console
from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from eval.benchmark_evaluator import BenchmarkEvaluator
from eval.eval_inference import EvalInference
from eval.eval_utils import visualize_benchmark_results


class Benchmark(Demo):
    """
    Benchmark class for evaluating the system with predefined questions.
    """

    def __init__(self):
        super().__init__()
        self.args.benchmark_mode = True

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        self.questions = list(BENCHMARK_QUESTIONS_INDEX.values())
        self.evaluator = None  # Will be initialized after agent creation

    def benchmark_eval(self, k_values: List[int]) -> Dict:
        """Run benchmark evaluation with k values that match system design."""
        # Initialize evaluator with agent's timespan information
        if self.evaluator is None:
            self.evaluator = BenchmarkEvaluator(
                self.args.benchmark_gt_path,
                self.agent.ts_doc_index,
                self.agent.timespans
            )

        results_dict = {}
        for query_num, question in BENCHMARK_QUESTIONS_INDEX.items():
            print(f"Evaluating Query {query_num}: {question}")
            results = self.agent.evaluate_retriever(question)
            results_dict[query_num] = results

        aggregated_metrics = self.evaluator.evaluate_all_queries(results_dict, k_values)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./eval/results/{timestamp}/benchmark_results.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)

        visualize_benchmark_results(save_path)

        return aggregated_metrics

    def run(self):
        """Execute evaluation workflow only."""
        print("Running benchmark evaluation...")
        metrics = self.benchmark_eval(k_values=[1, 3, 5])

        pprint_console("Benchmark completed, check results file for details.")

        return metrics


class BenchmarkRAG(Benchmark):
    def __init__(self):
        super().__init__()
        self.agent = EvalInference(args=self.args)
        self.agent.questions = self.questions
