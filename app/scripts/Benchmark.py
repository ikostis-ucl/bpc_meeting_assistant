import json
from typing import List, Dict

from app.scripts.Demo import Demo
from eval.benchmark_evaluator import BenchmarkEvaluator
from app.utils.benchmark_utils import BENCHMARK_QUESTIONS_INDEX
from eval.eval_inference import EvalInference


class Benchmark(Demo):
    """
    Benchmark class for evaluating the system with predefined questions.
    """

    def __init__(self):
        super().__init__()
        self.args.benchmark_mode = True  # Enable benchmark mode

        self.args.input_path = "./data/input"
        self.args.storage_dir = "./data/vector_db"
        if self.args.anon:
            self.args.input_path = "./data/input_anonymised"
            self.args.storage_dir = "./data/vector_db_anonymised"

        self.questions = list(BENCHMARK_QUESTIONS_INDEX.values())
        self.evaluator = BenchmarkEvaluator(self.args.benchmark_gt_path)

    def benchmark_eval(self, k_values: List[int]) -> Dict:  # Use realistic k values
        """Run benchmark evaluation with k values that match system design."""
        results_dict = {}

        for query_num, question in BENCHMARK_QUESTIONS_INDEX.items():
            print(f"Evaluating Query {query_num}: {question}")
            results = self.agent.evaluate_retriever(question)
            results_dict[query_num] = results

        aggregated_metrics = self.evaluator.evaluate_all_queries(results_dict, k_values)

        with open('./eval/results/benchmark_results.json', 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)

        return aggregated_metrics

    def run(self):
        """Execute evaluation workflow only."""
        print("Running benchmark evaluation...")
        metrics = self.benchmark_eval(k_values=[1, 3, 5])

        # Display key metrics
        print("\nBENCHMARK RESULTS:")
        print("=" * 40)
        for metric, value in metrics.items():
            if metric.startswith('avg_'):
                print(f"{metric}: {value:.4f}")

        return metrics


class BenchmarkRAG(Benchmark):
    def __init__(self):
        super().__init__()
        self.agent = EvalInference(args=self.args)
        self.agent.questions = self.questions
