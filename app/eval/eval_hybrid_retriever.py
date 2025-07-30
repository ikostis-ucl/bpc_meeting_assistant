def evaluate_retrieval_methods(inference_instance, query_string: str, ground_truth_doc_ids: set = None):
    """
    Compare different retrieval methods performance for a given inference instance.

    Args:
        inference_instance: Instance of BaseInference class to test.
        query_string: Query to test.
        ground_truth_doc_ids: Known relevant document IDs for evaluation.

    Returns:
        dict: Performance metrics for each method.
    """
    methods = {
        "dense_only": 1.0,
        "sparse_only": 0.0,
        "hybrid_balanced": 0.5,
        "hybrid_dense_heavy": 0.7,
        "hybrid_sparse_heavy": 0.3
    }

    results = {}

    for method_name, alpha in methods.items():
        method_results = inference_instance.query_llm_hybrid_enhanced(query_string, alpha=alpha)

        retrieved_ids = set()
        total_scores = []

        for _, metadata, _ in method_results:
            retrieved_ids.update(metadata.keys())
            total_scores.extend([data['score'] for data in metadata.values()])

        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0

        if ground_truth_doc_ids:
            precision = len(retrieved_ids.intersection(ground_truth_doc_ids)) / len(
                retrieved_ids) if retrieved_ids else 0
            recall = len(retrieved_ids.intersection(ground_truth_doc_ids)) / len(
                ground_truth_doc_ids) if ground_truth_doc_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[method_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_score": avg_score,
                "retrieved_count": len(retrieved_ids)
            }
        else:
            results[method_name] = {
                "avg_score": avg_score,
                "retrieved_count": len(retrieved_ids)
            }

    return results
