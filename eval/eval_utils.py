from typing import List, Set

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
