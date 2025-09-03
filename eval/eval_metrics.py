from typing import List, Set


def precision_at_k(retrieved_pages: List[str], relevant_pages: Set[str], k: int) -> float:
    """Calculate Precision@K at page level."""
    if not retrieved_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    relevant_retrieved = sum(1 for page in retrieved_list if page in relevant_pages)
    return relevant_retrieved / min(k, len(retrieved_list))


def recall_at_k(retrieved_pages: List[str], relevant_pages: Set[str], k: int) -> float:
    """Calculate Recall@K at page level."""
    if not relevant_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    relevant_retrieved = sum(1 for page in retrieved_list if page in relevant_pages)
    return relevant_retrieved / len(relevant_pages)


def hit_rate_at_k(retrieved_pages: List[str], relevant_pages: Set[str], k: int) -> float:
    """Calculate Hit Rate@K (binary: 1 if any relevant page found, 0 otherwise)."""
    if not relevant_pages:
        return 0.0

    retrieved_list = list(retrieved_pages)[:k]
    return 1.0 if any(page in relevant_pages for page in retrieved_list) else 0.0


def f1_at_k(retrieved_pages: List[str], relevant_pages: Set[str], k: int) -> float:
    """Calculate F1@K at page level."""
    if not relevant_pages or not retrieved_pages:
        return 0.0

    precision = precision_at_k(retrieved_pages, relevant_pages, k)
    recall = recall_at_k(retrieved_pages, relevant_pages, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics_with_normalization(retrieved_pages: List[str], relevant_pages: Set[str], k: int) -> dict:
    """
    Calculate metrics including normalized recall based on theoretical maximum.

    Args:
        retrieved_pages: List of retrieved page IDs
        relevant_pages: Set of relevant page IDs
        k: Number of top results to consider

    Returns:
        Dictionary containing standard and normalized metrics
    """
    if not retrieved_pages or not relevant_pages:
        return {f'precision@{k}': 0.0, f'recall@{k}': 0.0, f'f1@{k}': 0.0,
                f'hit_rate@{k}': 0.0, f'normalized_recall@{k}': 0.0,
                f'normalized_f1@{k}': 0.0, f'max_possible_recall@{k}': 0.0}

    retrieved_list = list(retrieved_pages)[:k]
    relevant_retrieved = sum(1 for page in retrieved_list if page in relevant_pages)

    # Standard metrics
    precision = relevant_retrieved / min(k, len(retrieved_list))
    recall = relevant_retrieved / len(relevant_pages)
    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0

    # Normalized variants
    max_possible_recall = min(k, len(relevant_pages)) / len(relevant_pages)
    normalized_recall = recall / min(1, max_possible_recall) if max_possible_recall > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    normalized_f1 = 2 * (precision * normalized_recall) / (precision + normalized_recall) \
        if (precision + normalized_recall) > 0 else 0.0

    return {
        f'precision@{k}': precision,
        f'recall@{k}': recall,
        f'f1@{k}': f1,
        f'hit_rate@{k}': hit_rate,
        f'normalized_recall@{k}': normalized_recall,
        f'normalized_f1@{k}': normalized_f1,
        f'max_possible_recall@{k}': max_possible_recall
    }


