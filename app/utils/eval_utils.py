import statistics
import time
from functools import wraps

from app.utils.app_utils import pprint_debug


def timed_operation(operation_name: str, timespan_aware: bool = False):
    """
    Decorator to time operations when benchmark_mode is enabled.
    Records timing data directly to the instance's timing_data attribute.

    Args:
        operation_name: Name of the operation for timing records
        timespan_aware: Whether this operation should track timespan_idx
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self.args, 'benchmark_mode', False):
                return func(self, *args, **kwargs)

            start_time = time.time()
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time

            timespan_idx = kwargs.get('timespan_idx') if timespan_aware else None

            _record_timing_data(self, operation_name, duration, timespan_idx)
            return result

        return wrapper

    return decorator


def _record_timing_data(instance, operation: str, duration: float, timespan_idx: int = None):
    """
    Record timing data for benchmark analysis.

    Args:
        instance: Object instance containing timing_data attribute
        operation: Name of the operation being timed
        duration: Duration of the operation in seconds
        timespan_idx: Optional timespan index for timespan-aware operations
    """
    if not hasattr(instance, 'timing_data'):
        return

    timing_entry = {
        'duration': duration,
        'timespan_idx': timespan_idx,
        'timestamp': time.time()
    }

    if operation in instance.timing_data:
        instance.timing_data[operation].append(timing_entry)


def debug_node_scores(nodes, stage_name):
    """Debug helper to print score statistics for nodes."""
    if not nodes:
        pprint_debug(f"{stage_name}: No nodes to analyze")
        return

    score_dict = {node.id_: node.score for node in nodes if hasattr(node, 'score')}
    scores = list(score_dict.values())

    if not scores:
        pprint_debug(f"{stage_name}: No scores available")
        return

    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    min_score = min(scores)
    max_score = max(scores)

    pprint_debug(f"{stage_name}: {len(nodes)} nodes - "
                 f"Mean: {mean_score:.4f}, Std: {std_score:.4f}, "
                 f"Min: {min_score:.4f}, Max: {max_score:.4f}")
