import time
from functools import wraps
import sys
from halo import Halo

from app.utils.app_utils import fmt_string, Color


def throttle_requests():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            if self.token_counter.total_llm_token_count >= self.model_tpm:
                sleep_time = 60 - elapsed_time
                if sleep_time > 0:
                    with Halo(text=fmt_string(Color.YELLOW, '[CONSOLE] API rate limit reached, waiting...'),
                              placement='right', animation='bounce', spinner='dots'):
                        time.sleep(sleep_time)
                self.token_counter.reset_counts()
            return result

        return wrapper

    return decorator

def throttle_cross_query_requests(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed_time = time.time() - start_time
        remainder = elapsed_time % 60
        sleep_time = 60 - remainder
        if sleep_time > 0:
            for remaining in range(int(sleep_time), 0, -1):
                sys.stdout.write("\r")
                sys.stdout.write(f"Waiting for {remaining} seconds...")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\rWaiting complete!                \n")
        return result
    return wrapper