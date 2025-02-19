import datetime
import sys
import time
from functools import wraps

from halo import Halo

from app.utils.app_utils import fmt_string, Color, pprint_hline


def throttle_requests():
    """
    Decorator to throttle requests based on token usage.

    Returns:
        function: Wrapped function with throttling.
    """

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
    """
    Decorator to throttle cross-query requests.
    To be used in app.scripts.Demo.Demo.run() if you run into an issue with the API's rate limiters.

    Args:
        func (function): Function to wrap.

    Returns:
        function: Wrapped function with throttling.
    """

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
        return result

    return wrapper


def pprint_qa(question, results):
    """
    Pretty print a question and its answers.

    Args:
        question (str): The question asked.
        results (list): List of tuples containing (response, metadata, timespan).
    """
    pprint_hline("=")
    print(f"{Color.GREEN}Question:{Color.END}\n{question}")
    pprint_hline("-", 3)
    print(f"{Color.CYAN}Answers:{Color.END}")

    for response, metadata, (start_date, end_date) in results:
        start_date_str = datetime.datetime.fromtimestamp(start_date).strftime('%d/%m/%Y')
        end_date_str = datetime.datetime.fromtimestamp(end_date).strftime('%d/%m/%Y')
        print(f"{Color.DARKCYAN}{start_date_str} - {end_date_str}:{Color.END} {response}")

        if metadata:
            file_pages = {}
            for node_id, node_values in metadata.items():
                file_name = node_values["file_name"]
                page_number = node_values["page_number"]
                if file_name in file_pages:
                    file_pages[file_name].append(page_number)
                else:
                    file_pages[file_name] = [page_number]

            print(f"{Color.YELLOW}Citations:{Color.END}")
            print(f"{Color.BOLD}Document name(s) and page number(s):{Color.END}")
            for file_name, page_numbers in file_pages.items():
                print(f"{file_name}: {list(set(page_numbers))}")
        pprint_hline("-", 3)
