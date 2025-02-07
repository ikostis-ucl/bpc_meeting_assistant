import os
import shutil
import datetime

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def empty_dir(path):
    if os.path.exists(path) and not os.path.isfile(path):
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")
        exit()


def fmt_string(fmt, string):
    return f"{fmt}{string}{Color.END}"


def pprint_console(text_entry):
    print(f"\n{Color.BLUE}[CONSOLE] {text_entry}{Color.END}")


def pprint_error(text_entry):
    print(f"{Color.RED}\n[ERROR] {text_entry}{Color.END}")


def pprint_debug(text_entry):
    print(f"{Color.PURPLE}\n[DEBUG] {text_entry}{Color.END}")


def pprint_hline(token, length=shutil.get_terminal_size().columns):
    print(f'{token}' * length)


def simplify_path(path):
    path = path.replace("\\", "/")
    components = path.split("/")
    to_index = len(components) - 1 - components[::-1].index("data")
    simplified_path = "./" + "/".join(components[to_index:])

    return simplified_path

def datetime_to_timestamp(dt):
    if isinstance(dt, list) and len(dt) == 3:
        try:
            day, month, year = map(int, dt)
            dt_obj = datetime.datetime(year, month, day)
            return int(dt_obj.timestamp())
        except ValueError:
            raise ValueError("List elements must be integers representing [dd, mm, yyyy]")
    else:
        raise ValueError("Input must be a list [dd, mm, yyyy]")


def pprint_qa(question, results):
    pprint_hline("=")
    print(f"{Color.GREEN}Question:{Color.END}\n{question}")
    pprint_hline("-", 3)
    print(f"{Color.CYAN}Answers:{Color.END}")

    for response, metadata, (start_date, end_date) in results:
        start_date_str = datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d')
        end_date_str = datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d')
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
