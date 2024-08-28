import os
import shutil


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


def pprint_qa(question, answer, metadata):
    f_names = []
    for node_id, node_values in metadata.items():
        f_names.append(node_values["file_name"])
    f_names = list(set(f_names))

    pprint_hline("=")
    print(f"{Color.GREEN}Question:{Color.END}\n{question}")
    pprint_hline("-", 3)
    print(f"{Color.CYAN}Answer:{Color.END}\n{answer}")
    pprint_hline("-", 3)
    print(f"{Color.YELLOW}Citation:{Color.END}")
    print(f"{Color.BOLD}Document name(s):{Color.END} {f_names}")

