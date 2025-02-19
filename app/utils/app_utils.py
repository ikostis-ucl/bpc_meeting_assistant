import os
import shutil


class Color:
    """
    ANSI color codes for terminal text formatting.
    """
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
    """
    Check if a directory is empty.

    Args:
        path (str): Path to the directory.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    if os.path.exists(path) and not os.path.isfile(path):
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")
        exit()


def fmt_string(fmt, string):
    """
    Args:
        fmt (str): ANSI color code.
        string (str): String to format.

    Returns:
        str: Formatted string.
    """
    return f"{fmt}{string}{Color.END}"


def pprint_console(text_entry):
    """
    Print a message to the console with a blue prefix.

    Args:
        text_entry (str): Message to print.
    """
    print(f"\n{Color.BLUE}[CONSOLE] {text_entry}{Color.END}")


def pprint_error(text_entry):
    """
    Print an error message to the console with a red prefix.

    Args:
        text_entry (str): Error message to print.
    """
    print(f"{Color.RED}\n[ERROR] {text_entry}{Color.END}")


def pprint_debug(text_entry):
    """
    Print a debug message to the console with a purple prefix.

    Args:
        text_entry (str): Debug message to print.
    """
    print(f"{Color.PURPLE}\n[DEBUG] {text_entry}{Color.END}")


def pprint_hline(token, length=shutil.get_terminal_size().columns):
    """
    Print a horizontal line in the console.

    Args:
        token (str): Character to use for the line.
        length (int): Length of the line.
    """
    print(f'{token}' * length)


def simplify_path(path):
    """
    Simplify a file path to start from the 'data' directory.

    Args:
        path (str): Original file path.

    Returns:
        str: Simplified file path.
    """
    path = path.replace("\\", "/")
    components = path.split("/")
    to_index = len(components) - 1 - components[::-1].index("data")
    simplified_path = "./" + "/".join(components[to_index:])
    return simplified_path
