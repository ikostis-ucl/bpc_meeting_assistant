import datetime

from fuzzywuzzy import fuzz


def is_match(text, phrase, threshold=80):
    """
    Check if two strings match based on a similarity threshold.

    Args:
        text (str): First string.
        phrase (str): Second string.
        threshold (int): Similarity threshold (default is 80).

    Returns:
        bool: True if the strings match, False otherwise.
    """
    if text == phrase:
        return True
    if phrase in text or text in phrase:
        return True
    similarity = fuzz.ratio(text, phrase)
    return similarity >= threshold


def datetime_to_timestamp(dt):
    """
    Convert a date list [dd, mm, yyyy] to a timestamp.

    Args:
        dt (list): List containing day, month, and year.

    Returns:
        int: Corresponding timestamp.

    Raises:
        ValueError: If the input list is not valid.
    """
    if isinstance(dt, list) and len(dt) == 3:
        try:
            day, month, year = map(int, dt)
            dt_obj = datetime.datetime(year, month, day)
            return int(dt_obj.timestamp())
        except ValueError:
            raise ValueError("List elements must be integers representing [dd, mm, yyyy]")
    else:
        raise ValueError("Input must be a list [dd, mm, yyyy]")
