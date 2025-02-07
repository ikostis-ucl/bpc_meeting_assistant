import datetime

from fuzzywuzzy import fuzz


def is_match(text, phrase, threshold=80):
    if text == phrase:
        return True
    if phrase in text or text in phrase:
        return True
    similarity = fuzz.ratio(text, phrase)
    return similarity >= threshold


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
