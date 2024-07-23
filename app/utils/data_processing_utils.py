from fuzzywuzzy import fuzz


def is_match(text, phrase, threshold=80):
    if text == phrase:
        return True
    if phrase in text or text in phrase:
        return True
    similarity = fuzz.ratio(text, phrase)
    return similarity >= threshold
