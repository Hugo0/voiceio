"""Convert spoken number words to digits in transcribed text.

English only for v1. Handles cardinals up to 999,999,999, percentages,
and basic ordinals. Zero external dependencies.
"""
from __future__ import annotations

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_ALL_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES) | {"and", "a"}

_ORDINAL_SUFFIXES = {
    1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd",
    31: "st", 32: "nd", 33: "rd",
}

_ORDINAL_WORDS = {
    "first": (1, "st"), "second": (2, "nd"), "third": (3, "rd"),
    "fourth": (4, "th"), "fifth": (5, "th"), "sixth": (6, "th"),
    "seventh": (7, "th"), "eighth": (8, "th"), "ninth": (9, "th"),
    "tenth": (10, "th"), "eleventh": (11, "th"), "twelfth": (12, "th"),
    "thirteenth": (13, "th"), "fourteenth": (14, "th"),
    "fifteenth": (15, "th"), "sixteenth": (16, "th"),
    "seventeenth": (17, "th"), "eighteenth": (18, "th"),
    "nineteenth": (19, "th"), "twentieth": (20, "th"),
    "thirtieth": (30, "th"), "fortieth": (40, "th"),
    "fiftieth": (50, "th"), "sixtieth": (60, "th"),
    "seventieth": (70, "th"), "eightieth": (80, "th"),
    "ninetieth": (90, "th"), "hundredth": (100, "th"),
    "thousandth": (1000, "th"), "millionth": (1_000_000, "th"),
}


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for a number."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return _ORDINAL_SUFFIXES.get(n % 10, "th")


def _words_to_number(words: list[str]) -> int | None:
    """Convert a list of number words to an integer.

    Handles: "three hundred forty two" → 342
             "two thousand five hundred" → 2500
             "a hundred" → 100
    """
    if not words:
        return None

    result = 0
    current = 0

    for word in words:
        low = word.lower()

        if low == "and":
            continue

        if low == "a":
            current = 1
            continue

        if low in _ONES:
            current += _ONES[low]
        elif low in _TENS:
            current += _TENS[low]
        elif low == "hundred":
            current = (current or 1) * 100
        elif low == "thousand":
            current = (current or 1) * 1000
            result += current
            current = 0
        elif low == "million":
            current = (current or 1) * 1_000_000
            result += current
            current = 0
        elif low == "billion":
            current = (current or 1) * 1_000_000_000
            result += current
            current = 0
        else:
            return None

    return result + current


def _is_number_word(word: str) -> bool:
    """Check if a word is part of a number expression."""
    return word.lower() in _ALL_NUMBER_WORDS


def convert_numbers(text: str, language: str = "en") -> str:
    """Replace spoken number words with digits in text.

    Only processes English. Other languages pass through unchanged.
    """
    if language not in ("en", "auto"):
        return text
    if not text:
        return text

    words = text.split()
    result: list[str] = []
    i = 0

    while i < len(words):
        # Check for ordinal words (standalone like "first", "twentieth")
        low = words[i].lower().rstrip(".,;:?!")
        trailing_punct = words[i][len(low):] if len(words[i]) > len(low) else ""

        if low in _ORDINAL_WORDS and not _is_number_word_at(words, i + 1):
            val, suffix = _ORDINAL_WORDS[low]
            result.append(f"{val}{suffix}{trailing_punct}")
            i += 1
            continue

        # Check for "a hundred/thousand/million" pattern
        if low == "a" and i + 1 < len(words):
            next_w = words[i + 1].lower().rstrip(".,;:?!")
            if next_w in _SCALES:
                low = next_w  # fall through to number collection

        # Collect consecutive number words
        if _is_number_word(low) and low != "a" and low != "and":
            num_words = []
            j = i
            while j < len(words):
                w = words[j].lower().rstrip(".,;:?!")
                if _is_number_word(w):
                    # "a" only valid as "a hundred", "a thousand" etc.
                    if w == "a" and j > i:
                        break
                    if w == "a" and j == i:
                        # "a" at start: only if followed by scale word
                        if j + 1 < len(words) and words[j + 1].lower().rstrip(".,;:?!") in _SCALES:
                            num_words.append(w)
                            j += 1
                            continue
                        break
                    if w == "and":
                        # "and" only valid between number parts
                        if j + 1 < len(words) and _is_number_word(words[j + 1].lower().rstrip(".,;:?!")):
                            num_words.append(w)
                            j += 1
                            continue
                        break
                    num_words.append(w)
                    j += 1
                else:
                    break

            if num_words:
                # Check for trailing "percent"
                percent = False
                if j < len(words) and words[j].lower().rstrip(".,;:?!") == "percent":
                    percent = True
                    trail_p = words[j][len("percent"):]
                    j += 1
                else:
                    trail_p = ""

                # Capture trailing punctuation from the last number word
                last_raw = words[j - 1 - (1 if percent else 0)]
                last_clean = last_raw.lower().rstrip(".,;:?!")
                num_trailing = last_raw[len(last_clean):] if not percent else ""

                # Check for ordinal ending
                ordinal = False
                if num_words and num_words[-1] in _ORDINAL_WORDS:
                    val, suffix = _ORDINAL_WORDS[num_words[-1]]
                    # Convert preceding words + ordinal value
                    if len(num_words) > 1:
                        base = _words_to_number(num_words[:-1])
                        if base is not None:
                            n = base + val
                            result.append(f"{n}{_ordinal_suffix(n)}{num_trailing}")
                            i = j
                            ordinal = True
                    if not ordinal:
                        result.append(f"{val}{suffix}{num_trailing}")
                        i = j
                        ordinal = True

                if not ordinal:
                    num = _words_to_number(num_words)
                    if num is not None:
                        s = str(num)
                        if percent:
                            s += "%" + trail_p
                        else:
                            s += num_trailing
                        result.append(s)
                        i = j
                    else:
                        result.append(words[i])
                        i += 1
                continue

        result.append(words[i])
        i += 1

    return " ".join(result)


def _is_number_word_at(words: list[str], idx: int) -> bool:
    """Check if word at index is a number word."""
    if idx >= len(words):
        return False
    return _is_number_word(words[idx].lower().rstrip(".,;:?!"))
