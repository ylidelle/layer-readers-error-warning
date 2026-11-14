"""SPRINT marking and parsing (2026-11-14, after Lucien's review, points 3 and 4). Pure functions, no model.

PRIMARY label, correct_em: the model's answer, normalised, EQUALS one of TriviaQA's accepted aliases, normalised.
  Normalisation is TriviaQA's own (lower case; underscores, punctuation and the quote marks ‘ ’ ´ ` to spaces;
  the articles a/an/the removed; single spaces).  This is exact match on a short-answer prompt: a correct answer
  wrapped in extra words is marked WRONG.
SECONDARY label, correct_contain: an accepted alias appears in the answer as whole words. A custom, looser rule,
  kept only as a labelled sensitivity check: it accepts "Not Paris; London" for Paris."""
import re, string

_EXCLUDE = set(string.punctuation + "‘’´`")

def normalize_answer(s):
    s = s.replace("_", " ").lower()
    s = "".join(ch if ch not in _EXCLUDE else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()

def exact_match(answer, aliases):
    a = normalize_answer(answer)
    return bool(a) and any(a == normalize_answer(x) for x in aliases)

def contains_alias(answer, aliases):
    a = " " + normalize_answer(answer) + " "
    return any((" " + normalize_answer(x) + " ") in a for x in aliases if normalize_answer(x))

_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")

def parse_confidence(text):
    """The WHOLE reply must be one number from 0 to 100 (decimals allowed, optional % sign). Anything else is invalid
    (None). 0.95 is read literally as 0.95 on the 0-100 scale."""
    m = _NUM.match(text or "")
    if not m: return None
    v = float(m.group(1))
    return v if 0.0 <= v <= 100.0 else None
