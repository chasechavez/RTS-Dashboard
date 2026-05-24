"""
Fuzzy athlete name matching across CSV sources.

Handles:
  - "Last, First"  /  "First Last"  /  "SMITH, JOHN"
  - Initials: "J. Smith"  →  matches "John Smith"
  - Suffixes stripped: Jr, Sr, II, III, IV
  - Abbreviated first: "J Smith" matches "John Smith" (first-char prefix)
  - Similarity via SequenceMatcher; threshold configurable
"""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# ── Normalisation ─────────────────────────────────────────────────────────────
_SUFFIX_RE = re.compile(
    r"\b(jr\.?|sr\.?|ii+|iv|v|esq\.?)\b", re.IGNORECASE
)
_PUNCT_RE  = re.compile(r"[.,'\-]")
_SPACE_RE  = re.compile(r"\s+")


def normalize_name(raw: str) -> str:
    """
    Return a canonical lowercase 'first last' string.

    Steps:
      1. Strip suffixes (Jr, Sr, II, III, IV, V)
      2. Flip "Last, First" → "First Last"  ← BEFORE stripping punctuation
      3. Strip remaining punctuation (periods, hyphens, apostrophes)
      4. Collapse whitespace, lowercase
    """
    s = str(raw).strip()
    s = _SUFFIX_RE.sub("", s).strip()

    # "Last, First" → "First Last" — flip while comma still present
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    # Strip remaining punctuation (periods from initials, hyphens, etc.)
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip().lower()
    return s


def _tokens(name: str) -> List[str]:
    """Split normalised name into tokens, drop empty."""
    return [t for t in name.split() if t]


def similarity(a: str, b: str) -> float:
    """
    Composite similarity score in [0, 1].

    Uses SequenceMatcher on normalized names, with a bonus for
    initial-only matches (e.g. 'j smith' vs 'john smith').
    """
    na, nb = normalize_name(a), normalize_name(b)
    base = SequenceMatcher(None, na, nb).ratio()

    ta, tb = _tokens(na), _tokens(nb)
    if not ta or not tb:
        return base

    # Last-name exact match bonus
    last_match = float(ta[-1] == tb[-1])

    # First-name initial match: 'j' matches 'john'
    fa, fb = ta[0], tb[0]
    if len(fa) == 1 and fb.startswith(fa):
        initial_bonus = 0.15
    elif len(fb) == 1 and fa.startswith(fb):
        initial_bonus = 0.15
    else:
        initial_bonus = 0.0

    # Weighted composite
    score = 0.50 * base + 0.35 * last_match + 0.15 * initial_bonus
    # Clamp
    return min(score, 1.0)


def best_match(
    query: str,
    candidates: List[str],
    threshold: float = 0.72,
) -> Tuple[Optional[str], float]:
    """
    Return (best_candidate, score) for *query* against *candidates*.

    Returns (None, 0.0) if no candidate exceeds *threshold*.
    Exact normalised match always returns 1.0.
    """
    nq = normalize_name(query)
    best_name: Optional[str] = None
    best_score = 0.0

    for cand in candidates:
        if normalize_name(cand) == nq:
            return cand, 1.0
        s = similarity(query, cand)
        if s > best_score:
            best_score = s
            best_name = cand

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def build_name_map(
    source_names: List[str],
    target_names: List[str],
    threshold: float = 0.72,
) -> Dict[str, Tuple[Optional[str], float]]:
    """
    Map every name in *source_names* → best match in *target_names*.

    Returns:
        {source_name: (matched_target_name_or_None, score)}
    """
    result: Dict[str, Tuple[Optional[str], float]] = {}
    for sn in source_names:
        matched, score = best_match(sn, target_names, threshold)
        result[sn] = (matched, score)
    return result
