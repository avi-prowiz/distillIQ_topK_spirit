"""
Name cleaning shared by the matcher's re-rank (matching.py): drops size, pack
count, proof, accents, apostrophes and filler words so two listings of the
same product compare equal.
"""
import re
import unicodedata

_APOS = re.compile(r"[’'`´]")
_UNIT = r"(?:ml|cl|lt|l|liters?|litres?|fl\.?\s*oz|fz|oz|ounces?|gal|gallons?)"
_SIZE_RE = re.compile(rf"(?<![a-z0-9])(?:\d+\s*-\s*)?\d+(?:\.\d+)?\s*{_UNIT}(?![a-z])")
_PACK_RE = re.compile(r"(?<![a-z0-9])\d+\s*(?:pk|packs?|ct|count|cans?|btls?|bottles?)(?![a-z])")
_PROOF_RE = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?\s*%?\s*(?:proof|abv|alc)(?![a-z])")
_TOKEN = re.compile(r"[a-z0-9]+")
_DROP = {"ml", "oz", "fl", "fz", "lt", "liter", "litre", "pk", "ct", "proof", "abv", "the", "a", "an", "of", "and", "in"}


def tokens(raw):
    s = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode("ascii").lower()
    s = _APOS.sub("", s).replace("&", " and ")
    s = _PROOF_RE.sub(" ", s)
    s = _SIZE_RE.sub(" ", s)
    s = _PACK_RE.sub(" ", s)
    return [t for t in _TOKEN.findall(s) if t not in _DROP]


def signature(raw):
    """Order-free identity of a raw name: its tokens, sorted and de-duplicated."""
    return " ".join(sorted(set(tokens(raw))))
