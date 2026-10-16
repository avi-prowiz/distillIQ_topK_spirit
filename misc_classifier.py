"""
Non-alcoholic / "Misc. Accessories & Gifts" detector.

The hard-truth file now carries ~16k rows labeled "Misc. Accessories &
Gifts" (food, soft drinks, mixers, accessories, value-added packs). Almost
all of them use that string as the item AND brand, so there is no catalog
item to match against; instead this module learns what such raw names look
like from the labeled rows and returns P(Misc) for a raw name.

Pure numpy at run time (no scikit-learn): hashed word / bigram / word-prefix
features into a logistic regression whose weights are stored in graph.pkl by
build_graph.py. MatchEngine.predict() combines this probability with the
catalog match score (see matching.py).
"""
import re
import zlib

import numpy as np

MISC_CATEGORY = "Misc. Accessories & Gifts"
DIM = 1 << 19

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_APOS_RE = re.compile(r"[’']")
_NUM_RE = re.compile(r"^\d+$")


# Names the classifier cannot learn from the labeled data alone: beer and wine
# brands/varietals the catalog barely carries, whose words also show up in
# Misc rows ("Corona" the beer vs a Corona-size cigar; "Beer" in "Ginger
# Beer"). A raw name containing one of these is only ever called Misc when
# it also names a clearly non-beverage thing (glass, cigar, sauce, ...).
ALCOHOL_HINT_RE = re.compile(
    r"\b(corona|modelo|heineken|budweiser|bud light|coors|miller|michelob|busch|pabst|stella artois|guinness|blue moon|"
    r"dos equis|sam adams|samuel adams|sierra nevada|lagunitas|fat tire|yuengling|natural light|keystone|tecate|pacifico|"
    r"lager|pilsner|ipa|stout|porter|hefeweizen|merlot|chardonnay|cabernet|pinot|riesling|moscato|prosecco|"
    r"sauvignon|zinfandel|malbec|champagne|rose wine|red wine|white wine|sparkling wine)\b", re.I)
NON_BEVERAGE_HINT_RE = re.compile(
    r"\b(glass|glasses|glassware|mug|mugs|opener|koozie|coozie|cooler|tumbler|shaker|cigar|cigars|cigarillo|cigarillos|"
    r"tobacco|cigarette|cigarettes|shirt|hat|cap|hoodie|sign|poster|coaster|stirrer|ornament|candle|lighter|sauce|"
    r"seasoning|marinade|dressing|cheese|chicken|beef|pork|bacon|sausage|pizza|olives|salsa|soup|noodles|cookies|"
    r"chocolate|candy|cake|bread|jerky|vinegar|decanter|corkscrew|bag|box|basket|apron|towel)\b", re.I)


def alcohol_hint_blocks_misc(raw):
    """True when the raw name carries an alcohol brand/varietal hint and
    nothing marking it as a non-beverage product."""
    s = str(raw)
    return bool(ALCOHOL_HINT_RE.search(s)) and not NON_BEVERAGE_HINT_RE.search(s)


def _norm_tokens(raw):
    s = _APOS_RE.sub("", str(raw).lower())
    toks = []
    for t in _TOKEN_RE.findall(s):
        toks.append("<n>" if _NUM_RE.match(t) else t)
    return toks


def feature_indices(raw):
    """Unique hashed feature ids for a raw name: unigrams, bigrams and
    4-letter prefixes of longer words (so "noodles"/"noodle" share signal)."""
    toks = _norm_tokens(raw)
    feats = {"__bias_doc__"}
    for i, t in enumerate(toks):
        feats.add("u:" + t)
        if len(t) >= 6 and t[0] != "<":
            feats.add("p:" + t[:4])
        if i + 1 < len(toks):
            feats.add("b:" + t + "_" + toks[i + 1])
    return np.fromiter((zlib.crc32(f.encode("utf-8")) & (DIM - 1) for f in feats), dtype=np.int64)


def _csr(raws):
    idx = [feature_indices(r) for r in raws]
    counts = np.array([len(a) for a in idx], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    indices = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
    return indptr, indices, counts


def _scores(w, b, indptr, indices):
    return np.add.reduceat(w[indices], indptr[:-1]) + b


def train(raws, labels, l2=5e-7, epochs=300, lr=0.15):
    """Logistic regression (Adam, full batch). labels: 1 = Misc, 0 = not."""
    y = np.asarray(labels, dtype=np.float64)
    indptr, indices, counts = _csr(raws)
    n = len(y)
    w = np.zeros(DIM, dtype=np.float64)
    b = 0.0
    m = np.zeros(DIM); v = np.zeros(DIM); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        z = _scores(w, b, indptr, indices)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) / n
        g = np.bincount(indices, weights=np.repeat(err, counts), minlength=DIM) + l2 * w
        gb = err.sum()
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        w -= lr * mh / (np.sqrt(vh) + eps)
        bh = mb / (1 - b1 ** t); vbh = vb / (1 - b2 ** t)
        b -= lr * bh / (np.sqrt(vbh) + eps)
    # Vocabulary seen at least twice in training, so a raw name made mostly of
    # words the model has never learned from is not asserted as Misc.
    seen = np.zeros(DIM, dtype=np.int32)
    for a in (indices[indptr[i]:indptr[i + 1]] for i in range(n)):
        seen[a] += 1
    return {"w": w.astype(np.float32), "b": float(b), "seen": (seen >= 2)}


def _word_hash(t):
    return zlib.crc32(("u:" + t).encode("utf-8")) & (DIM - 1)


class MiscModel:
    def __init__(self, model):
        self.w = np.asarray(model["w"], dtype=np.float32)
        self.b = float(model["b"])
        self.seen = np.asarray(model["seen"], dtype=bool) if "seen" in model else None

    def coverage(self, raw):
        """Share of the raw name's words (numbers and 1-letter words aside)
        the model saw at least twice in training; 1.0 if unknown."""
        if self.seen is None:
            return 1.0
        toks = [t for t in _norm_tokens(raw) if t != "<n>" and len(t) > 1]
        if not toks:
            return 0.0
        return sum(bool(self.seen[_word_hash(t)]) for t in toks) / len(toks)

    def prob(self, raw):
        idx = feature_indices(raw)
        z = float(self.w[idx].sum()) + self.b
        return 1.0 / (1.0 + np.exp(-max(-30.0, min(30.0, z))))

    def prob_many(self, raws):
        indptr, indices, _ = _csr(raws)
        z = _scores(self.w.astype(np.float64), self.b, indptr, indices)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
