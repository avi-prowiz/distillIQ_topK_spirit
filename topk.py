"""
distillIQ top-K spirit suggestions.

Given a raw item name, return the K most likely item names ("brand item_name", e.g.
"Jack Daniels Tennessee Honey Whiskey"), best first. When the raw name looks like a non-liquor /
accessory item, a "<Misc - not a Liquor>" option is part of the list.

    from topk import suggest, suggest_rows
    suggest("Jack Daniels Honey Whiskey 750 ml")          # -> list of 5 dicts
    suggest_rows([[1, "Jack Daniels Honey 750ml"]])        # -> [[id, raw, option1 .. option5]]

How it works (see README.md): the candidate pool is the hard-truth items plus the product names of
the brand-wise lists; candidates come from the TF-IDF word-weight graph (matching.py), from a memory
of the labeled raw names of the hard truth, and from the brand-lock rules; a small learned model
(a conditional-logit over 17 word/brand/memory features, numpy only) ranks them; the Misc model
(misc_classifier.py) decides whether and where the Misc option appears.
"""
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from math import log, sqrt
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

import rawsig
from matching import MatchEngine, APOSTROPHE_RE
from misc_classifier import MISC_CATEGORY, alcohol_hint_blocks_misc

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MISC_OPTION = "<Misc - not a Liquor>"

N_UNFILTERED, N_FILTERED, N_MEMORY = 30, 10, 8
FEATURES = ["cos", "cos_gap", "rank_inv", "filt_score", "filt_top1", "is_hard_truth", "log_count", "mem_cos", "mem_exact",
            "jaccard", "seq", "exact_name", "brand_ok", "item_cov", "raw_cov", "extra_words", "brand_in_raw"]
NF = len(FEATURES)

DEFAULT_MIX = {"tau_first": 0.6, "tau_inc": 0.25, "slot": 2, "hint_mult": 0.4, "cov_mult": 0.6}


def item_key(name):
    """Order-free identity of an item name (size, proof, case, accents, punctuation ignored)."""
    return " ".join(sorted(set(rawsig.tokens(name))))


def _bk(b):
    return APOSTROPHE_RE.sub("", str(b).lower())


class RawMemory:
    """Labeled raw names of the hard truth, by signature (rawsig tokens): for a query, the closest labeled
    raw name of every item -- the 'name mappings in the hard truth' evidence."""
    RARE_DF = 400

    def __init__(self, sig_votes):
        self.sigs = list(sig_votes)
        self.items = [list(sig_votes[s]) for s in self.sigs]
        from collections import Counter
        df = Counter()
        for s in self.sigs:
            df.update(s.split())
        n = len(self.sigs)
        self.df = df
        self.idf = {w: log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.vec, self.post = [], defaultdict(list)
        for i, s in enumerate(self.sigs):
            w = {t: self.idf[t] for t in s.split()}
            self.vec.append((w, sqrt(sum(x * x for x in w.values())) or 1.0))
            for t in w:
                self.post[t].append(i)
        self.unseen_pen = log(1 + n) ** 2 + 1

    def lookup(self, raw):
        """-> {item: best cosine of the query against a labeled raw name of that item}"""
        toks = sorted(set(rawsig.tokens(raw)))
        known = [t for t in toks if t in self.idf]
        if not known:
            return {}
        q = {t: self.idf[t] for t in known}
        qn = sqrt(sum(x * x for x in q.values()) + self.unseen_pen * sum(1 for t in toks if t not in self.idf)) or 1.0
        rare = sorted(known, key=lambda t: self.df[t])[:4]
        cand = set()
        for t in rare:
            if self.df[t] <= self.RARE_DF:
                cand.update(self.post[t])
        if not cand:
            cand.update(self.post[rare[0]])
        best = {}
        for i in cand:
            w, norm = self.vec[i]
            c = sum(q[t] * w[t] for t in q if t in w) / (qn * norm)
            for it in self.items[i]:
                if c > best.get(it, 0.0):
                    best[it] = c
        return best


def softmax_rank_train(X, mask, pos, iters=500, lr=0.05, l2=1e-4):
    """Conditional logit: P(item) = softmax over a query's candidates of w.x; maximise the probability mass
    on the correct candidate(s). X [Q,C,F] standardised, mask/pos [Q,C] bool. Pure numpy (Adam)."""
    Q, C, F = X.shape
    w = np.zeros(F)
    m = np.zeros(F); v = np.zeros(F)
    ok = pos.any(axis=1)
    X, mask, pos = X[ok], mask[ok], pos[ok]
    n = len(X)
    for t in range(1, iters + 1):
        s = np.where(mask, X @ w, -1e9)
        s = s - s.max(axis=1, keepdims=True)
        e = np.where(mask, np.exp(s), 0.0)
        p = e / e.sum(axis=1, keepdims=True)
        ppos = (p * pos).sum(axis=1, keepdims=True) + 1e-12
        ds = p - p * pos / ppos
        g = np.einsum("qc,qcf->f", ds, X) / n + l2 * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w


CAL_FEATURES = ["log_p", "p", "cos", "mem_cos", "jaccard", "brand_ok", "extra_words", "is_top1"]


def cal_vector(p, frow, is_top1):
    """Features of one ranked option for the confidence calibration (frow: its row of FEATURES)."""
    g = lambda n: frow[FEATURES.index(n)]
    return np.array([log(p + 1e-4), p, g("cos"), g("mem_cos"), g("jaccard"), g("brand_ok"), g("extra_words"), 1.0 if is_top1 else 0.0])


def fit_logistic(X, y, iters=400, lr=0.05, l2=1e-3):
    """Plain logistic regression (Adam). Returns (w, b, mu, sd) on standardised X."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    m = np.zeros_like(w); v = np.zeros_like(w); mb = vb = 0.0
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - y) / len(y) + l2 * w
        gb = float((p - y).mean())
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return w, b, mu, sd


def mix_options(ranked, p_misc, k, cfg):
    """ranked: [(item, prob)...] best first (candidates only). Returns the final option list (item or MISC_OPTION)
    of length <= k: Misc first when p_misc >= tau_first, else inserted at `slot` when p_misc >= tau_inc."""
    items = [r for r in ranked]
    if p_misc >= cfg["tau_first"]:
        return [(MISC_OPTION, p_misc)] + items[:k - 1]
    if p_misc >= cfg["tau_inc"]:
        slot = min(cfg["slot"], k - 1)
        out = items[:slot] + [(MISC_OPTION, p_misc)] + items[slot:]
        return out[:k]
    return items[:k]


class TopK:
    def __init__(self, model_dir=DATA_DIR, use_embeddings=True):
        model_dir = Path(model_dir)
        self.eng = MatchEngine(model_dir / "graph.pkl", use_embeddings=use_embeddings)
        with open(model_dir / "topk_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.item_source = meta["item_source"]
        self.item_count = meta["item_count"]
        self.memory = RawMemory(meta["mem"])
        self.w = meta.get("w")
        self.mu, self.sd = meta.get("mu"), meta.get("sd")
        self.mix = {**DEFAULT_MIX, **(meta.get("mix") or {})}
        self.cal = meta.get("cal")
        self._tok, self._rr = {}, {}
        self._cache = {}

    # ---- candidate features ----
    def _item_info(self, it):
        v = self._tok.get(it)
        if v is None:
            v = self._tok[it] = (set(rawsig.tokens(it)), self.eng._rr_tokens(it), " ".join(rawsig.tokens(it)))
        return v

    def candidates(self, raw):
        """-> (items [n], feature matrix [n, NF]); the unranked candidate pool for a raw name."""
        eng = self.eng
        s = eng._raw_dot(raw)
        dot, rn = s["dot"], s["raw_norm"]
        cos = {it: min(1.0, d / (rn * eng.item_norm[it])) for it, d in dot.items()}
        top = sorted(cos, key=lambda i: -cos[i])[:N_UNFILTERED]
        rank = {it: r for r, it in enumerate(top)}
        filt = eng._catalog_only(raw, topk=N_FILTERED)
        filt_score = {c["item"]: c["score"] for c in filt}
        mem = self.memory.lookup(raw)
        mem_top = sorted(mem, key=lambda i: -mem[i])[:N_MEMORY]
        pool = list(dict.fromkeys(top + [c["item"] for c in filt] + mem_top))
        if not pool:
            return [], np.zeros((0, NF))
        best_cos = cos[top[0]] if top else 0.0
        raw_toks = set(rawsig.tokens(raw))
        rt = eng._rr_tokens(raw)
        rseq = " ".join(rawsig.tokens(raw))
        locked = _bk(s["locked_brand"]) if s["locked_brand"] else None
        rtw = s["raw_term_weight"]
        F = np.zeros((len(pool), NF))
        for j, it in enumerate(pool):
            itoks, irr, iseq = self._item_info(it)
            c = cos.get(it, 0.0)
            iv = eng.item_vec.get(it, {})
            bl = eng._item_brand_lower.get(it, "")
            u = len(rt | irr) or 1
            F[j] = [
                c, c - best_cos, 1.0 / (1 + rank[it]) if it in rank else 0.0,
                filt_score.get(it, 0.0), 1.0 if filt and filt[0]["item"] == it else 0.0,
                1.0 if self.item_source.get(it) == "H" else 0.0, log(1 + self.item_count.get(it, 0)) / 5.0,
                mem.get(it, 0.0), 1.0 if mem.get(it, 0.0) >= 0.999 else 0.0,
                len(rt & irr) / u, fuzz.ratio(rseq, iseq) / 100.0, 1.0 if (rseq == iseq and rt == irr) else 0.0,
                (0.0 if locked is None else (1.0 if bl == locked else -1.0)),
                (sum(1 for w in iv if w in rtw) / len(iv)) if iv else 0.0,
                len(raw_toks & itoks) / (len(raw_toks) or 1),
                len(itoks - raw_toks) / (len(itoks) or 1),
                1.0 if (bl and set(rawsig.tokens(bl)) <= raw_toks) else 0.0,
            ]
        return pool, F

    def p_misc(self, raw):
        """How likely the raw name is a non-liquor / accessory item, after the beer-wine hint and unseen-word guards."""
        p = self.eng.misc_prob(raw)
        if p >= 1.0:
            return 1.0
        if alcohol_hint_blocks_misc(raw):
            p *= self.mix["hint_mult"]
        if self.eng.misc_model is not None and self.eng.misc_model.coverage(raw) < 0.3:
            p *= self.mix["cov_mult"]
        return float(p)

    def rank(self, items, F):
        """-> [(item, probability)] best first"""
        if not len(items):
            return []
        z = (F - self.mu) / self.sd
        s = z @ self.w
        s = s - s.max()
        e = np.exp(s)
        p = e / e.sum()
        order = np.argsort(-p)
        self._frows = {items[i]: F[i] for i in order}
        return [(items[i], float(p[i])) for i in order]

    def confidence(self, p, frow, is_top1):
        """Calibrated probability that this option is the right item (not just its share among the candidates)."""
        if self.cal is None:
            return p
        z = (cal_vector(p, frow, is_top1) - self.cal["mu"]) / self.cal["sd"]
        return float(1 / (1 + np.exp(-(z @ self.cal["w"] + self.cal["b"]))))

    # ---- public ----
    def suggest(self, raw, k=5):
        raw = "" if raw is None else str(raw).strip()
        if not raw:
            return []
        if raw not in self._cache:
            items, F = self.candidates(raw)
            ranked = self.rank(items, F)
            frows = self._frows if ranked else {}
            conf = {it: self.confidence(p, frows[it], i == 0) for i, (it, p) in enumerate(ranked)}
            self._cache[raw] = (ranked, self.p_misc(raw), conf)
        ranked, pm, conf = self._cache[raw]
        out = []
        for item, p in mix_options(ranked, pm, k, self.mix):
            if item == MISC_OPTION:
                out.append(dict(rank=len(out) + 1, name=MISC_OPTION, brand=MISC_CATEGORY, category=MISC_CATEGORY,
                                score=round(p, 4), confidence=round(p, 4), source="misc"))
            else:
                out.append(dict(rank=len(out) + 1, name=item, brand=self.eng.item_brand.get(item) or "",
                                category=self.eng.item_cat.get(item) or "", score=round(p, 4),
                                confidence=round(conf[item], 4),
                                source="hard_truth" if self.item_source.get(item) == "H" else "brand_list"))
        return out


_model = None


def get_model():
    global _model
    if _model is None:
        _model = TopK()
    return _model


def suggest(raw, k=5):
    """Top-k options for one raw name: list of dicts (rank, name, brand, category, score, confidence, source).
    score = share among the candidates; confidence = calibrated probability that this option is the right item."""
    return get_model().suggest(raw, k)


def suggest_rows(rows, k=5):
    """rows: [[id, raw_item_name], ...] -> [[id, raw_item_name, option_1, ..., option_k]] (same order, one row per input;
    options are "brand item_name" strings or <Misc - not a Liquor>, blank when there are fewer than k)."""
    m = get_model()
    out = []
    for r in rows:
        opts = [o["name"] for o in m.suggest(r[1], k)]
        out.append([r[0], r[1]] + opts + [""] * (k - len(opts)))
    return out


def columns(k=5):
    return ["id", "raw_item_name"] + [f"option_{i}" for i in range(1, k + 1)]


def main(argv):
    """python topk.py input.(json|csv) [output.(json|csv)] [k]   -- input rows are [id, raw_item_name]"""
    if len(argv) < 2:
        print(main.__doc__)
        return 1
    src = Path(argv[1])
    if src.suffix.lower() == ".csv":
        with open(src, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
        if rows and rows[0][0].strip().lower() in ("id", "item_id", "row_id"):
            rows = rows[1:]
        rows = [[r[0], r[1] if len(r) > 1 else ""] for r in rows]
    else:
        rows = json.loads(src.read_text(encoding="utf-8"))
    k = int(argv[3]) if len(argv) > 3 else 5
    result = suggest_rows(rows, k)
    if len(argv) > 2 and argv[2].lower().endswith(".csv"):
        with open(argv[2], "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(columns(k))
            w.writerows(result)
        print(f"{len(result)} rows -> {argv[2]}")
    elif len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(result)} rows -> {argv[2]}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
