"""
Build the top-K model data from the hard truth + the brand-wise product lists.

    python build_topk.py "path\\to\\super-set-map-v2.1.xlsx" "path\\to\\brand wise items.zip" \\
        [--extra-negatives pseudo_negatives.txt] [--out-dir data] [--folds 5]

Steps: (1) catalog = hard-truth items + brand-list products not already in it; (2) word-weight graph,
Misc model and labeled-raw-name memory over that catalog; (3) out-of-fold candidate features
(`--folds` rebuilds, never scoring a raw name against its own labels), from which the ranking model and
the Misc placement thresholds are fitted and written into data/topk_meta.pkl.
evaluate_topk.py reports how well that works.
"""
import argparse
import pickle
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import brand_lists as bl
import build_graph as bg
import rawsig
import textfix
from misc_classifier import MISC_CATEGORY, train as train_misc_model


def load_truth(path, drop_suspect=True):
    """Hard-truth rows. Alcohol-labeled rows whose raw name shares no word at all with their own item name
    ("Red Bull, Original" -> Long Drink Peach; ~30 rows) are labeling errors: they would teach the model
    wrong name mappings, so they are left out (returned in df.attrs['suspect'])."""
    df = bg.load_items(Path(path))
    # one spelling per product: repair broken encodings (UTF-8 read as cp1252 / GBK, lost letters) and fold accents
    vocab = {w for it in df["item"].unique() for w in rawsig.tokens(it)} | {w for b in df["brand"].unique() for w in rawsig.tokens(b)}
    for col in ("item", "brand"):
        df[col] = [textfix.clean_text(v) for v in df[col]]
    df["raw"] = [textfix.clean_text(v, vocab) for v in df["raw"]]
    df, dropped = bg.drop_conflicting_misc(df)
    df["key"] = df["raw"].map(bg.norm_key)
    alc = df["category"] != MISC_CATEGORY
    zero = pd.Series([alc_ and not (set(rawsig.tokens(r)) & set(rawsig.tokens(i))) for alc_, r, i in zip(alc, df["raw"], df["item"])], index=df.index)
    df.attrs["suspect"] = df[zero]
    if drop_suspect:
        df = df[~zero].reset_index(drop=True)
        df.attrs["suspect"] = df.attrs.get("suspect")
    return df


def brand_items(truth, brand_path):
    """Candidate items from the brand lists (whole table; build_model drops the ones the hard truth already has)."""
    alc = truth[truth.category != MISC_CATEGORY]
    known = {}
    for b in alc.brand.value_counts().index:
        known.setdefault(bl._brand_key(b), b)
    brand_cat = alc.groupby("brand").category.agg(lambda s: s.value_counts().index[0]).to_dict()
    return bl.to_items(bl.load_brand_products(brand_path), known, brand_cat)


SKU_WORDS = {"pack", "gift", "set", "with", "kit", "glass", "glasses", "can", "cans", "bottle", "bottles"}


GENERIC_WORDS = {"vodka", "tequila", "rum", "whiskey", "whisky", "bourbon", "gin", "mezcal", "liqueur", "rye", "scotch",
                 "spirit", "spirits", "cocktail", "cocktails", "original", "classic", "hard", "seltzer", "flavored",
                 "flavoured", "can", "cans", "bottle", "bottles", "straight", "kentucky", "tennessee", "blended", "irish"}


def _near_duplicate(toks, same_brand_sets):
    """A brand-list name that is a hard-truth item under another wording adds nothing but a duplicate option, so
    it is not added: >= 80% of the words shared, or one name's words all inside the other with only generic
    type words ("Vodka", "Tequila", "Original") left over -- and the same pack/gift/with markers."""
    for h in same_brand_sets:
        if (toks & SKU_WORDS) != (h & SKU_WORDS):
            continue
        if len(toks & h) / (len(toks | h) or 1) >= 0.8:
            return True
        if toks <= h and (h - toks) <= GENERIC_WORDS:
            return True
        if h <= toks and (toks - h) <= GENERIC_WORDS:
            return True
    return False


def build_model(train, B, out_dir, extra_negatives=None):
    """Write graph.pkl + topk_meta.pkl (without ranking weights) for catalog = items of `train` + new items of B."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, item_brand, item_cat = bg.build_catalog(train)
    source = {it: "H" for it in items}
    hkeys = {" ".join(sorted(set(rawsig.tokens(it)))) for it in items}
    added = 0
    by_brand = defaultdict(list)
    for it in items:
        by_brand[bl._brand_key(item_brand.get(it, ""))].append(set(rawsig.tokens(it)))
    for r in B.itertuples(index=False):
        key = " ".join(sorted(set(rawsig.tokens(r.item))))
        if key in hkeys or r.item in source:
            continue
        if _near_duplicate(set(rawsig.tokens(r.item)), by_brand.get(bl._brand_key(r.brand), ())):
            continue
        hkeys.add(key)
        items.append(r.item)
        item_brand[r.item], item_cat[r.item] = r.brand, r.category
        source[r.item] = "B"
        by_brand[bl._brand_key(r.brand)].append(set(rawsig.tokens(r.item)))
        added += 1
    items = sorted(items)
    graph = bg.build_graph(items, item_brand, item_cat, use_embeddings=True)

    is_misc = train["category"] == MISC_CATEGORY
    raws, labels = train["raw"].tolist(), is_misc.astype(int).tolist()
    if extra_negatives:
        known = {bg.norm_key(r) for r in raws}
        extra = [textfix.clean_text(ln) for ln in Path(extra_negatives).read_text(encoding="utf-8").splitlines()]
        extra = [r for r in extra if r and bg.norm_key(r) not in known]
        raws += extra
        labels += [0] * len(extra)
    graph["misc_model"] = train_misc_model(raws, labels)
    placeholder = train.apply(lambda r: bg.is_misc_placeholder(r["item"], r["brand"], r["category"]), axis=1)
    real_named = {bg.norm_key(r) for r in train.loc[is_misc & ~placeholder, "raw"]}
    graph["misc_exact"] = frozenset(k for k in (bg.norm_key(r) for r in train.loc[placeholder, "raw"]) if k not in real_named)
    with open(out_dir / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    lab = train[~placeholder]
    mem = defaultdict(Counter)
    for raw, item in zip(lab["raw"], lab["item"]):
        sig = rawsig.signature(raw)
        if sig:
            mem[sig][item] += 1
    count = Counter(lab["item"])
    meta = dict(item_source=source, item_count=dict(count), mem={k: dict(v) for k, v in mem.items()})
    with open(out_dir / "topk_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return added, len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", type=Path)
    ap.add_argument("brands", type=Path, nargs="?", default=Path(__file__).resolve().parent / "brand wise items.zip",
                    help="'brand wise items.zip' or the unzipped folder (default: the zip in this folder)")
    ap.add_argument("--extra-negatives", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", type=Path, default=None, help="folder for the per-fold models/features (reused when present)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import evaluate_topk
    evaluate_topk.fit_and_save(args)


if __name__ == "__main__":
    main()
