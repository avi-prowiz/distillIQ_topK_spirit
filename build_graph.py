"""
Rebuild data/graph.pkl and data/reference.json from a new source catalog.

This is the offline pipeline step described in CLAUDE.md as "not yet
built" -- a way to regenerate the reference graph and hard-truth data when
the source catalog changes, without having to hand-edit the two derived
files the app actually loads. It reproduces the same graph shape and
TF-IDF construction that matching.py/MatchEngine expects (see its
docstring / design doc for the field meanings): items, item_brand,
item_cat, item_vec, item_norm, inv_index, idf, canon_vocab, emb_words,
emb_matrix, STOP.

Usage:
    cd backend
    .venv/Scripts/python.exe build_graph.py <path-to-source.xlsx> [--out-dir ../data]

The source file is expected to have the same two-sheet shape as the
super-set-map workbook: an "Items" sheet with columns "raw item name",
"cleaned item name", "cleaned brand name", "cleaned category", "Source",
and a "Brands+Category" sheet (read but not required -- the per-item
brand/category on the Items sheet is authoritative; Brands+Category is
informational only and not currently cross-checked against it).
"""
import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from misc_classifier import MISC_CATEGORY, train as train_misc_model

TOKEN_RE = re.compile(r"[a-z0-9]+")
APOSTROPHE_RE = re.compile(r"[’']")

# ---- size parsing: item_size / pack_count / unit, mirroring matching.py's
# parse_size() exactly (kept in sync with it -- see that module's comments
# for the reasoning behind the tight-hyphen distinction and the 1-24
# pack-count bound). Not "proof": that's ABV, not a size.
_UNIT_ALT = r"(ml|lt|l|liter|litre|fl\.?\s*oz|fz|oz)"
_PACK_ALT = r"(?:pk|packs|pack|ct)"
_PACK_NUM = r"(?:[1-9]|1\d|2[0-4])"
_SIZE_COMPOUND_A_RE = re.compile(rf"\b({_PACK_NUM})-(\d+(?:\.\d+)?)\s*{_UNIT_ALT}\b", re.I)
_SIZE_COMPOUND_B_RE = re.compile(rf"\b({_PACK_NUM})\s*{_PACK_ALT}\b.{{0,25}}?(\d+(?:\.\d+)?)\s*{_UNIT_ALT}\b", re.I)
_SIZE_COMPOUND_C_RE = re.compile(rf"\b(\d+(?:\.\d+)?)\s*{_UNIT_ALT}\b\s*-\s*({_PACK_NUM})\s*{_PACK_ALT}\b", re.I)
_SIZE_STANDALONE_RE = re.compile(rf"\b(\d+(?:\.\d+)?)\s*{_UNIT_ALT}\b", re.I)
_PACK_STANDALONE_RE = re.compile(rf"\b({_PACK_NUM})\s*{_PACK_ALT}\b", re.I)

# Same stopword list matching.py's tokenizer already assumes (structural
# words -- units, packaging, connectors -- not domain content). Kept
# identical across rebuilds so the two stay in sync.
#
# "each" added after it was found to hash-collide with the unrelated word
# "extra" via the embedding tier (spaCy's medium model shares a reduced
# vector table across a much larger vocabulary -- the same root cause as
# "ult"->"ps" and "cello"->"alto", see matching.py's notes on
# MIN_EMBEDDING_LEN): raw text ending "... 50ml (Each)" was matching
# "each" to "extra" at ~1.0 similarity, a spurious term with nothing to do
# with the actual item. No catalog item uses "each" as a real word, so
# stopping it entirely costs nothing.
STOP = {
    'a', 'and', 'bottle', 'bottles', 'can', 'cans', 'ct', 'each', 'for',
    'gift', 'in', 'l', 'liter', 'litre', 'ml', 'of', 'oz', 'pack', 'pk',
    'proof', 'set', 'the', 'with',
}

# Category strings that are the same category spelled/cased/spaced
# differently -- the same normalization CLAUDE.md documents for the
# original catalog (SS2: "Vodkka" -> "Vodka", "Ready-to-Drink" ->
# "Ready-To-Drink"), re-derived from what this new file actually contains.
CATEGORY_FIXES = {
    'vodkka': 'Vodka',
    'ready-to-drink': 'Ready-To-Drink',
    'cocktail': 'Prepared Cocktails',
}


def fix_mojibake(s: str) -> str:
    """Repair UTF-8-decoded-as-CP1252 mojibake (e.g. "â€¢" -> "•").

    Round-trips through the byte representation CP1252 would have produced
    and re-decodes as UTF-8; falls back to the original string if that
    isn't valid (i.e. the text was never mis-decoded, or the corruption
    already destroyed the original bytes -- 2 of ~16k rows in the v2.1
    source, left as-is).
    """
    try:
        fixed = s.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return s if '�' in fixed else fixed


def clean_text(s) -> str:
    s = '' if s is None else str(s)
    s = fix_mojibake(s)
    return s.strip()


def clean_category(s: str) -> str:
    s = clean_text(s)
    return CATEGORY_FIXES.get(s.lower(), s)


def tokenize(s) -> list:
    s = APOSTROPHE_RE.sub("", str(s).lower())
    return TOKEN_RE.findall(s)


def parse_size(raw: str):
    """item_size / pack_count / unit -- see matching.py's parse_size() for
    the full reasoning; kept identical here so a rebuilt reference.json
    always agrees with what the live app would compute for the same text.
    """
    raw = str(raw)
    m = _SIZE_COMPOUND_A_RE.search(raw)
    if m:
        return m.group(2), m.group(1), m.group(3)
    m = _SIZE_COMPOUND_B_RE.search(raw)
    if m:
        return m.group(2), m.group(1), m.group(3)
    m = _SIZE_COMPOUND_C_RE.search(raw)
    if m:
        return m.group(1), m.group(3), m.group(2)
    item_size, unit, pack_count = '', '', ''
    m = _SIZE_STANDALONE_RE.search(raw)
    if m:
        item_size, unit = m.group(1), m.group(2)
    m = _PACK_STANDALONE_RE.search(raw)
    if m:
        pack_count = m.group(1)
    return item_size, pack_count, unit


def load_items(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='Items', dtype=str, keep_default_na=False)
    df = df.rename(columns={
        'raw item name': 'raw',
        'cleaned item name': 'item',
        'cleaned brand name': 'brand',
        'cleaned category': 'category',
        'Source': 'source',
    })
    for col in ['raw', 'item', 'brand', 'category', 'source']:
        df[col] = df[col].map(clean_text)
    df['category'] = df['category'].map(clean_category)
    return df


def is_misc_placeholder(item, brand, category) -> bool:
    """Rows labeled "Misc. Accessories & Gifts" use that string as the item
    AND brand: a bucket for non-alcoholic items, not a catalog item with a
    name to match. They feed the Misc classifier (misc_classifier.py) and
    the exact-lookup set, and stay out of the matching graph."""
    return item == brand == category == MISC_CATEGORY


def norm_key(s) -> str:
    """Case- and whitespace-insensitive identity of a raw name (matches the
    exact-lookup key MatchEngine.misc_prob() builds at predict time)."""
    return ' '.join(str(s).lower().split())


def drop_conflicting_misc(df: pd.DataFrame):
    """A Misc row whose raw text is ALSO labeled with an alcohol category
    elsewhere contradicts itself. The alcohol label wins (it comes from the
    original hand-checked set; many Misc rows were added later, in bulk), so
    drop the Misc side and report it."""
    key = df['raw'].map(norm_key)
    alcohol_keys = set(key[df['category'] != MISC_CATEGORY])
    clash = (df['category'] == MISC_CATEGORY) & key.isin(alcohol_keys)
    return df[~clash].reset_index(drop=True), df[clash]


def build_catalog(df: pd.DataFrame):
    """canonical item -> (brand, category), picking the most common pairing
    per item name when the source has more than one (a real data-quality
    property of this file, not something to silently average away)."""
    item_brand = {}
    item_cat = {}
    by_item_brand = defaultdict(Counter)
    by_item_cat = defaultdict(Counter)
    for row in df.itertuples():
        if is_misc_placeholder(row.item, row.brand, row.category):
            continue
        by_item_brand[row.item][row.brand] += 1
        by_item_cat[row.item][row.category] += 1
    items = sorted(by_item_brand.keys())
    for it in items:
        brands = by_item_brand[it]
        brands.pop('', None)
        item_brand[it] = brands.most_common(1)[0][0] if brands else ''
        cats = by_item_cat[it]
        cats.pop('', None)
        item_cat[it] = cats.most_common(1)[0][0] if cats else ''
    return items, item_brand, item_cat


def build_graph(items, item_brand, item_cat, use_embeddings=True):
    item_tokens = {it: [w for w in tokenize(it) if w not in STOP and len(w) > 1] for it in items}
    N = len(items)
    df_count = Counter()
    for it, toks in item_tokens.items():
        for w in set(toks):
            df_count[w] += 1

    # Smoothed idf, standard tf-idf form: log((1+N)/(1+df)) + 1. Guarantees
    # idf > 0 even for a word in every item, and is well-behaved for the
    # df=1 case (a word unique to one item, e.g. a rare brand-coined name),
    # which is common and deliberately meant to score as highly
    # distinguishing (see design doc SS3.1).
    idf = {w: log((1 + N) / (1 + df)) + 1 for w, df in df_count.items()}

    item_vec = {}
    item_norm = {}
    inv_index = defaultdict(list)
    for it, toks in item_tokens.items():
        vec = {}
        for w in set(toks):
            vec[w] = idf[w]
        item_vec[it] = vec
        item_norm[it] = sqrt(sum(v * v for v in vec.values())) or 1.0
        for w, v in vec.items():
            inv_index[w].append((it, v))

    canon_vocab = sorted(idf.keys())

    emb_words, emb_matrix = [], None
    if use_embeddings:
        try:
            import spacy
            nlp = spacy.load('en_core_web_md', disable=['tagger', 'parser', 'ner', 'lemmatizer'])
            vecs = []
            for w in canon_vocab:
                lex = nlp.vocab[w]
                if lex.has_vector and lex.vector_norm > 0:
                    emb_words.append(w)
                    vecs.append(lex.vector / lex.vector_norm)
            emb_matrix = np.array(vecs, dtype=np.float32) if vecs else None
            print(f"  embeddings: {len(emb_words)} of {len(canon_vocab)} canonical words have vectors", file=sys.stderr)
        except Exception as e:
            print(f"  embeddings skipped: {e}", file=sys.stderr)

    return {
        'items': items,
        'item_brand': item_brand,
        'item_cat': item_cat,
        'item_vec': item_vec,
        'item_norm': item_norm,
        'inv_index': dict(inv_index),
        'idf': idf,
        'canon_vocab': canon_vocab,
        'emb_words': emb_words,
        'emb_matrix': emb_matrix,
        'STOP': STOP,
    }


def build_reference_rows(df: pd.DataFrame) -> list:
    rows = []
    for i, row in enumerate(df.itertuples()):
        item_size, pack_count, unit = parse_size(row.raw)
        rows.append({
            'id': f'ref_{i}',
            'raw': row.raw,
            'item': row.item,
            'brand': row.brand,
            'category': row.category,
            'item_size': item_size,
            'pack_count': pack_count,
            'unit': unit,
            'source': row.source,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path, help='Path to the source .xlsx (Items + Brands+Category sheets)')
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data')
    ap.add_argument('--no-embeddings', action='store_true', help='Skip building the spaCy embedding table')
    ap.add_argument('--extra-negatives', type=Path, default=None,
                    help='Text file, one raw name per line, of known-alcohol names to add as extra negatives for the '
                         'Misc model only (e.g. ../google_search_non_alcoholic/pseudo_negatives.txt). The hard truth has '
                         'almost no beer/wine, so without these the Misc model mislabels them.')
    args = ap.parse_args()

    print(f"Loading {args.source} ...", file=sys.stderr)
    df = load_items(args.source)
    print(f"  {len(df)} rows, {df['raw'].nunique()} unique raw names, {df['item'].nunique()} unique items", file=sys.stderr)

    df, clashes = drop_conflicting_misc(df)
    if len(clashes):
        print(f"  dropped {len(clashes)} Misc rows whose raw text is also labeled as alcohol "
              f"(alcohol label wins): {sorted(set(clashes['raw']))}", file=sys.stderr)

    items, item_brand, item_cat = build_catalog(df)
    print(f"  catalog: {len(items)} items, {len(set(item_brand.values()) - {''})} brands, "
          f"{len(set(item_cat.values()) - {''})} categories", file=sys.stderr)

    graph = build_graph(items, item_brand, item_cat, use_embeddings=not args.no_embeddings)

    # Misc detector: positives are every Misc-category row (placeholder and
    # real-named, e.g. "Tres Agaves Margarita Mix"), negatives every other row.
    is_misc = (df['category'] == MISC_CATEGORY)
    train_raws = df['raw'].tolist()
    train_labels = is_misc.astype(int).tolist()
    if args.extra_negatives:
        known = {norm_key(r) for r in train_raws}
        extra = [ln.strip() for ln in args.extra_negatives.read_text(encoding='utf-8').splitlines()]
        extra = [r for r in extra if r and norm_key(r) not in known]
        train_raws += extra
        train_labels += [0] * len(extra)
        print(f"  + {len(extra)} extra negative names for the Misc model from {args.extra_negatives.name}", file=sys.stderr)
    graph['misc_model'] = train_misc_model(train_raws, train_labels)
    placeholder = df.apply(lambda r: is_misc_placeholder(r['item'], r['brand'], r['category']), axis=1)
    real_named_keys = {norm_key(r) for r in df.loc[is_misc & ~placeholder, 'raw']}
    graph['misc_exact'] = frozenset(
        k for k in (norm_key(r) for r in df.loc[placeholder, 'raw']) if k not in real_named_keys
    )
    print(f"  misc model trained on {len(train_raws)} names ({int(is_misc.sum())} Misc), "
          f"{len(graph['misc_exact'])} exact-lookup raw names", file=sys.stderr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.out_dir / 'graph.pkl'
    with open(graph_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"  wrote {graph_path}", file=sys.stderr)

    ref_rows = build_reference_rows(df)  # df already excludes the dropped clashes
    ref_path = args.out_dir / 'reference.json'
    ref_path.write_text(json.dumps({'rows': ref_rows}), encoding='utf-8')
    print(f"  wrote {ref_path} ({len(ref_rows)} rows)", file=sys.stderr)


if __name__ == '__main__':
    main()
