"""
Load the brand-wise product lists ("brand wise items.zip": a brands/ folder with one Excel file per
brand, plus a summary/ folder that is only documentation) into a clean table of product names.

Every product becomes one candidate item written as "<brand> <product name>" (the product name is
used as is when it already starts with the brand). Names are folded to plain ASCII (BACARDI, Anejo)
so the word matcher sees the same words as the hard truth. Merchandise, apparel and accessories rows
are dropped.
"""
import io
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

import rawsig

DROP_CATEGORY_RE = re.compile(r"merch|apparel|accessor|^other$", re.I)
# Words that mark a row as something other than a drink product when no Category column says so.
NON_PRODUCT_RE = re.compile(
    r"\b(t-?shirt|hoodie|hat|cap|beanie|jacket|apron|tote|poster|sticker|candle|keychain|coaster|"
    r"glassware|glass set|rocks glass|shot glass|koozie|tumbler|gift card|e-?gift|dice|playing cards|"
    r"cigar|lighter|bandana|socks|sweatshirt|mug)\b", re.I)


def ascii_fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def _brand_key(b):
    return re.sub(r"[^a-z0-9]", "", ascii_fold(b).lower())


def _read_files(path):
    """-> list of (file name, bytes) for every brands/*.xlsx in a folder or a .zip"""
    path = Path(path)
    out = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if re.search(r"(^|/)brands/[^/]+\.xlsx$", n):
                    out.append((Path(n).name, z.read(n)))
    else:
        root = path / "brands" if (path / "brands").is_dir() else path
        for f in sorted(root.glob("*.xlsx")):
            out.append((f.name, f.read_bytes()))
    return out


def load_brand_products(path):
    """DataFrame: brand, product, category_hint, file. One row per listed product (merch dropped)."""
    rows = []
    for name, data in _read_files(path):
        x = pd.ExcelFile(io.BytesIO(data))
        if "Products" not in x.sheet_names:
            continue
        d = x.parse("Products", dtype=str)
        if "Product Name" not in d.columns or "Brand" not in d.columns:
            continue
        hint = ""
        if "Source & Method" in x.sheet_names:
            sm = x.parse("Source & Method", dtype=str)
            for r in sm.itertuples(index=False):
                if str(r[0]).lower().startswith("category"):
                    hint = str(r[1])
                    break
        for r in d.to_dict("records"):
            prod = r.get("Product Name")
            if not isinstance(prod, str) or not prod.strip():
                continue
            cat = r.get("Category") if isinstance(r.get("Category"), str) else ""
            if cat and DROP_CATEGORY_RE.search(cat):
                continue
            if NON_PRODUCT_RE.search(prod):
                continue
            rows.append(dict(brand=r["Brand"], product=prod, category_hint=cat or hint, file=name))
    return pd.DataFrame(rows)


def to_items(products, known_brands, brand_category):
    """Turn the product table into candidate items.

    known_brands: {brand key: brand spelling} from the hard truth, so "BACARDI" files under the hard
    truth's "Bacardi". brand_category: {brand spelling: most common hard-truth category}.
    Returns DataFrame: item, brand, category, product, file (one row per distinct item name).
    """
    out, seen = [], set()
    for r in products.itertuples(index=False):
        brand_raw = re.sub(r"\s*\(.*?\)", "", ascii_fold(r.brand)).strip()   # "Kraken (The Kraken ...)" -> "Kraken"
        brand = known_brands.get(_brand_key(brand_raw), brand_raw)
        prod = ascii_fold(r.product)
        prod = re.sub(r"\s*[-,]\s*\d+(\.\d+)?\s*(ml|l|oz|fl\.? ?oz)\b.*$", "", prod, flags=re.I).strip()
        prod = re.sub(r"\s+\d+(\.\d+)?\s*(ml|l|oz)\s*$", "", prod, flags=re.I).strip()
        # a size or "Bottle" inside the name ("400 Conejos Cuishe 750 mL Bottle") is not part of the product name
        prod = re.sub(r"(?<![A-Za-z0-9])\d+(\.\d+)?\s*(ml|l|oz|fl\.? ?oz|liter|litre)\b\.?", " ", prod, flags=re.I)
        prod = re.sub(r"\s+(bottles?|btl)\s*$", "", prod, flags=re.I)
        prod = re.sub(r"\s+", " ", prod).strip(" -,")
        if not prod:
            continue
        item = _with_brand(brand, prod)
        key = " ".join(sorted(set(rawsig.tokens(item))))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(item=item, brand=brand, category=brand_category.get(brand) or _map_category(r.category_hint),
                        product=r.product, file=r.file))
    return pd.DataFrame(out)


def _with_brand(brand, prod):
    """'<brand> <product>' without repeating the brand: "Dobel Humito" under Maestro Dobel becomes
    "Maestro Dobel Humito"; a product that already carries every brand word is kept as is."""
    bt, pt = rawsig.tokens(brand), rawsig.tokens(prod)
    if not bt:
        return prod
    if pt[:len(bt)] == bt or set(bt) <= set(pt):
        return prod
    pw = prod.split()
    for k in range(min(len(bt), len(pt)), 0, -1):          # product starts with the tail of the brand
        if pt[:k] == bt[-k:]:
            return f"{brand} {' '.join(pw[k:])}".strip() if len(pw) >= k else prod
    return f"{brand} {prod}"


_CAT_WORDS = [("mezcal", "Mezcal"), ("tequila", "Tequila"), ("whisk", "Whiskey"), ("bourbon", "Whiskey"), ("rum", "Rum"),
              ("vodka", "Vodka"), ("gin", "Gin"), ("liqueur", "Liqueur"), ("seltzer", "Hard Seltzer"),
              ("cocktail", "Prepared Cocktails"), ("ready", "Ready-To-Drink"), ("rtd", "Ready-To-Drink"),
              ("cognac", "Cognac & Brandy"), ("brandy", "Cognac & Brandy"), ("wine", "Wine"), ("beer", "Beer")]


def _map_category(text):
    t = str(text).lower()
    for w, c in _CAT_WORDS:
        if w in t:
            return c
    return ""
