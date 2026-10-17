"""
Out-of-fold evaluation + fitting of the top-K model.

For each of K folds (grouped by raw name, so no raw string is on both sides) the catalog, word-weight
graph, Misc model and raw-name memory are rebuilt WITHOUT that fold, and the fold's raw names are
turned into candidate lists + features. The pooled features then
  * fit the ranking weights (each fold is scored with weights fitted on the other folds -> honest numbers), and
  * choose where the "<Misc - not a Liquor>" option goes.
A hit means one of the K options has the same words as the hard-truth item (item_key); the strict
variant requires the identical item string.

    python evaluate_topk.py <super-set-map.xlsx> <brand wise items.zip> [--extra-negatives f] [--folds 5]
                            [--cache dir] [--out report.xlsx]
"""
import argparse
import itertools
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_topk as bt
import rawsig
from misc_classifier import MISC_CATEGORY, alcohol_hint_blocks_misc
from topk import FEATURES, NF, TopK, item_key, mix_options, softmax_rank_train, DEFAULT_MIX, MISC_OPTION, cal_vector, fit_logistic

K = 5


def fold_split(truth, folds):
    keys = np.array(sorted(truth["key"].unique()))
    np.random.RandomState(1).shuffle(keys)
    return truth["key"].map({k: i % folds for i, k in enumerate(keys)})


def oof_records(truth, B, folds, extra_neg, cache):
    """-> DataFrame, one row per labeled raw name occurrence, with candidate pool + features (pickled per fold in cache)."""
    truth = truth.copy()
    truth["fold"] = fold_split(truth, folds)
    placeholder = truth.apply(lambda r: bt.bg.is_misc_placeholder(r["item"], r["brand"], r["category"]), axis=1)
    truth["misc_truth"] = placeholder
    recs = []
    for f in range(folds):
        pk = cache / f"oof_fold{f}_of{folds}.pkl"
        if pk.exists():
            recs.append(pd.read_pickle(pk))
            print(f"fold {f + 1}/{folds}: loaded cache", flush=True)
            continue
        tr, te = truth[truth.fold != f], truth[truth.fold == f]
        mdir = cache / f"model_fold{f}_of{folds}"
        if (mdir / 'graph.pkl').exists() and (mdir / 'topk_meta.pkl').exists():
            added, n = -1, -1       # model for this fold already built (cache): reuse it
        else:
            added, n = bt.build_model(tr, B, mdir, extra_neg)
        tk = TopK(mdir)
        pool = {}
        rows = []
        cat_keys = {item_key(i) for i in tk.eng.items}
        for r in te.itertuples():
            if r.raw not in pool:
                items, F = tk.candidates(r.raw)
                p = tk.eng.misc_prob(r.raw)
                pool[r.raw] = (items, F.astype(np.float32), p, bool(alcohol_hint_blocks_misc(r.raw)),
                               tk.eng.misc_model.coverage(r.raw) < 0.3 if p < 1.0 else False)
            items, F, p, hint, covlow = pool[r.raw]
            rows.append(dict(raw=r.raw, fold=f, truth_item=r.item, truth_brand=r.brand, truth_cat=r.category,
                             misc_truth=r.misc_truth, truth_key=item_key(r.item), items=items, F=F, p_raw=p, hint=hint,
                             covlow=covlow, in_catalog=item_key(r.item) in cat_keys))
        d = pd.DataFrame(rows)
        d.to_pickle(pk)
        recs.append(d)
        print(f"fold {f + 1}/{folds}: {len(te)} rows, catalog {n} items (+{added} from brand lists)", flush=True)
    return pd.concat(recs, ignore_index=True)


def tensors(R):
    """Padded [Q,C,F] features, mask and positive mask for the rows of R (alcohol-truth rows only matter for training)."""
    C = max((len(x) for x in R["items"]), default=1)
    X = np.zeros((len(R), C, NF), dtype=np.float32)
    mask = np.zeros((len(R), C), dtype=bool)
    pos = np.zeros((len(R), C), dtype=bool)
    for i, (items, F, tk, mt) in enumerate(zip(R["items"], R["F"], R["truth_key"], R["misc_truth"])):
        n = len(items)
        if n:
            X[i, :n] = F
            mask[i, :n] = True
            if not mt:
                pos[i, :n] = [item_key(it) == tk for it in items]
    return X, mask, pos


def fit_weights(R, iters=300):
    X, mask, pos = tensors(R)
    ok = mask.any(axis=1)
    flat = X[mask]
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Z = ((X - mu) / sd) * mask[:, :, None]
    w = softmax_rank_train(Z.astype(np.float64), mask, pos, iters=iters)
    return w, mu, sd


def rank_rows(R, w, mu, sd):
    out = []
    for items, F in zip(R["items"], R["F"]):
        if not len(items):
            out.append([])
            continue
        s = ((F - mu) / sd) @ w
        s = s - s.max()
        e = np.exp(s)
        p = e / e.sum()
        o = np.argsort(-p)
        out.append([(items[i], float(p[i])) for i in o])
    return out


def eff_pmisc(p, hint, covlow, cfg):
    if p >= 1.0:
        return 1.0
    if hint:
        p *= cfg["hint_mult"]
    if covlow:
        p *= cfg["cov_mult"]
    return p


def evaluate(R, ranked, cfg, k=K):
    """-> per-row DataFrame of hit flags for the final option lists under mixing config cfg."""
    res = []
    for row, rk in zip(R.itertuples(), ranked):
        pm = eff_pmisc(row.p_raw, row.hint, row.covlow, cfg)
        opts = mix_options(rk, pm, k, cfg)
        names = [o[0] for o in opts]
        if row.misc_truth:
            pos = names.index(MISC_OPTION) if MISC_OPTION in names else None
            strict = pos
        else:
            keys = [item_key(n) if n != MISC_OPTION else None for n in names]
            pos = keys.index(row.truth_key) if row.truth_key in keys else None
            strict = names.index(row.truth_item) if row.truth_item in names else None
        res.append(dict(pos=pos, strict=strict, has_misc=MISC_OPTION in names, misc_first=bool(names) and names[0] == MISC_OPTION))
    return pd.DataFrame(res)


def hit(series, n):
    return float(series.map(lambda v: v is not None and not pd.isna(v) and v < n).mean())


def summarise(R, E):
    out = {}
    for name, m in (("alcohol", ~R.misc_truth.values), ("misc", R.misc_truth.values), ("all", np.ones(len(R), bool))):
        e = E[m]
        out[name] = dict(rows=int(m.sum()), top1=hit(e.pos, 1), top3=hit(e.pos, 3), top5=hit(e.pos, 5), strict_top5=hit(e.strict, 5))
    return out


def tune_mix(R, ranked, cfg0):
    best, rows = None, []
    for tf, ti, slot in itertools.product([0.5, 0.7, 0.9], [0.02, 0.05, 0.1, 0.15, 0.2, 0.3], [2]):
        if ti >= tf:
            continue
        cfg = {**cfg0, "tau_first": tf, "tau_inc": ti, "slot": slot}
        E = evaluate(R, ranked, cfg)
        s = summarise(R, E)
        macro5 = (s["alcohol"]["top5"] + s["misc"]["top5"]) / 2
        macro1 = (s["alcohol"]["top1"] + s["misc"]["top1"]) / 2
        shown = float(E[~R.misc_truth.values].has_misc.mean())   # alcohol rows that get a Misc option they do not need
        score = macro5 + 0.3 * macro1 - 0.5 * shown
        rows.append(dict(tau_first=tf, tau_inc=ti, slot=slot, alc_top5=s["alcohol"]["top5"], misc_top5=s["misc"]["top5"],
                         alc_top1=s["alcohol"]["top1"], misc_top1=s["misc"]["top1"], misc_on_alcohol=shown, score=score))
        if best is None or score > best[0]:
            best = (score, cfg)
    return best[1], pd.DataFrame(rows).sort_values("score", ascending=False)


def cross_fit(R, folds, iters=300):
    """Rank every row with weights fitted on the other folds."""
    ranked = [None] * len(R)
    for f in range(folds):
        tr = R[R.fold != f]
        w, mu, sd = fit_weights(tr, iters)
        idx = np.where(R.fold.values == f)[0]
        sub = R.iloc[idx]
        for i, rk in zip(idx, rank_rows(sub, w, mu, sd)):
            ranked[i] = rk
        print(f"cross-fit fold {f + 1}/{folds} done", flush=True)
    return ranked


def baseline_ranked(R, feat="cos"):
    j = FEATURES.index(feat)
    out = []
    for items, F in zip(R["items"], R["F"]):
        if not len(items):
            out.append([])
            continue
        o = np.argsort(-F[:, j])
        out.append([(items[i], float(F[i, j])) for i in o])
    return out


def calib_dataset(R, ranked, rows=None):
    """One example per top-5 candidate option: calibration features, 1 if it is the right item."""
    X, y, q = [], [], []
    for i in (range(len(R)) if rows is None else rows):
        items = R["items"][i]
        if not len(items) or not ranked[i]:
            continue
        pos = {it: j for j, it in enumerate(items)}
        for r, (it, p) in enumerate(ranked[i][:K]):
            X.append(cal_vector(p, R["F"][i][pos[it]], r == 0))
            y.append(0.0 if R["misc_truth"][i] else float(item_key(it) == R["truth_key"][i]))
            q.append((i, r))
    return np.array(X), np.array(y), q


def fit_calibration(R, ranked, folds):
    """Fit the confidence calibration; returns (final params, out-of-fold calibrated confidence per (row, rank))."""
    X, y, q = calib_dataset(R, ranked)
    fold = np.array([R.fold[i] for i, _ in q])
    oof = np.zeros(len(y))
    for f in range(folds):
        w, b, mu, sd = fit_logistic(X[fold != f], y[fold != f])
        oof[fold == f] = 1 / (1 + np.exp(-(((X[fold == f] - mu) / sd) @ w + b)))
    w, b, mu, sd = fit_logistic(X, y)
    return dict(w=w, b=b, mu=mu, sd=sd), oof, y, q


def floor_table(R, ranked, oof, q, max_loss=0.0025):
    """Options whose calibrated confidence is below a floor are not shown (fewer options is fine). For each floor:
    liquor top-5 hit rate, average options shown, share of liquor names left with no option. Picks the largest
    floor that loses at most `max_loss` of top-5 hit rate."""
    by_row = {}
    for (i, r), c in zip(q, oof):
        by_row.setdefault(i, []).append((r, c))
    alc = [i for i in range(len(R)) if not R["misc_truth"][i]]
    rows = []
    for floor in [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        hits = shown = empty = 0
        for i in alc:
            opts = [(r, c) for r, c in by_row.get(i, []) if c >= floor]
            shown += len(opts)
            empty += not opts
            keys = [item_key(ranked[i][r][0]) for r, _ in opts]
            hits += R["truth_key"][i] in keys
        rows.append(dict(min_confidence=floor, liquor_top5=hits / len(alc), avg_options=shown / len(alc), no_options=empty / len(alc)))
    T = pd.DataFrame(rows)
    ok = T[T.liquor_top5 >= T.liquor_top5.iloc[0] - max_loss]
    return float(ok.min_confidence.max()), T


def calibration_table(oof, y, q):
    top1 = np.array([r == 0 for _, r in q])
    rows = []
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
        m = top1 & (oof >= lo) & (oof < hi)
        rows.append(dict(confidence=f"{lo:.2f}-{min(hi, 1):.2f}", top1_options=int(m.sum()), actually_correct=float(y[m].mean()) if m.any() else None,
                         mean_confidence=float(oof[m].mean()) if m.any() else None))
    return pd.DataFrame(rows)


def report(R, ranked, cfg, label=""):
    E = evaluate(R, ranked, cfg)
    s = summarise(R, E)
    A = R[~R.misc_truth]
    ceiling = float(A.in_catalog.mean())
    pool_hit = float(np.mean([any(item_key(i) == t for i in its) for its, t in zip(A["items"], A["truth_key"])]))
    print(f"\n== {label}")
    for k_, v in s.items():
        print(f"  {k_:8} rows={v['rows']:6}  top1={v['top1']:.1%}  top3={v['top3']:.1%}  top5={v['top5']:.1%}  strict-top5={v['strict_top5']:.1%}")
    print(f"  alcohol rows whose item exists in the training catalog: {ceiling:.1%};  in the candidate pool: {pool_hit:.1%}")
    ea = E[~R.misc_truth.values]
    print(f"  Misc option shown on alcohol rows: {ea.has_misc.mean():.1%} (first: {ea.misc_first.mean():.1%})")
    return E, s


def category_table(R, E):
    rows = []
    for cat, g in R.groupby("truth_cat"):
        e = E.loc[g.index]
        rows.append(dict(category=cat, rows=len(g), top1=hit(e.pos, 1), top3=hit(e.pos, 3), top5=hit(e.pos, 5),
                         strict_top5=hit(e.strict, 5), in_catalog=float(g.in_catalog.mean()) if cat != MISC_CATEGORY else 1.0))
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def confidence_table(R, ranked, E):
    """Top-5 hit rate by the model's top-1 probability (calibration)."""
    A = R[~R.misc_truth]
    p1 = np.array([ranked[i][0][1] if ranked[i] else 0.0 for i in A.index])
    h = E.loc[A.index].pos.map(lambda v: v is not None and not pd.isna(v) and v < 1).values
    h5 = E.loc[A.index].pos.map(lambda v: v is not None and not pd.isna(v) and v < 5).values
    rows = []
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        m = (p1 >= lo) & (p1 < hi)
        rows.append(dict(top1_prob=f"{lo:.1f}-{min(hi, 1):.1f}", rows=int(m.sum()), top1_hit=float(h[m].mean()) if m.any() else None,
                         top5_hit=float(h5[m].mean()) if m.any() else None))
    return pd.DataFrame(rows)


def run(args):
    truth = bt.load_truth(args.truth)
    B = bt.brand_items(truth, args.brands)
    cache = Path(args.cache) if args.cache else Path(tempfile.mkdtemp())
    cache.mkdir(parents=True, exist_ok=True)
    R = oof_records(truth, B, args.folds, args.extra_negatives, cache)
    ranked = cross_fit(R, args.folds)
    cfg, grid = tune_mix(R, ranked, DEFAULT_MIX)
    print("\nbest mixing config:", cfg)
    print(grid.head(8).to_string(index=False))
    return truth, B, R, ranked, cfg, grid


def fit_and_save(args):
    truth, B, R, ranked, cfg, grid = run(args)
    E, s = report(R, ranked, cfg, "top-5 model (out-of-fold)")
    report(R, baseline_ranked(R, "cos"), {**cfg, "tau_first": 2, "tau_inc": 2}, "baseline: TF-IDF cosine only, no Misc option")
    w, mu, sd = fit_weights(R)
    print("weights:", {k_: round(float(v), 3) for k_, v in zip(FEATURES, w)})
    cal, oof, y, q = fit_calibration(R, ranked, args.folds)
    print(chr(10) + "confidence calibration (out-of-fold), top-1 options:")
    print(calibration_table(oof, y, q).to_string(index=False))
    floor, ftab = floor_table(R, ranked, oof, q)
    print(chr(10) + "display floor (options below this confidence are not shown):")
    print(ftab.to_string(index=False))
    cfg = {**cfg, "min_conf": floor}
    print("chosen min_conf:", floor)
    out = Path(args.out_dir)
    bt.build_model(truth, B, out, args.extra_negatives)
    with open(out / "topk_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    meta.update(w=w, mu=mu, sd=sd, mix=cfg, cal=cal)
    sus = truth.attrs.get('suspect')
    if sus is not None and len(sus):
        sus[['raw', 'item', 'brand', 'category']].to_csv(out / 'suspect_hardtruth_rows.csv', index=False, encoding='utf-8-sig')
    with open(out / "topk_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote {out / 'graph.pkl'} and {out / 'topk_meta.pkl'}")
    return R, ranked, cfg, E, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", type=Path)
    ap.add_argument("brands", type=Path, nargs="?", default=Path(__file__).resolve().parent / "brand wise items.zip")
    ap.add_argument("--extra-negatives", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    a = ap.parse_args()
    truth, B, R, ranked, cfg, grid = run(a)
    E, s = report(R, ranked, cfg, "top-5 model (out-of-fold)")
    report(R, baseline_ranked(R, "cos"), {**cfg, "tau_first": 2, "tau_inc": 2}, "baseline: TF-IDF cosine only, no Misc option")
    print(category_table(R, E).to_string(index=False))
    print(confidence_table(R, ranked, E).to_string(index=False))
    if a.out:
        with pd.ExcelWriter(a.out) as w:
            category_table(R, E).to_excel(w, sheet_name="by category", index=False)
            confidence_table(R, ranked, E).to_excel(w, sheet_name="calibration", index=False)
            grid.to_excel(w, sheet_name="misc placement grid", index=False)
