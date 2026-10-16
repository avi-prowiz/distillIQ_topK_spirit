# distillIQ top-K spirit suggestions

A lightweight model that takes a raw item name and returns the **5 most likely item names**, each written
as `brand item_name` (for example `Jack Daniels Tennessee Honey Whiskey`), best first. When the raw name is
close to the non-liquor / accessory items of the hard truth, the list also contains a
**`<Misc - not a Liquor>`** option. It is meant to feed the `distillIQ_spirit_tagger_api` model, and it is a
separate, self-contained folder (pure Python + numpy, no server, no network).

- [Quick start](#quick-start)
- [Output](#output)
- [How it works](#how-it-works)
- [Accuracy](#accuracy)
- [Limitations](#limitations)
- [Rebuilding the model](#rebuilding-the-model)
- [Folder contents](#folder-contents)
- [Settings](#settings)

## Quick start

```bash
cd distillIQ_topK_spirit
pip install -r requirements.txt
python -m spacy download en_core_web_md      # optional, helps with typos
python test_topk.py                          # should end with "ok"
```

```python
from topk import suggest, suggest_rows

suggest("Jack Daniels Honey Whiskey 750 ml")          # list of 5 option dicts
suggest("Coca Cola Classic 2 L")[0]["name"]            # '<Misc - not a Liquor>'

# batch: [[id, raw_item_name], ...] -> [[id, raw_item_name, option_1 ... option_5]]
suggest_rows([[1, "Jack Daniels Honey Whiskey 750 ml"], [2, "Patron Silver Tequila 1.75 L"]])
```

Command line (CSV or JSON in, CSV or JSON out; rows are `id, raw_item_name`, header row optional):

```bash
python topk.py example.csv options.csv        # id, raw_item_name, option_1 ... option_5
python topk.py example.json                   # printed as JSON
python topk.py example.csv options.csv 4      # a different K (here 4)
```

`example.csv` / `example.json` are small inputs to try it with.

## Output

`suggest(raw, k=5)` returns up to `k` dicts, best first:

| Key | Meaning |
|---|---|
| `rank` | 1 = best |
| `name` | The option, `brand item_name`, or `<Misc - not a Liquor>`. This is the string to pass on. |
| `brand` | The option's brand (`Misc. Accessories & Gifts` for the Misc option). |
| `category` | Category of the option (Whiskey, Vodka, Tequila, Ready-To-Drink ... or `Misc. Accessories & Gifts`). |
| `score` | Share of the probability among the candidates (sums to about 1 over all candidates). Use it to compare options of one name. |
| `confidence` | Calibrated probability that **this option is the right item** (0-1). Use it to decide whether to trust an option at all; see [Accuracy](#accuracy). |
| `source` | `hard_truth` (an item of the hard-truth list), `brand_list` (a product from the brand-wise lists that the hard truth does not have) or `misc`. |

- A blank or `None` raw name returns `[]`. A name that matches nothing at all can also return fewer than `k`
  options or none.
- `suggest_rows` returns one row per input row, in the same order. Missing options are blank strings.
- The Misc option is first when the model is at least 50% sure the item is not liquor, and third when it is
  15-50% sure (so a possible liquor item is not hidden). It rarely appears for a clearly alcoholic name (2.3% of liquor rows in the test below).
- The same raw name always gives the same answer.

### Using it with the tagger

`name` is already in the `brand item_name` form the hard truth uses, so the options can be passed straight to
the tagger / a reviewer. A reasonable rule: take option 1 when its `confidence` is high (0.95 or more is right
98% of the time), otherwise show the 5 options for a person to pick from.

## How it works

1. **Candidate pool.** The hard-truth items (about 3,500) plus the products of the brand-wise lists
   (`brand wise items.zip`, 95 brands) that the hard truth does not already have (about 580 after removing
   merchandise rows and names that are the same product as a hard-truth item under another wording). Brand-list
   names are written `brand product` and folded to plain ASCII (BACARDI, Anejo).
2. **Candidate retrieval** from three sources:
   - the TF-IDF word-weight graph (`matching.py`): every word of the raw name is matched to the catalog
     vocabulary (exact, alias, embedding, fuzzy) and items are scored by cosine; both the unrestricted ranking
     and the ranking after the brand-lock rules are used;
   - a **memory of the hard truth's labeled raw names**: the closest labeled raw name of every item. This is the
     "name mappings in the hard truth get priority" evidence; it carries the largest weight of the ranker;
   - the brand-list products, which have no labeled raw names and are found through the word graph.
3. **Ranking model.** A conditional-logit model (a softmax over each name's candidates, 17 numeric features,
   numpy only) learned from the hard truth: cosine, rank, brand-lock agreement, memory similarity, share of the
   item's words found in the raw name and the reverse, word-order similarity, pack/gift/with markers, how many
   labeled raw names the item has, and so on. It is trained on **out-of-fold** features, so a raw name is never
   ranked using its own label.
4. **Confidence.** A small logistic calibration turns the ranker's output into the probability that an option
   is the right item.
5. **Misc option.** `misc_classifier.py` (logistic regression over word weights, trained on the ~16,000 labeled
   Misc rows and ~56,000 extra known-alcohol names) gives P(Misc). A beer/wine hint or unfamiliar words
   lower it, so that a beer is not called a non-liquor item. The thresholds above (0.5 and 0.15) were chosen on
   held-out data.

Two data-quality steps happen when the model is built: the 31 hard-truth rows whose raw name shares no word
with their own item name ("Red Bull, Original" labeled `Long Drink Peach`, a rice bowl labeled `Taaka Vodka`)
are left out (`data/suspect_hardtruth_rows.csv`), and brand-list names that are only another wording of a
hard-truth item are not added as separate options.

## Accuracy

Measured out-of-fold: the model is rebuilt on 4/5 of the labeled raw names (grouped by raw name, so no raw string
is on both sides) and scored on the other 1/5, five times. A **hit** means one of the options has the same words
as the hard-truth item (size, case, punctuation and word order ignored); the strict version needs the identical
string and is within 0.5 points. 32,774 labeled rows.

| Rows | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Liquor items (17,117) | 79.4% | 87.4% | **88.7%** |
| Misc / non-liquor names (15,657): `<Misc>` option shown | 95.8% (first) | 98.2% | **98.2%** |
| All rows | 87.2% | 92.6% | **93.2%** |
| *Baseline: TF-IDF cosine only* (liquor rows) | 74.1% | 85.2% | 86.8% |

- **The ceiling for liquor rows is 90.4%**: that share of held-out rows has their item somewhere in the catalog
  built without them, so top-5 reaches 98% of what is reachable. The rest are items no list contains.
- The Misc option is also shown on 2.3% of liquor rows (first on 1.7%), which costs liquor top-1 about 1 point.
- Brand-list products (not in the hard truth) queried by their own name are found at top-1 95%, top-5 100%.

By category (top-1 / top-5):

| Category | Rows | Top-1 | Top-5 | Item in catalog |
|---|---|---|---|---|
| Whiskey | 5,371 | 78.1% | 87.1% | 89.7% |
| Vodka | 2,956 | 83.6% | 91.4% | 92.9% |
| Tequila | 2,902 | 85.0% | 93.1% | 94.4% |
| Ready-To-Drink | 2,631 | 78.4% | 85.8% | 87.6% |
| Rum | 1,103 | 86.4% | 92.0% | 93.6% |
| Prepared Cocktails | 896 | 76.0% | 88.3% | 89.2% |
| Hard Seltzer | 430 | 77.9% | 87.2% | 89.8% |
| Mezcal | 247 | 88.3% | 91.1% | 91.1% |
| Liqueur (78 rows) | 78 | 79.5% | 84.6% | 84.6% |
| Gin (61 rows) | 61 | 86.9% | 86.9% | 86.9% |
| Wine (36 rows) | 36 | 88.9% | 88.9% | 88.9% |
| Beer (62 rows) | 62 | 41.9% | 48.4% | 48.4% |
| Cider (2), Cognac (2) | 4 | too few | | |
| Misc. Accessories & Gifts | 15,997 | 94.0% | 97.7% | 100% |

(The Misc rows include 340 real-named items such as "Tres Agaves Margarita Mix" that are ordinary catalog items.)

**The `confidence` is reliable.** Out-of-fold, for the top option:

| Confidence | Options | Actually correct |
|---|---|---|
| 0.95 - 1.00 | 6,825 | 98.2% |
| 0.90 - 0.95 | 3,109 | 92.1% |
| 0.80 - 0.90 | 2,175 | 80.1% |
| 0.60 - 0.80 | 1,863 | 64.5% |
| 0.40 - 0.60 | 1,209 | 49.9% |
| 0.20 - 0.40 | 1,119 | 33.5% |
| below 0.20 | 16,462 | 1.9% |

(The below-0.20 group is mostly Misc rows, where no liquor option is right.) The full tables are in
`topk_evaluation.xlsx`. Reproduce them with `evaluate_topk.py`.

On the 72,000 unique raw names of the Oct 2026 raw-items file (3,000 sampled): 22% were recognised as non-liquor,
and about 60% got only low-confidence options (wines, beers and other products the hard truth does not cover,
sometimes with a spirit brand in front), as intended. Speed: about 200 names per second after a 3 second load.

## Limitations

- **Beer, wine and cider are barely in the data** (hard truth: 62 beers, 36 wines, 2 ciders; the brand lists are
  spirits). Their raw names get low-confidence liquor options or the Misc option; the model cannot suggest an
  item it has never seen. Add beer/wine rows to the hard truth and rebuild to fix this.
- **The hard truth names the same product in several ways** ("Svedka 100 Proof Vodka" vs "Svedka Vodka", with and
  without "Pack"). The model returns both wordings as separate options; top-5 is more dependable than top-1.
- **A brand-prefixed non-liquor name** ("Del Maguey Hy-Vee Beef Roast") is recognised as Misc, but its
  alcohol options are still listed after it with very low confidence.
- The accuracy above is for names that look like the hard truth. Names from a very different source will score
  lower.

## Rebuilding the model

When the hard truth or the brand lists change:

```bash
python build_topk.py "path\to\super-set-map-v2.1.xlsx" "path\to\brand wise items.zip" \
    --extra-negatives pseudo_negatives.txt --folds 5 --cache some_folder
```

`--extra-negatives` is a text file of known-alcohol raw names (one per line; the project's
`google_search_non_alcoholic/pseudo_negatives.txt`) that keeps the Misc model from calling beers and wines
non-liquor. `--cache` keeps the per-fold models so a rerun is fast. The command rebuilds the five fold models
(about 8 minutes), prints the out-of-fold report, fits the ranker, calibration and Misc thresholds, and writes
`data/graph.pkl` and `data/topk_meta.pkl`. `python evaluate_topk.py <truth> <brands> --out report.xlsx` only
evaluates. The brand lists can be given as the zip or the unzipped folder (it needs the `brands/` folder).

## Folder contents

| File | Purpose |
|---|---|
| `topk.py` | The public API (`suggest`, `suggest_rows`, CLI), candidate features, ranking, calibration, Misc placement. |
| `build_topk.py` | Builds the catalog (hard truth + brand lists), the word graph, the Misc model, the raw-name memory. |
| `evaluate_topk.py` | Out-of-fold evaluation and fitting of ranker, calibration and Misc thresholds. |
| `brand_lists.py` | Reads `brand wise items.zip` into clean `brand product` names. |
| `matching.py`, `misc_classifier.py`, `rawsig.py`, `build_graph.py` | Copies of the tagger engine pieces it builds on. |
| `data/graph.pkl`, `data/topk_meta.pkl` | The trained model. |
| `data/suspect_hardtruth_rows.csv` | The 31 hard-truth rows left out as labeling errors (worth a manual look). |
| `test_topk.py` | Sanity tests. |
| `topk_evaluation.xlsx` | By-category table, confidence calibration, Misc placement grid. |
| `example.csv`, `example.json` | Sample inputs. |

## Settings

| Setting | Where | Default | Effect |
|---|---|---|---|
| `k` | `suggest(raw, k)` | 5 | Number of options. |
| `tau_first`, `tau_inc`, `slot` | `DEFAULT_MIX` in `topk.py` (fitted values are stored in `data/topk_meta.pkl`) | 0.5 / 0.15 / 2 | P(Misc) at which the Misc option goes first / is included, and the position it is inserted at. |
| `hint_mult`, `cov_mult` | same | 0.4 / 0.6 | How much a beer/wine hint or unfamiliar words lower P(Misc). |
