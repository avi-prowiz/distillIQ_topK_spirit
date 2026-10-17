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

- **Fewer than 5 options is normal.** Options with a calibrated `confidence` below 0.5% are not shown (on the
  test below that cost 0.15 points of top-5 and cut the average list from 5.0 to 3.7 options), and a name with a
  brand neither list knows returns none (see [Brand rules](#brand-rules)). A blank or `None` raw name returns `[]`.
- `suggest_rows` returns one row per input row, in the same order. Missing options are blank strings.
- The Misc option is first when the model is at least 50% sure the item is not liquor, and third when it is
  15-50% sure (so a possible liquor item is not hidden). It rarely appears for a clearly alcoholic name (2.3% of liquor rows in the test below).
- The same raw name always gives the same answer.

### Using it with the tagger

`name` is already in the `brand item_name` form the hard truth uses, so the options can be passed straight to
the tagger / a reviewer. A reasonable rule: take option 1 when its `confidence` is high (0.95 or more is right
98% of the time), otherwise show the 5 options for a person to pick from.

## Brand rules

The brand is the strongest signal, so it is a **hard rule**, not just a weight:

- **A brand named in the raw name locks the answer to that brand.** The brand can come from the hard truth or
  from the brand-wise lists (spelling variants such as `Jack Daniel's` / `Jack Daniels` are one brand; a single
  distinctive brand word such as `1800` or `Cuervo` counts too). Only that brand's items are candidates, so
  `Jack Daniels Double Barrel` can only bring up Jack Daniels items (it will not show a `1800 ... Double Barrel`),
  and `1800 Tequila Double Barrel` only 1800 items. Two named brands (a co-branded pack) allow both.
  Checked on 4,000 raw names of the Oct 2026 file: 2,972 name a known brand, and **0 options came from another brand**.
- **A shared generic phrase is fine within a brand, but never across brands.** `Double Barrel`, `Single Malt` and the
  like only match items of the named brand; with no brand named, such a name shows its closest items from any brand,
  each with a low confidence.
- **A brand neither list knows gets no liquor options.** If one of the first two words is a word of 4+ letters that
  matches nothing in the vocabulary ("Glenfiddich Single Malt 12 Year", "Louis Jadot Beaujolais Villages",
  "Coors Light"), the options would only be other brands' look-alikes, so the list is empty (the Misc option can still
  appear). Misspellings of known words ("Absolute", "Smirnof") and short abbreviations ("WL", "Smir") do not count as
  unknown. This rule hides 0.3% of the hard truth's liquor names (mostly misspelled or unlisted brands).
- Accented letters are folded first (`Patron`, `Nutrl`), so they no longer split a word.
- Because the hard truth's own mappings are used as evidence (see the next section), a hard-truth name is
  answered the way the hard truth answers it.

## How it works

1. **Candidate pool.** The hard-truth items (about 3,500) plus the products of the brand-wise lists
   (`brand wise items.zip`, 95 brands) that the hard truth does not already have (about 580 after removing
   merchandise rows and names that are the same product as a hard-truth item under another wording). Brand-list
   names are written `brand product` and folded to plain ASCII (BACARDI, Anejo).
2. **Candidate retrieval** from three sources, all restricted to the named brand when the raw name has one
   (see [Brand rules](#brand-rules)):
   - the TF-IDF word-weight graph (`matching.py`): every word of the raw name is matched to the catalog
     vocabulary (exact, alias, embedding, fuzzy) and items are scored by cosine; both the unrestricted ranking
     and the ranking after the brand-lock rules are used;
   - a **memory of the hard truth's labeled raw names**: the closest labeled raw name of every item. This is the
     "name mappings in the hard truth get priority" evidence; it carries the largest weight of the ranker;
   - the brand-list products, which have no labeled raw names and are found through the word graph.
3. **Ranking model.** A conditional-logit model (a softmax over each name's candidates, 18 numeric features,
   numpy only) learned from the hard truth: cosine, rank, brand-lock agreement, memory similarity, share of the
   item's words found in the raw name and the reverse, word-order similarity, pack/gift/with markers, how many
   labeled raw names the item has, and so on. It is trained on **out-of-fold** features, so a raw name is never
   ranked using its own label.
4. **Confidence.** A small logistic calibration turns the ranker's output into the probability that an option
   is the right item (it also looks at how many of the raw name's words the model has never seen).
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
string and is within 0.5 points. These numbers are before the 0.5% confidence floor, which removes options that are almost never right. 32,774 labeled rows.

| Rows | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Liquor items (17,117) | 79.3% | 87.2% | **88.5%** |
| Misc / non-liquor names (15,657): `<Misc>` option shown | 96.4% (first) | 98.2% | **98.2%** |
| All rows | 87.5% | 92.5% | **93.1%** |
| *Baseline: TF-IDF cosine only* (liquor rows) | 74.1% | 85.2% | 86.8% |

- **The ceiling for liquor rows is 90.4%**: that share of held-out rows has their item somewhere in the catalog
  built without them, so top-5 reaches 98% of what is reachable (the brand rules cost 0.1 point against an unrestricted ranking). The rest are items no list contains.
- The Misc option is also shown on 2.3% of liquor rows (first on 1.7%), which costs liquor top-1 about 1 point.
- Brand-list products (not in the hard truth) queried by their own name are found at top-1 95%, top-5 100%.

By category (top-1 / top-5):

| Category | Rows | Top-1 | Top-5 | Item in catalog |
|---|---|---|---|---|
| Whiskey | 5,371 | 78.0% | 87.1% | 89.7% |
| Vodka | 2,956 | 83.5% | 91.3% | 92.9% |
| Tequila | 2,902 | 84.8% | 93.1% | 94.4% |
| Ready-To-Drink | 2,631 | 78.3% | 85.5% | 87.6% |
| Rum | 1,103 | 86.3% | 91.9% | 93.6% |
| Prepared Cocktails | 896 | 76.0% | 87.7% | 89.2% |
| Hard Seltzer | 430 | 78.1% | 87.4% | 89.8% |
| Mezcal | 247 | 87.4% | 90.3% | 91.1% |
| Liqueur (78 rows) | 78 | 76.9% | 82.1% | 84.6% |
| Gin (61 rows) | 61 | 86.9% | 86.9% | 86.9% |
| Wine (36 rows) | 36 | 88.9% | 88.9% | 88.9% |
| Beer (62 rows) | 62 | 41.9% | 48.4% | 48.4% |
| Cider (2), Cognac (2) | 4 | too few | | |
| Misc. Accessories & Gifts | 15,997 | 94.6% | 97.6% | 100% |

(The Misc rows include 340 real-named items such as "Tres Agaves Margarita Mix" that are ordinary catalog items.)

**The `confidence` is reliable.** Out-of-fold, for the top option:

| Confidence | Options | Actually correct |
|---|---|---|
| 0.95 - 1.00 | 6,679 | 98.3% |
| 0.90 - 0.95 | 3,262 | 92.2% |
| 0.80 - 0.90 | 2,217 | 80.2% |
| 0.60 - 0.80 | 1,826 | 63.7% |
| 0.40 - 0.60 | 1,172 | 48.0% |
| 0.20 - 0.40 | 1,111 | 36.4% |
| below 0.20 | 15,147 | 1.9% |

(The below-0.20 group is mostly Misc rows, where no liquor option is right.) The full tables are in
`topk_evaluation.xlsx`. Reproduce them with `evaluate_topk.py`.

On 4,000 sampled names of the 72,000 unique raw names of the Oct 2026 raw-items file: 2,972 name a known brand (0 options
from another brand), 22.8% got no liquor option (non-liquor items, wines, beers and unlisted brands), and the others
average 3.2 options. Speed: about 140 names per second after a 3 second load.

## Limitations

- **Beer, wine and cider are barely in the data** (hard truth: 62 beers, 36 wines, 2 ciders; the brand lists are
  spirits). Their raw names get low-confidence liquor options or the Misc option; the model cannot suggest an
  item it has never seen. Add beer/wine rows to the hard truth and rebuild to fix this.
- **The hard truth names the same product in several ways** ("Svedka 100 Proof Vodka" vs "Svedka Vodka", with and
  without "Pack"). The model returns both wordings as separate options; top-5 is more dependable than top-1.
- **A brand-prefixed non-liquor name** ("Del Maguey Hy-Vee Beef Roast") is recognised as Misc; the brand's
  items are still listed after it, with very low confidence, and are hidden once below the 0.5% floor.
- **Unlisted brands get nothing.** A raw name starting with a brand that is in neither list (a Scotch, a beer, a wine)
  returns no liquor option rather than another brand's look-alike. Add the brand to the hard truth or the brand lists
  and rebuild to cover it.
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
| `min_conf` | same | 0.005 | Options with a calibrated confidence below this are not shown. |
| `tau_first`, `tau_inc`, `slot` | `DEFAULT_MIX` in `topk.py` (fitted values are stored in `data/topk_meta.pkl`) | 0.5 / 0.15 / 2 | P(Misc) at which the Misc option goes first / is included, and the position it is inserted at. |
| `hint_mult`, `cov_mult` | same | 0.4 / 0.6 | How much a beer/wine hint or unfamiliar words lower P(Misc). |
