"""Smoke + sanity tests for the top-K model. Run from this folder: python test_topk.py"""
import random

from topk import MISC_OPTION, get_model, suggest, suggest_rows

K = 5


def names(raw, k=K):
    return [o["name"] for o in suggest(raw, k)]


# 1. a normal liquor name: five distinct options, the right item among the first two, ranks 1..5
opts = suggest("Jack Daniels Honey Whiskey 750 ml")
assert [o["rank"] for o in opts] == [1, 2, 3, 4, 5]
assert len({o["name"] for o in opts}) == 5
assert any("Honey" in o["name"] for o in opts[:2]), opts
assert all(0.0 <= o["confidence"] <= 1.0 and 0.0 <= o["score"] <= 1.0 for o in opts)
print("1 ok:", [o["name"] for o in opts[:2]])

# 2. non-liquor names get the Misc option, first when the match is clear
assert names("Coca Cola Classic 2 L")[0] == MISC_OPTION
assert names("Kozy Shack Original Recipe Tapioca Pudding Tub - 22 Oz")[0] == MISC_OPTION
assert MISC_OPTION in names("Viski Tequila Copita Glass Amber")
print("2 ok")

# 3. a clearly alcoholic name does not get the Misc option
for raw in ["Patron Silver Tequila 1.75 L", "Titos Handmade Vodka 1L", "Smirnoff Ice Screwdriver", "Bacardi Superior Rum 375ml"]:
    assert MISC_OPTION not in names(raw), raw
print("3 ok")

# 4. brand-list-only products (not in the hard truth) can be suggested
m = get_model()
b_items = sorted(it for it, s in m.item_source.items() if s == "B")
assert len(b_items) > 400, len(b_items)
random.seed(0)
sample = random.sample(b_items, 60)
hit5 = sum(it in names(it) for it in sample) / len(sample)
hit1 = sum(names(it)[0] == it for it in sample) / len(sample)
print(f"4 brand-list items queried by their own name: top-1 {hit1:.0%}, top-5 {hit5:.0%}")
assert hit5 >= 0.9 and hit1 >= 0.7

# 5. suggest_rows: one row per input, same order, blank names stay blank, always id + raw + 5 options
rows = [[10, "Jack Daniels Honey Whiskey 750 ml"], [11, ""], [12, None], [13, "Zzxq Wrbl Qplm 750 ml"], [14, "Coca Cola Classic 2 L"]]
out = suggest_rows(rows)
assert [r[0] for r in out] == [10, 11, 12, 13, 14] and all(len(r) == 2 + K for r in out)
assert out[1][2:] == [""] * K and out[2][2:] == [""] * K
assert out[0][2] != "" and out[4][2] == MISC_OPTION
print("5 ok")

# 6. the same raw name always gives the same answer
assert names("Crown Royal Regal Apple Flavored Whisky - 375 Ml") == names("Crown Royal Regal Apple Flavored Whisky - 375 Ml")
print("ok")

# 7. a brand named in the raw name is a hard rule: every option is that brand's item, never another brand's
from topk import fold_accents
eng = get_model().eng
for raw, brand in [("Jack Daniels Double Barrel", "jackdaniels"), ("1800 Tequila Double Barrel", "1800tequila"), ("Knob Creek Double Barrel", "knobcreek"),
                   ("Bulleit Single Malt", "bulleit"), ("Patron Silver Tequila 1.75 L", "patron")]:
    ops = [o for o in suggest(raw) if o["name"] != MISC_OPTION]
    assert ops, raw
    assert all(eng._bk(o["brand"]).replace(" ", "").startswith(brand[:5]) for o in ops), (raw, [(o["name"], o["brand"]) for o in ops])
assert not any("1800" in o["name"] for o in suggest("Jack Daniels Double Barrel"))
assert not any("Jack Dan" in o["name"] for o in suggest("1800 Tequila Double Barrel"))
print("7 ok")

# 8. a brand neither list knows gets no other-brand lookalikes ("Single Malt" alone must not pull in Bushmills / Bulleit)
for raw in ["Glenfiddich Single Malt 12 Year", "Macallan Double Cask Single Malt", "Louis Jadot Beaujolais Villages"]:
    assert suggest(raw) == [], raw
assert fold_accents("Patron Citronge Nutrl") == "Patron Citronge Nutrl" and suggest("Nütrl Vodka Seltzer Pineapple 4pk")[0]["name"].startswith("Nutrl")
print("8 ok")
