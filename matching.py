"""
Graph-based item name matching engine.

This is the same algorithm developed and validated against the 8,175-row
labeled catalog (see the project's design doc), ported into a plain module
a backend can import. It loads a pre-built reference graph (graph.pkl,
produced offline from the canonical catalog) and, given a raw/messy item
name, returns ranked candidate matches for item name / brand / category.

Pretrained word embeddings (spaCy en_core_web_md) are used when available,
to catch raw words that are typos/synonyms of a canonical word but too far
apart for fuzzy string matching. The embedding model is a ~40MB download
that requires network access on first setup, so it's optional here: if it
isn't installed, this module logs a warning and simply skips that tier,
falling back to exact/alias/fuzzy matching only (this is also exactly what
the browser-only/client-side version of this tool does, since a browser
can't run spaCy at all).
"""
import re
import math
import pickle
import logging
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from misc_classifier import MISC_CATEGORY, MiscModel, alcohol_hint_blocks_misc
import rawsig

log = logging.getLogger("matching")

TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---- "Misc. Accessories & Gifts" (non-alcoholic items, beverages, value-added
# packs). Most hard-truth Misc rows use that string as item AND brand, so
# there is nothing in the catalog to match them to; MiscModel (trained by
# build_graph.py from the labeled rows) says how Misc-like a raw name is.
# predict() runs the normal catalog match first and only lets the Misc
# decision replace it when the catalog has no strong answer of its own:
#   - a raw name that is itself a labeled Misc row is Misc (exact lookup);
#   - a near-exact catalog hit (>= MISC_CATALOG_WINS) is a labeled item
#     and wins, even when the name also looks Misc-like (e.g. "Red Bull
#     Energy Drink" is a hard-truth Ready-To-Drink row);
#   - otherwise P(Misc) >= MISC_THRESHOLD makes it Misc, provided at least
#     MISC_MIN_COVERAGE of its words were seen in training (an unfamiliar,
#     evidence-free name stays blank/low-confidence rather than being
#     asserted as non-alcoholic). A real Misc item
#     the catalog matched well (MISC_REAL_ITEM_MIN, e.g. "Tres Agaves
#     Margarita Mix") keeps its own name.
# Thresholds were chosen on held-out cross-validation; see evaluate.py and the held-out numbers in its commit message.
MISC_THRESHOLD = 0.6
MISC_THRESHOLD_WEAK = 0.5   # used when the catalog's best score is below MISC_WEAK_CATALOG
MISC_WEAK_CATALOG = 0.5
MISC_HINT_THRESHOLD = 0.98  # alcohol brand/varietal hint present and nothing marking a non-beverage
MISC_MIN_COVERAGE = 0.3
MISC_CATALOG_WINS = 0.99
MISC_REAL_ITEM_MIN = 0.85

# Re-rank the top RERANK_N catalog candidates by word overlap incl. stop words (see _rr_tokens).
RERANK_W = 1.0
RERANK_N = 10
MISC_SIZE_PENALTY = 0.3   # a raw name with a size / pack count is a drink, not a coaster or cigar
EXACT_W = 1.0   # the raw text, cleaned, is exactly this item's name
RERANK_DELTA = 0.1   # only candidates this close to the top raw score are re-ranked
SEQ_W = 0.3   # word-order sensitive string similarity, same re-rank
_RR_CLASS = {'gift', 'set', 'with', 'can', 'cans', 'bottle', 'bottles', 'pack', 'pk'}
_RR_PACK_RE = re.compile(r"(?<![a-z0-9])\d+\s*(?:pk|packs?|ct)(?![a-z])")

# ---- size parsing: item size / pack count / unit, as three separate
# fields rather than one free-text string -- see parse_size() below. Not
# "proof": that's an ABV declaration, not a size, and is handled entirely
# separately (PROOF_ABV_RE) -- treating it as one caused the exact bug
# these regexes replaced (re.search returning "80 Proof" ahead of the
# actual size "750ml" just because it comes first in the raw text).
_UNIT_ALT = r"(ml|lt|l|liter|litre|fl\.?\s*oz|fz|oz)"
_PACK_ALT = r"(?:pk|packs|pack|ct)"
# A raw word that's JUST a number fused to a unit with no space ("375ml",
# "750ml") is a size declaration, never identifying content -- but unlike
# "80 Proof" (two separate tokens, already handled by PROOF_ABV_RE/
# SIZE_NUMBER_RE discounting the "80"), TOKEN_RE fuses "375ml" into ONE
# alnum token, which then goes through match_raw_word's normal exact/
# embedding/fuzzy tiers like any other word. That let "1800 Reposado
# Tequila 375ml Bottle" fuzzy-match "375ml" to "375mls" (90.9% similarity)
# -- the plural ONLY exists because of one oddball gift-set item, "Patron
# Silver With Three 375mls Gift" -- inflating the query's own norm with a
# term that has nothing to do with the actual "1800 Reposado Tequila" and
# dragging an otherwise-perfect match down to 71%. Skipped entirely in
# match_raw_word, not just discounted: there is no legitimate catalog word
# this pattern should ever resolve to.
_COMPOUND_SIZE_TOKEN_RE = re.compile(rf"^\d+(?:\.\d+)?{_UNIT_ALT}s?$", re.I)
# Real multipacks in this domain top out around 20-24 (a case of cans);
# nothing sold as a "pack" is a 3-digit number. Bounding every pack-count
# capture to this range (not just the ambiguous tight-hyphen case below)
# keeps a bare proof value or year from ever being read as a pack count.
_PACK_NUM = r"(?:[1-9]|1\d|2[0-4])"

# "4-355 Ml" / "8-12 FZ" -- a pack count and an item size, TIGHTLY
# hyphenated (no surrounding spaces). The tight hyphen matters: a LOOSE
# one ("Toba 80 - 750 Ml") is how this catalog writes a bare proof number
# (no "Proof" word attached) followed by the real size, not a multipack --
# found via a full-dataset sweep turning up "80"/"90"/"94" as bogus "pack
# counts" before this distinction was added.
_SIZE_COMPOUND_A_RE = re.compile(rf"\b({_PACK_NUM})-(\d*\.\d+|\d+)\s*{_UNIT_ALT}\b", re.I)
# "4 Pk 355 Ml" / "4pk-200ml" -- pack count and unit-word stated
# separately, pack first.
_SIZE_COMPOUND_B_RE = re.compile(rf"\b({_PACK_NUM})\s*{_PACK_ALT}\b.{{0,25}}?(\d*\.\d+|\d+)\s*{_UNIT_ALT}\b", re.I)
# "50ml-10pk" -- the reverse order, size first.
_SIZE_COMPOUND_C_RE = re.compile(rf"(?<![A-Za-z0-9])(\d*\.\d+|\d+)\s*{_UNIT_ALT}\b\s*-\s*({_PACK_NUM})\s*{_PACK_ALT}\b", re.I)
# Either half appearing alone, with nothing to pair it with.
_SIZE_STANDALONE_RE = re.compile(rf"(?<![A-Za-z0-9])(\d*\.\d+|\d+)\s*{_UNIT_ALT}\b", re.I)
_PACK_STANDALONE_RE = re.compile(rf"\b({_PACK_NUM})\s*{_PACK_ALT}\b", re.I)

EMB_THRESH = 0.72
FUZZ_THRESH = 82
MIN_EMBEDDING_LEN = 4  # see note in build_graph / design doc: spaCy's medium
                        # model hash-collides short/rare tokens onto unrelated
                        # vectors below this length -- skip the tier entirely.

# Below this length, rapidfuzz's fuzz.ratio is also unreliable, for a
# related but separate reason: it scores similarity via matching-block
# length (Gestalt/Ratcliff-Obershelp), not plain edit distance, so short
# strings can score deceptively high off a small absolute number of shared
# characters. Found via "ane" (from "1800 ane") fuzzy-matching "cane" at
# 85.7% -- comfortably over the 82 cutoff -- while the actually-intended
# "anejo" only scores 75% and never even gets considered, since only the
# single best-scoring candidate is returned. "cane" then wins the whole
# query outright, producing "Fireball Candy Cane Gift" for a query about
# 1800 Anejo Tequila. This is specific to the Python/rapidfuzz engine (this
# backend, and the offline pipeline) -- the browser-only client-side port
# uses a plain normalized-Levenshtein ratio, which scores this exact case
# below 82 for both candidates, so it never had this bug.
MIN_FUZZY_LEN = 4

# A merge of two adjacent raw tokens REPLACES both of their individual
# contributions (see predict()), so a bad merge is expensive: it can throw
# away a confident exact/alias match. Merging means "the user typed one
# canonical word split across two tokens", possibly truncated -- so the
# concatenation should align with the canonical word from the START, not
# merely score well overall. Without this guard, "1800 ult mar" merged
# "ult"+"mar" -> "ultima" (fuzz 83.3, just over the cutoff), which threw
# away the correct "ult" -> "ultimate" alias match and sent a 1800 query to
# "Don Julio Ultima Anejo Tequila". The genuine merges keep working:
# "limon"+"cell" -> "limoncello" shares a 9-character prefix, and
# "limon"+"cello" -> "limoncello" is an exact hit that skips fuzzy entirely.
MERGE_MIN_PREFIX = 4

# An accessory/bundle item ("... With Glasses", "... Gift Set With 2 Cans")
# is a different SKU from the plain bottle, and a raw name that never
# mentions the accessory must never be tagged as one -- no matter how well
# the rest of the name scores. Bundles are recognised structurally, by a
# "with" connector or the word "gift" in the canonical name, and the words
# after that connector are what the raw text has to mention. Matching on a
# word list alone would be wrong: "can", "bowl", "tin" and "rocks" are
# ordinary product words in this catalog ("Jack Daniels Cherry Coke Can",
# "Crown Royal NFL Super Bowl Lix", "Tin Cup Straight Bourbon", "On The
# Rocks Mule Can"), so only items that actually carry a with-clause or
# "gift" are ever subject to this rule.
WITH_CONNECTOR_RE = re.compile(r"\b(?:with|w/|w)\b", re.I)
GIFT_RE = re.compile(r"\bgift\b", re.I)
# A with-clause word counts as accessory evidence only if it is genuinely
# accessory vocabulary rather than an ordinary product word -- measured on
# the catalog itself: it must appear at least twice as often inside bundle
# names as it does in plain item names. That keeps "glasses" (48 vs 2),
# "glass" (41 vs 16) and "flask", and drops "tequila" (25 vs 401),
# "whiskey" (17 vs 351) and "blanco" (15 vs 79).
ACCESSORY_FREQ_RATIO = 2.0

# Domain alias dictionary: spirits-industry synonyms/shorthand a generic
# pretrained embedding doesn't know. Matched at full confidence alongside
# exact match.
ALIASES = {
    'silver': ['blanco'],
    'blanco': ['silver'],
    'white': ['blanco'],
    'ult': ['ultimate'],
    'ultimate': ['ult'],
    'marg': ['margarita'],
    'margarita': ['marg'],
    # Plural variant of the same word -- the catalog itself is inconsistent
    # about which form an item's own name uses ("12 Years aged" vs some
    # other item's singular "18 Year Old"), so a raw name using the other
    # form was scored as if it were unrelated content, diluting confidence
    # on an otherwise complete match (e.g. "Johnnie Walker Aged 18
    # Years..." vs the catalog's own "...18 Year Old...").
    #
    # "whisky"/"whiskey" was tried the same way and dropped: unlike this
    # pair, both spellings are independently common across many DIFFERENT
    # items (382 use "whiskey", 40 use "whisky"), so adding one as the
    # other's alias usually added pure noise to an item that already used
    # its own preferred spelling, rather than helping -- full reference set
    # went from 80.4465% to 80.3792% item accuracy with it in.
    'years': ['year'],
    'year': ['years'],
    # A compound-word split, not a synonym: this catalog spells one
    # Buchanan's item "De Luxe" (two words) while every raw name and every
    # other item using the word spells it "Deluxe" (one word) -- an alias
    # entry can point at more than one canonical word, so this reaches
    # both halves. One-directional: "de" alone is too common/short a
    # fragment to safely trigger back onto "deluxe".
    'deluxe': ['de', 'luxe'],
    # Shorthand seen in the raw data for Jack Daniel's lines.
    'jd': ['jack', 'daniels'],
    'jdcc': ['jack', 'daniels', 'country', 'cocktail'],
}

# Apostrophes are a real inconsistency between raw and canonical spelling,
# not just noise: TOKEN_RE splits "Angel's" into "angel" + "s", and the
# lone "s" is then dropped by predict()'s len>1 filter -- so a possessive
# brand loses its trailing letter ("angel", not "angels") while the
# catalog's own cleaning just deleted the punctuation and kept the letter
# attached ("Angels Envy Triple Oak Bourbon"). That silent mismatch showed
# up as "angel" not being part of "Angels Envy"'s own vector, understating
# confidence on an otherwise perfect match. Removing the apostrophe
# CHARACTER before tokenizing (rather than letting TOKEN_RE treat it as a
# separator) aligns the two: "Angel's" -> "Angels", "Jack Daniel's" ->
# "Jack Daniels" -- the same way the catalog's own text was cleaned.
APOSTROPHE_RE = re.compile(r"[’']")

# A proof/ABV declaration ("80 Proof", "92 Proof", "42% ABV", "15%ABV") is
# a specification of the product, not a word that identifies WHICH item it
# is -- and some of these numbers ("80", "92"...) also happen to be exact
# words in the catalog vocabulary purely because some OTHER item's own name
# happens to contain that number as its own proof declaration (e.g. "New
# Amsterdam Vodka 80 Proof"). Counted at full weight, they inflate the
# raw-side vector's norm without adding anything to the correct item's own
# score, understating confidence on names that are otherwise a complete,
# correct match (see the Bacardi/1800/Angel's Envy/BuzzBallz cases in the
# design doc). Down-weighted, not excluded outright, since the underlying
# number could occasionally still be part of a real distinguishing term.
PROOF_ABV_RE = re.compile(r"\b(\d+)(?:\.\d+)?\s*%?\s*(?:proof|abv)\b", re.I)

# Same reasoning, for a plain size/pack declaration ("50 ml", "750ml",
# "1.75 L", "8pk") instead of a proof/ABV one: "Patron Tequila Reposado 50
# ml" matched all 3 of "Patron Reposado Tequila"'s own words at full
# confidence and still scored 71%, because "50" also happens to be an
# exact catalog word (from some unrelated item's own "50 ml"/"50" pack
# count) with nothing to do with THIS item. Reuses the same unit
# vocabulary as parse_size()'s _UNIT_ALT/_PACK_ALT so the two stay in
# sync; "proof" is deliberately left out here since PROOF_ABV_RE already
# covers it with its own (looser) pattern.
SIZE_NUMBER_RE = re.compile(rf"\b(\d+)(?:\.\d+)?\s?(?:{_UNIT_ALT[1:-1]}|{_PACK_ALT[3:-1]})\b", re.I)

# Generic RTD/serving boilerplate ("Ready to Drink", "Ready to Serve") --
# down-weighted for the same reason as proof/ABV, but NOT zeroed like a
# stopword: "drink" is a genuine distinguishing word for 34 real catalog
# items (BuzzBallz ... Mixed Drink, Long Drink, ...), and "ready"/"serve"
# for Skyy's "Ready-to-serve" line, so fully excluding them would break
# those. A steep discount keeps them from dominating a match the way a raw
# name's real content words should, while still letting them break a tie
# when nothing else does.
LOW_SIGNAL_WORDS = {'ready', 'to', 'drink', 'drinks', 'serve', 'served', 'gluten', 'free', 'flavored'}
# "pet" (PET-plastic bottle material, as in "200 mL PET Bottle") was tried
# here too and reverted: unlike the words above, it turned out to be a
# genuine, undiscounted part of several real items' own canonical names
# ("Fireball Blazin Apple Pet Whiskey", "Jim Beam Pet Bourbon", "Bacardi
# Oakheart (PET bottle)", ...), not a narrow one-off. Discounting it
# helped one reported case (Bulleit Bourbon Whiskey in a PET bottle) but
# net cost more elsewhere on the full reference set (80.4771% ->
# 80.3180% item accuracy) -- reverted, and left as a note so it isn't
# tried again without re-checking this.
LOW_SIGNAL_WEIGHT = 0.15

# See predict()'s minimum-coverage filter: the fraction of an item's own
# distinguishing words that the raw text has to explain before that item is
# even allowed to be a candidate, regardless of how any one shared word
# scores. 0.5 keeps every item whose vector is fully or mostly matched
# (including short, single-word-vector items -- 1/1 clears it) and drops
# an item explained by only a minority of its own name.
MIN_ITEM_COVERAGE = 0.5


class MatchEngine:
    def __init__(self, graph_path: Path, use_embeddings: bool = True):
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        self.items = G['items']
        self.item_brand = G['item_brand']
        self.item_cat = G['item_cat']
        self.item_vec = G['item_vec']
        self.item_norm = G['item_norm']
        self.inv_index = G['inv_index']
        self.idf = G['idf']
        self.canon_vocab = G['canon_vocab']
        self.STOP = G['STOP']
        self.misc_model = MiscModel(G['misc_model']) if G.get('misc_model') else None
        self.misc_exact = G.get('misc_exact') or frozenset()
        self._item_rr = {}

        self.nlp = None
        self.emb_words = []
        self.emb_matrix = None
        if use_embeddings:
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_md', disable=['tagger', 'parser', 'ner', 'lemmatizer'])
                self.emb_words = G.get('emb_words', [])
                self.emb_matrix = G.get('emb_matrix')
                if self.emb_matrix is None or len(self.emb_words) == 0:
                    log.warning("graph.pkl has no embedding table -- embedding tier disabled")
                    self.nlp = None
                else:
                    log.info("Pretrained embeddings loaded (%d words)", len(self.emb_words))
            except Exception as e:
                log.warning("spaCy/en_core_web_md not available (%s) -- "
                            "running without the embedding tier. Install with: "
                            "pip install spacy && python -m spacy download en_core_web_md", e)
                self.nlp = None

        # Brand hard-lock patterns, longest brand first so "Jim Beam" is
        # tried before a shorter brand that might be a substring of it.
        # Brands are compared by a key with apostrophes removed ("Jack Daniel's"
        # and "Jack Daniels" are one brand family): the hard truth spells some
        # brands both ways, and keying on the literal string used to hide every
        # item filed under the other spelling from the brand lock.
        all_brands = sorted({b for b in self.item_brand.values() if b}, key=len, reverse=True)
        rep_by_key = {}
        for b in all_brands:
            rep_by_key.setdefault(self._bk(b), b)
        self._brand_patterns = [(b, re.compile(r"\b" + re.escape(k) + r"\b")) for k, b in rep_by_key.items()]
        self._brand_keys = list(rep_by_key)
        self._brand_word_sets = [
            (b, pat, frozenset(self._acc_norm(w) for w in self.tokenize(b)))
            for b, pat in self._brand_patterns
        ]
        self.brands_all = all_brands
        self.categories_all = sorted({c for c in self.item_cat.values() if c})
        self.brand_token_family = self._build_brand_token_index()
        self.item_accessories = self._build_accessory_index()

        # Caches for predict()'s per-candidate filtering (brand lock, brand
        # anchor, accessory rule). These used to look up/lower() every
        # candidate item's brand on every call; on a large upload that is
        # O(rows x candidates), the dominant cost once the catalog is big
        # enough (profiled: ~2.8M redundant calls tagging the 8,175-row
        # reference set). Precomputing them once at load time turns each
        # filter into a couple of set operations instead.
        self._item_brand_lower = {it: self._bk(self.item_brand.get(it) or '') for it in self.items}
        self._items_by_brand_lower = {}
        for it, bl in self._item_brand_lower.items():
            if bl:
                self._items_by_brand_lower.setdefault(bl, []).append(it)
        self._bundle_item_set = set(self.item_accessories.keys())
        # brand string (lower) -> its own tokenized/plural-folded words, for
        # _brand_word_matched(): this catalog now prefixes almost every
        # item's own name with its brand (99% of items -- see design doc),
        # so requiring the raw text to actually explain THAT word, not just
        # a generic majority of the item's other words, is now a safe,
        # high-value check rather than the blunt instrument it was on the
        # older, less consistently brand-prefixed catalog.
        self._brand_tokens_by_lower = {
            self._bk(b): frozenset(self._acc_norm(w) for w in self.tokenize(b))
            for b in all_brands
        }

    @staticmethod
    def _bk(b):
        return APOSTROPHE_RE.sub("", str(b).lower())

    # ---- accessory / bundle items ----
    @staticmethod
    def _acc_norm(w):
        """Fold the obvious plurals so "2 Cans" is matched by a raw "can".

        Applied until stable, so that a word and its plural always land on
        the same form regardless of which side of the comparison they are
        on: "glasses" -> "glass" -> "glas" and "glass" -> "glas" alike,
        where a single pass would leave them as "glass" vs "glas" and never
        match. Over-stemming is harmless here -- both sides go through it.
        """
        while True:
            if len(w) > 4 and w.endswith("es"):
                w = w[:-2]
            elif len(w) > 3 and w.endswith("s"):
                w = w[:-1]
            else:
                return w

    def _accessory_clause(self, item):
        """The words an item's accessory is named by, or [] if it isn't a bundle."""
        parts = WITH_CONNECTOR_RE.split(str(item))
        if len(parts) > 1:
            return self.tokenize(" ".join(parts[1:]))
        if GIFT_RE.search(str(item)):
            return self.tokenize(item)
        return []

    def _build_accessory_index(self):
        """item -> the set of words the raw text must mention to be allowed to match it."""
        clauses = {it: self._accessory_clause(it) for it in self.items}
        in_bundle, in_plain = {}, {}
        for it in self.items:
            if clauses[it]:
                for w in set(clauses[it]):
                    in_bundle[w] = in_bundle.get(w, 0) + 1
            else:
                for w in set(self.tokenize(it)):
                    in_plain[w] = in_plain.get(w, 0) + 1

        index = {}
        for it, clause in clauses.items():
            if not clause:
                continue
            terms = {
                self._acc_norm(w) for w in clause
                if w == "gift" or in_plain.get(w, 0) == 0
                or in_bundle.get(w, 0) >= ACCESSORY_FREQ_RATIO * in_plain.get(w, 0)
            }
            terms.discard("")
            if terms:
                index[it] = frozenset(terms)
        return index

    def mentions_accessory(self, raw_norm_words, item):
        """True unless `item` is a bundle whose accessory the raw text never names.

        A misspelling is still a mention -- "1800 Blanco W/2 Galsses" is
        plainly asking for the glasses bundle -- so the accessory word is
        also allowed to arrive by fuzzy match, at the same threshold the
        rest of the engine uses. Only genuine silence about the accessory
        rules the bundle out.
        """
        terms = self.item_accessories.get(item)
        if not terms:
            return True
        if raw_norm_words & terms:
            return True
        for term in terms:
            if len(term) < MIN_FUZZY_LEN:
                continue
            for w in raw_norm_words:
                if len(w) >= MIN_FUZZY_LEN and fuzz.ratio(w, term) >= FUZZ_THRESH:
                    return True
        return False

    def _item_well_covered(self, item, raw_term_weight):
        """Whether a real share of `item`'s own name is explained by the raw text.

        Presence-based (does the canonical word show up at all), not
        weighted -- a low-signal/discounted word (see LOW_SIGNAL_WORDS)
        still counts as "this word was mentioned", which is what coverage
        is asking.

        Requires BOTH a minimum ratio and a minimum absolute count of
        matched words, not ratio alone: a bare ratio let a 2-word-vector
        item ("Kraken Kraken & Cola Pack" -- vector {kraken, cola}) through
        on a single matched word ("cola") purely because 1/2 happens to
        clear a 50% cutoff, the same failure mode this whole rule exists to
        catch. Two matched words is a much harder coincidence than one.
        """
        iv = self.item_vec.get(item)
        if not iv:
            return False
        matched = sum(1 for cw in iv if cw in raw_term_weight)
        return matched >= 2 and matched / len(iv) >= MIN_ITEM_COVERAGE

    def _brand_word_matched(self, item, raw_term_weight):
        """Whether the raw text explains `item`'s own BRAND word specifically,
        not just some majority of its other, more generic words.

        "Hazelbaker Bottled In Bond 6yr Ky Bourbon 750 ml" matched "Bottled"/
        "Bond"/"Bourbon" -- 3 of "Bulleit Bottled In Bond Bourbon"'s 4 own
        terms, comfortably clearing _item_well_covered on ratio alone --
        while never mentioning "Bulleit" (or "Hazelbaker", the brand it
        actually names) at all. "Bottled In Bond" is itself a distinctive,
        high-idf phrase shared by only a few bourbons, so a plain coverage
        ratio has no way to tell that apart from genuine brand evidence.
        Requiring the brand word too closes that gap: 99% of this catalog's
        items already have their own brand as one of their own vector
        terms (see design doc), so this is a real, high-value signal now
        rather than the blunt, brand-column-mismatched instrument an
        earlier version of this same idea turned out to be on the older
        catalog (see the design doc's note on why that version was
        reverted -- items without their own brand in their own name are
        exempted here, not penalized, for exactly that reason).
        """
        bl = self._item_brand_lower.get(item, '')
        btoks = self._brand_tokens_by_lower.get(bl)
        if not btoks:
            return True
        iv = self.item_vec.get(item, {})
        if not (btoks & iv.keys()):
            # Self-validation: this item's OWN brand-column value doesn't
            # even appear in its OWN printed name, i.e. the catalog's brand
            # metadata is internally inconsistent for this one item --
            # found via "Parrot Bay Gold Rum", filed under brand "Patron"
            # (unrelated majority-vote noise from build_graph.py, not
            # anything about this item). Enforcing a brand word that isn't
            # really part of the item's own identity would only produce a
            # false block, so this item is treated the same as one with no
            # known brand at all -- exempt, not penalized.
            return True
        # _brand_tokens_by_lower is plural-folded (the Blanton's fix --
        # "Blanton" has no trailing "s" at all, but raw "Blanton's"
        # apostrophe-strips to "blantons"). The matched term needs the same
        # fold to compare fairly: folding is symmetric, so a brand that
        # legitimately ends in "s" after its own apostrophe-strip ("Tito's"
        # -> "titos") still meets a raw "titos" match at the same folded
        # "tito", rather than only the brand side getting over-stripped.
        raw_folded = {self._acc_norm(w) for w in raw_term_weight}
        return bool(btoks & raw_folded)

    # ---- brand-token priority ----
    def _build_brand_token_index(self):
        """Map each *distinctive* brand word to the brand family it belongs to.

        The full-name brand lock (detect_brand) only fires when the whole
        brand string is present in the raw text -- "1800 ult mar" never
        contains the literal brand "1800 Tequila", so nothing stopped it
        escaping to another brand entirely. A single brand word is the far
        more common case in messy POS text, and it should carry the same
        weight: a query naming "1800" must not come back as a Don Julio item.

        Not every brand word can be trusted to do that, though -- "tequila"
        and "margarita" are both brand words ("1800 Tequila", "Jose Cuervo
        Golden Margarita") but they are obviously generic. Rather than guess
        at a stopword list or an idf cutoff, a word earns brand-anchor status
        from the catalog itself: it is an anchor only if EVERY canonical item
        whose name contains it already belongs to that word's brand family.
        By construction, locking on an anchor can never exclude a catalog
        item that legitimately uses the word. That admits "1800", "absolut",
        "cuervo", "buzzballz" (106 words) and rejects "tequila", "margarita",
        "mezcal", "jose", "one", "el" -- no thresholds involved.
        """
        brand_tokens = {}
        for b in self.brands_all:
            for w in self.tokenize(b):
                if len(w) < 2:
                    continue
                brand_tokens.setdefault(w, set()).add(self._bk(b))

        items_with_word = {}
        for it in self.items:
            for w in set(self.tokenize(it)):
                items_with_word.setdefault(w, []).append(it)

        anchors = {}
        for w, family in brand_tokens.items():
            occurrences = items_with_word.get(w)
            if not occurrences:
                continue
            if all(self._bk(self.item_brand.get(it) or '') in family for it in occurrences):
                anchors[w] = frozenset(family)
        return anchors

    def anchor_brand_family(self, terms):
        """Brand family the given words pin the prediction to, if any.

        Multiple anchors are intersected when they agree (e.g. "1800" ->
        {1800 Tequila, 1800 Ultimate Margarita} narrowed by "ultimate" ->
        {1800 Ultimate Margarita}); an anchor that disagrees with what is
        already pinned is ignored rather than emptying the set.
        """
        allowed = None
        for w in terms:
            # The anchor dict's own keys come from each catalog ITEM's
            # literal tokenization, deliberately left unfolded (see
            # _build_brand_token_index -- folding those risked pulling in
            # an unrelated item whose name just happens to share the
            # singular/plural form). The raw QUERY side still needs it,
            # though: a brand this catalog spells without a trailing "'s"
            # ("Blanton") otherwise never matches a raw "Blanton's" once
            # apostrophe-stripping turns that into "blantons".
            family = self.brand_token_family.get(w) or self.brand_token_family.get(self._acc_norm(w))
            if not family:
                continue
            if allowed is None:
                allowed = set(family)
            elif allowed & family:
                allowed &= family
        return allowed

    # ---- tokenizing ----
    def tokenize(self, s):
        s = APOSTROPHE_RE.sub("", str(s).lower())
        return TOKEN_RE.findall(s)

    def parse_size(self, raw):
        """Split a raw name's size info into three separate fields instead of
        one free-text string: item_size (e.g. "750"), pack_count (e.g. "4"
        from a 4-pack), and unit (e.g. "Ml"). Tries the compound patterns
        (both a pack count and an item size present, in either order and
        notation) before falling back to whichever half appears alone.
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


    # ---- word-level matching ----
    def _raw_word_vec(self, w):
        if self.nlp is None:
            return None
        lex = self.nlp.vocab[w]
        if lex.has_vector and lex.vector_norm > 0:
            return lex.vector / lex.vector_norm
        return None

    def _exact_or_alias_match(self, w):
        """The exact/alias tier only -- see match_raw_word. Used by the merge
        decision (predict()) to decide whether a word is trustworthy enough
        on its own to skip a merge attempt for its neighbor. Embedding/fuzzy
        hits are excluded on purpose: unlike an exact or alias hit, they can
        be spurious lookalikes rather than the word actually being present
        in the catalog (see MIN_EMBEDDING_LEN's note on hash-collisions) --
        "cello" hash-collided onto the unrelated word "alto" at ~1.0
        similarity in one rebuild, which, if trusted here, would report
        "cello" as already matched and never let the merge fallback try
        "limoncello" at all.
        """
        matches = []
        if w in self.idf:
            matches.append((w, 1.0))
        for alias in ALIASES.get(w, []):
            if alias in self.idf:
                matches.append((alias, 1.0))
        return matches

    def match_raw_word(self, w):
        if _COMPOUND_SIZE_TOKEN_RE.match(w):
            return []
        matches = self._exact_or_alias_match(w)
        if matches:
            return matches
        best = None
        v = self._raw_word_vec(w) if len(w) >= MIN_EMBEDDING_LEN else None
        if v is not None and len(self.emb_words) > 0:
            sims = self.emb_matrix @ v
            j = int(np.argmax(sims))
            if sims[j] >= EMB_THRESH:
                best = (self.emb_words[j], float(sims[j]))
        if len(w) >= MIN_FUZZY_LEN:
            fz = process.extractOne(w, self.canon_vocab, scorer=fuzz.ratio, score_cutoff=FUZZ_THRESH)
            if fz is not None:
                cand = (fz[0], fz[1] / 100.0)
                if best is None or cand[1] > best[1]:
                    best = cand
        return [best] if best else []

    @staticmethod
    def _shares_prefix(a, b):
        need = min(MERGE_MIN_PREFIX, len(a), len(b))
        return a[:need] == b[:need]

    def merged_word_match(self, w1, w2):
        # Two adjacent NUMBERS are never a word split across two tokens --
        # that's what the merge fallback exists for (a typo/space breaking
        # up something like "limoncello"). Concatenating them is just
        # coincidence, and coincidence is exactly what causes trouble: "19"
        # + "75" (from "...19.9 Proof 1.75 L") exact-matches the unrelated
        # catalog word "1975", injecting a pure-noise term for a totally
        # different item into a query that never mentioned it.
        if w1.isdigit() and w2.isdigit():
            return []
        merged = w1 + w2
        if merged in self.idf:
            return [(merged, 1.0)]
        fz = process.extractOne(merged, self.canon_vocab, scorer=fuzz.ratio, score_cutoff=FUZZ_THRESH)
        if fz is not None and self._shares_prefix(merged, fz[0]):
            return [(fz[0], fz[1] / 100.0)]
        return []

    def detect_brand(self, raw_name):
        raw_lower = APOSTROPHE_RE.sub("", raw_name.lower())
        # Plural-folded (see _acc_norm) so a brand this catalog spells
        # WITHOUT a trailing "'s" ("Blanton") still overlaps a raw name
        # that has it ("Blanton's" -> tokenize's apostrophe-stripping turns
        # this into "blantons", not "blanton") -- this prefilter is only a
        # performance guard, so over-matching here is harmless; the regex
        # search right after is what actually decides.
        raw_words = {self._acc_norm(w) for w in self.tokenize(raw_name)}
        found = []
        # A brand can only appear in the raw text if it shares at least one
        # word with it -- checking that with a set intersection first skips
        # the (comparatively expensive) regex .search() for the ~124 brands
        # that plainly cannot match, rather than running all of them on
        # every single raw name.
        for b, pat, words in self._brand_word_sets:
            if not (words & raw_words):
                continue
            if pat.search(raw_lower):
                if any(self._bk(b) in self._bk(f) for f in found):
                    continue
                found.append(b)
        return found[0] if len(found) == 1 else None

    def matched_words_for(self, raw_name):
        words = [w for w in self.tokenize(raw_name) if w not in self.STOP and len(w) > 1]
        out = []
        for w in words:
            m = self.match_raw_word(w)
            out.append({"raw": w, "canon": m[0][0] if m else None, "conf": round(m[0][1], 3) if m else 0})
        return out

    # ---- full prediction ----
    def _raw_dot(self, raw_name):
        """Pure TF-IDF/graph-weight scoring, before any of the hard
        structural rules (accessory, brand lock, brand anchor, ambiguity
        gate) narrow it down. Shared by predict() (which applies those
        rules) and predict_unfiltered() (which deliberately doesn't, for
        reviewing what the raw graph weights alone would suggest when
        predict() itself came back blank).
        """
        words = [w for w in self.tokenize(raw_name) if w not in self.STOP and len(w) > 1]
        word_matches = [self.match_raw_word(w) for w in words]

        # Proof/ABV declarations ("80 Proof", "42% ABV") are product spec,
        # not identifying content -- down-weight whichever raw word carries
        # the number, rather than treating it as ordinary evidence.
        spec_numbers = set(PROOF_ABV_RE.findall(raw_name)) | set(SIZE_NUMBER_RE.findall(raw_name))

        # Merge adjacent pairs only when at least one word failed to match
        # EXACTLY/via alias alone; a successful merge REPLACES both words'
        # individual contributions (see design doc SS3g -- "limon cell" ->
        # "limoncello" case). Deliberately checked against the exact/alias
        # tier only, not the full word_matches (which also includes
        # embedding/fuzzy hits): an embedding hash-collision can make a word
        # look "matched" at ~1.0 similarity to something completely
        # unrelated (e.g. "cello" -> "alto"), and trusting that would block
        # the merge from ever being tried at all.
        exact_matches = [self._exact_or_alias_match(w) for w in words]
        consumed = set()
        merge_contribs = []
        for i in range(len(words) - 1):
            if exact_matches[i] and exact_matches[i + 1]:
                continue
            mm = self.merged_word_match(words[i], words[i + 1])
            if not mm:
                continue
            # Reject a merge that just re-finds a word ALREADY matched
            # exactly, with junk appended -- not a genuine split-word
            # rescue. "Margarita 75l" merges to "margarita"+"75l" ->
            # fuzzy-matches back to plain "margarita" at 85.7%, which would
            # otherwise REPLACE "margarita"'s own perfect 100% exact match
            # with a worse re-discovery of the exact same word, for no
            # reason other than an unrelated size token sitting next to it.
            already_exact = {cw for cw, _ in exact_matches[i]} | {cw for cw, _ in exact_matches[i + 1]}
            if all(cw in already_exact for cw, _ in mm):
                continue
            merge_contribs.append(mm)
            consumed.add(i)
            consumed.add(i + 1)

        raw_term_weight = {}
        confident_terms = []  # exact/alias hits only -- used for the brand-anchor lock
        full_weight_terms = set()  # matched terms that AREN'T low-signal/proof-abv discounted
        any_unmatched = False  # a content word that matched nothing at all, in any tier
        # Tracked separately from raw_term_weight for the norm (see below):
        # one entry per raw WORD, not per canonical term it maps to.
        position_weights = []
        for idx, matches in enumerate(word_matches):
            if idx in consumed:
                continue
            if not matches:
                any_unmatched = True
                continue
            raw_w = words[idx]
            # Marketing/spec boilerplate ("Ready to Drink", "80 Proof")
            # counts for very little -- down-weighted rather than excluded,
            # since a couple of these words ("drink") are also the genuine
            # distinguishing word for real catalog items.
            discount = LOW_SIGNAL_WEIGHT if (raw_w in LOW_SIGNAL_WORDS or raw_w in spec_numbers) else 1.0
            # The FIRST match, not the largest -- _exact_or_alias_match (and
            # therefore match_raw_word) always puts the word's own exact
            # self-match first when one exists, only falling back to an
            # alias/embedding/fuzzy candidate otherwise. Taking the max
            # instead re-broke "Cutwater Peach Margarita": raw "margarita"
            # exact-matches itself (idf 4.10) AND, via the marg<->margarita
            # alias, also matches "marg" -- which happens to have a HIGHER
            # idf (7.78, since "marg" is rarer in the catalog) purely
            # because it's a less common word, not because it's the right
            # interpretation of what was actually typed. What was actually
            # typed should win.
            position_best = None
            for cw, conf in matches:
                w = conf * self.idf.get(cw, 1.0) * discount
                raw_term_weight[cw] = raw_term_weight.get(cw, 0.0) + w
                if position_best is None:
                    position_best = w
                if conf >= 1.0 and discount == 1.0:
                    confident_terms.append(cw)
                if discount == 1.0:
                    full_weight_terms.add(cw)
            position_weights.append(position_best)
        for mm in merge_contribs:
            merge_best = 0.0
            for cw, conf in mm:
                w = conf * self.idf.get(cw, 1.0)
                raw_term_weight[cw] = raw_term_weight.get(cw, 0.0) + w
                merge_best = max(merge_best, w)
                full_weight_terms.add(cw)
                if conf >= 1.0:
                    confident_terms.append(cw)
            position_weights.append(merge_best)

        dot = {}
        for cw, rw in raw_term_weight.items():
            for it, iw in self.inv_index.get(cw, []):
                dot[it] = dot.get(it, 0.0) + rw * iw
        # The query's own norm is computed per RAW WORD (one contribution
        # each), not per canonical term raw_term_weight ends up holding --
        # those aren't the same thing once an alias is involved. "Cutwater
        # Peach Margarita" exact-matches "margarita" AND, via the marg<->
        # margarita alias (added for "1800 ult marg" -- see ALIASES),
        # ALSO matches "marg", so raw_term_weight carries two separate
        # entries from that one word. Both need to stay in the dot product
        # (so items spelled either way are still reachable), but counting
        # both squared into the query's own norm effectively records ONE
        # word as two independent pieces of evidence, inflating the
        # denominator against a target that (correctly) only uses one of
        # them -- "Cutwater Peach Margarita" matched all 3 of its own words
        # at full confidence and still scored 70%, purely from this. Taking
        # the larger of a position's alternatives for the norm (not the
        # sum) fixes that while leaving alias reachability untouched.
        raw_norm = math.sqrt(sum(w * w for w in position_weights)) or 1.0

        # Resolved once, reused by both the ambiguity gate below and the
        # narrowing steps further down.
        locked_brand = self.detect_brand(raw_name)
        anchor_family = self.anchor_brand_family(list(words) + confident_terms)

        return {
            "dot": dot, "raw_norm": raw_norm, "raw_term_weight": raw_term_weight,
            "words": words, "confident_terms": confident_terms,
            "full_weight_terms": full_weight_terms, "any_unmatched": any_unmatched,
            "locked_brand": locked_brand, "anchor_family": anchor_family,
        }

    def misc_prob(self, raw_name):
        """P(raw name is a Misc. Accessories & Gifts item); 1.0 for a raw
        name that is itself a labeled Misc row, 0.0 with no model loaded."""
        if self.misc_model is None:
            return 0.0
        if ' '.join(str(raw_name).lower().split()) in self.misc_exact:
            return 1.0
        return float(self.misc_model.prob(raw_name))

    def predict(self, raw_name, topk=3):
        cands = self._predict_catalog(raw_name, topk)
        top_score = cands[0].pop("top_score", None) if cands else None
        if self.misc_model is None:
            return cands
        # The Misc decision looks at the strongest catalog evidence, not at
        # whichever candidate the tie-break re-rank put first.
        best = dict(cands[0], score=top_score) if cands else None
        p = self.misc_prob(raw_name)
        if p < 1.0 and best and best["score"] >= MISC_CATALOG_WINS:
            return cands
        weak = best is None or best["score"] < MISC_WEAK_CATALOG
        need = MISC_THRESHOLD_WEAK if weak else MISC_THRESHOLD
        if p < 1.0 and alcohol_hint_blocks_misc(raw_name):
            need = max(need, MISC_HINT_THRESHOLD)
        if p < need:
            return cands
        if p < 1.0 and self.misc_model.coverage(raw_name) < MISC_MIN_COVERAGE:
            return cands
        if best and best["category"] == MISC_CATEGORY and best["score"] >= MISC_REAL_ITEM_MIN:
            return cands
        misc = {"item": MISC_CATEGORY, "brand": MISC_CATEGORY, "category": MISC_CATEGORY, "score": round(p, 4)}
        return [misc] + cands[:max(0, topk - 1)]

    def _predict_catalog(self, raw_name, topk=3):
        return self._catalog_only(raw_name, topk)

    def _catalog_only(self, raw_name, topk=3):
        s = self._raw_dot(raw_name)
        dot, raw_norm, raw_term_weight = s["dot"], s["raw_norm"], s["raw_term_weight"]
        words, confident_terms = s["words"], s["confident_terms"]
        full_weight_terms, any_unmatched = s["full_weight_terms"], s["any_unmatched"]
        locked_brand, anchor_family = s["locked_brand"], s["anchor_family"]

        # Ambiguous-match hard rule: when the raw text names no specific
        # brand at all (no full brand string, no brand-anchor word) and the
        # surviving candidates span MORE THAN ONE brand, a single shared
        # word is not enough to pick a winner. A candidate is exempted from
        # this if it ALREADY fully accounts for the raw text on its own --
        # nothing in the raw text went unrecognized, and every full-weight
        # matched term belongs to that candidate's own vector -- since that
        # is a materially different, safer situation than "one coincidental
        # generic word, plus other content the candidate doesn't explain".
        #
        # This was tried first as an unconditional coverage requirement
        # (does the candidate cover a real SHARE of its own name), which
        # wrongly caught genuinely short-but-clean queries like "TITOS" ->
        # "Titos Vodka" (the entire raw text is just the brand's own name;
        # its 2-word vector isn't "mostly covered" by that alone, but there
        # is nothing else in the raw text to explain) and "1800 ane" ->
        # "1800 Anejo Tequila" (all remaining candidates are 1800-family,
        # and "1800" is itself a confirmed brand anchor) and "limon cell"
        # -> "1010 Limoncello Cream" (the merge to "limoncello" leaves
        # exactly one candidate). All three are fully explained by the raw
        # text -- exempt here -- while "4Ks Diamond" leaves "4ks" totally
        # unrecognized, and "Coca Cola Classic 2 L" matches "classic" to
        # something that has nothing to do with the "Jack Daniels & Coca
        # Cola Whiskey" it would otherwise win on -- neither is exempt, and
        # both correctly end up blank. Re-verified against the full
        # reference set with no regression -- see the design doc for exact
        # before/after numbers.
        #
        # Coverage ratio alone is not enough when the raw text also has a
        # genuinely UNRECOGNIZED word, though: "Hazelbaker Bottled In Bond
        # 6yr Ky Bourbon 750 ml" matched "Bottled"/"Bond"/"Bourbon" -- 3 of
        # "Bulleit Bottled In Bond Bourbon"'s 4 own terms, comfortably over
        # the ratio cutoff -- while "Hazelbaker" (a real, different brand
        # this catalog doesn't carry) went completely unrecognized and
        # "Bulleit" was never mentioned at all. In that specific situation
        # -- an unrecognized word, itself plausibly a brand we don't carry
        # -- a non-exempt candidate must also explain its OWN brand word,
        # not just a majority of its other, more generic words (see
        # _brand_word_matched).
        #
        # Restricted to exactly that situation, not applied whenever a
        # brand word is merely absent: plenty of genuinely correct hard-
        # truth rows never name their brand at all and are still fully
        # explained by ordinary words ("15 Year Irish Whiskey" -> Redbreast,
        # "101 Kentucky Straight Bourbon Whiskey" -> Wild Turkey) -- nothing
        # about those raw names is suspicious, they simply don't repeat the
        # brand, which unconditionally requiring it wrongly punished (a
        # full-dataset run caught this: item accuracy fell to 79.35% before
        # this restriction, from the 80.4771% baseline).
        if locked_brand is None and not anchor_family:
            brands_present = {self._item_brand_lower.get(it, '') for it in dot}
            brands_present.discard('')
            if len(brands_present) >= 2:
                def _exempt(it):
                    if any_unmatched:
                        return False
                    iv = self.item_vec.get(it, {})
                    return full_weight_terms.issubset(iv)

                dot = {
                    it: d for it, d in dot.items()
                    if _exempt(it) or (
                        self._item_well_covered(it, raw_term_weight)
                        and (not any_unmatched or self._brand_word_matched(it, raw_term_weight))
                    )
                }

        # Accessory/bundle hard rule: a raw name that never mentions the
        # accessory can never be tagged as the bundle SKU, however well the
        # rest of it scores. Applied before the brand locks so those rank
        # only items this raw name is actually allowed to match, and applied
        # with every raw token (not just the scored ones), since "glass" and
        # "2" are exactly the words tokenizing would otherwise drop.
        #
        # Only 230 of 3,515 catalog items are bundles at all, so only
        # candidates that ARE one need the (comparatively expensive)
        # mentions_accessory() check -- the other ~94% pass automatically.
        # Checking the intersection first instead of calling it for every
        # candidate is what keeps a big upload (thousands of rows, each
        # with potentially hundreds of candidates for a common query like
        # "vodka") from spending most of its time here.
        raw_acc_words = {self._acc_norm(w) for w in self.tokenize(raw_name)}
        bundle_hits = self._bundle_item_set & dot.keys()
        if bundle_hits:
            failed = {it for it in bundle_hits if not self.mentions_accessory(raw_acc_words, it)}
            if failed:
                dot = {it: d for it, d in dot.items() if it not in failed}

        if locked_brand is not None:
            lb = self._bk(locked_brand)
            # Resolve the locked brand to its item set once (a handful of
            # brand-string comparisons among ~124 brands), then filter `dot`
            # with a plain set-membership test instead of a per-item
            # lower()/startswith() call -- the version this replaced showed
            # up as ~2.2M calls / 7s profiling the full reference set.
            family_items = set()
            for bl in self._brand_keys:
                if bl == lb or bl.startswith(lb) or lb.startswith(bl):
                    family_items.update(self._items_by_brand_lower.get(bl, ()))
            # Unconditional, not "if locked_dot" -- a raw name that
            # explicitly names one brand must never fall back to ranking
            # among whatever candidates happen to be left (which can
            # include a genuinely different brand also named in the raw
            # text). No candidate in the named brand's own family means no
            # candidate at all, not a guess from a brand the raw name never
            # asked for.
            dot = {it: d for it, d in dot.items() if it in family_items}

        # Brand-word priority: the raw text naming a brand word ("1800") pins
        # the prediction to that brand family even when the full brand string
        # ("1800 Tequila") never appears -- structurally, the same way the
        # full-name lock works, not as a scoring nudge that a rarer word from
        # some other brand could still outweigh. Unconditional for the same
        # reason as the full-name lock above: an anchor word, by
        # construction (see _build_brand_token_index), only ever occurs in
        # items belonging to its one brand family, so if nothing in `dot`
        # belongs to that family the raw name's own brand word is not
        # explaining any of the surviving candidates -- they belong to some
        # other, unnamed brand, and per hard requirement that is blank, not
        # a fallback guess.
        if anchor_family:
            anchor_items = set()
            for bl in anchor_family:
                anchor_items.update(self._items_by_brand_lower.get(bl, ()))
            dot = {it: d for it, d in dot.items() if it in anchor_items}

        scored = [(it, min(1.0, d / (raw_norm * self.item_norm[it]))) for it, d in dot.items()]
        scored.sort(key=lambda x: -x[1])
        top_score = scored[0][1] if scored else 0.0
        if RERANK_W > 0 and len(scored) > 1:
            scored = self._rerank(raw_name, scored)
        top = scored[:topk]
        out = [
            {"item": it, "brand": self.item_brand.get(it), "category": self.item_cat.get(it), "score": round(sc, 4)}
            for it, sc in top
        ]
        if out:
            out[0]["top_score"] = round(top_score, 4)  # best raw score, before the re-rank; predict() pops it
        return out

    # ---- tie-break on the words the TF-IDF score ignores ----
    def _rr_tokens(self, text):
        """Identifying words plus the ones the TF-IDF vocabulary drops as stop
        words but that separate real SKUs: pack vs single, gift/with bundles,
        can vs bottle, and the proof number ("100" in "New Amsterdam 100")."""
        text = str(text)
        toks = set(rawsig.tokens(text))
        low = APOSTROPHE_RE.sub("", text.lower())
        words = set(TOKEN_RE.findall(low))
        toks |= words & _RR_CLASS
        if _RR_PACK_RE.search(low) or _SIZE_COMPOUND_A_RE.search(text) or _SIZE_COMPOUND_B_RE.search(text) or _SIZE_COMPOUND_C_RE.search(text):
            toks.add("pack")
        for n in PROOF_ABV_RE.findall(text):
            toks.add(n)
        return toks

    def _rerank(self, raw_name, scored):
        head = [x for x in scored[:RERANK_N] if x[1] >= scored[0][1] - RERANK_DELTA]
        if len(head) < 2:
            return scored
        rt = self._rr_tokens(raw_name)
        rseq = " ".join(rawsig.tokens(raw_name))
        liquid = bool(_SIZE_STANDALONE_RE.search(raw_name) or _PACK_STANDALONE_RE.search(raw_name))
        out = []
        for it, sc in head:
            it_t = self._item_rr.get(it)
            if it_t is None:
                it_t = self._item_rr[it] = (self._rr_tokens(it), " ".join(rawsig.tokens(it)))
            u = len(rt | it_t[0]) or 1
            adj = sc + RERANK_W * len(rt & it_t[0]) / u
            if SEQ_W > 0:
                adj += SEQ_W * fuzz.ratio(rseq, it_t[1]) / 100.0
            if liquid and self.item_cat.get(it) == MISC_CATEGORY:
                adj -= MISC_SIZE_PENALTY
            if rseq == it_t[1] and rt == it_t[0]:
                adj += EXACT_W
            out.append((it, sc, adj))
        out.sort(key=lambda x: -x[2])
        return [(it, sc) for it, sc, _ in out] + scored[len(head):]

    def predict_unfiltered(self, raw_name, topk=5):
        """Ranked candidates by raw graph weight alone, skipping every hard
        structural rule (accessory, brand lock, brand anchor, ambiguity
        gate) that predict() applies before committing to an answer.

        For reviewing a raw name predict() refused to tag at all (score 0,
        no candidates) -- most of those are genuinely nothing this catalog
        carries (raw_term_weight ends up empty because nothing in the text
        matches ANY catalog vocabulary at all, e.g. "GE Light Bulbs..."),
        and this returns [] for those too, same as predict() -- but some
        were blocked by a hard rule despite real, if weak, overlap existing
        underneath it. This shows that underlying overlap for a human to
        judge, rather than the app's own "not confident enough" refusal.
        Not used anywhere in the live tagging path -- reporting only.
        """
        s = self._raw_dot(raw_name)
        dot, raw_norm = s["dot"], s["raw_norm"]
        scored = [(it, min(1.0, d / (raw_norm * self.item_norm[it]))) for it, d in dot.items()]
        scored.sort(key=lambda x: -x[1])
        return [
            {"item": it, "brand": self.item_brand.get(it), "category": self.item_cat.get(it), "score": round(sc, 4)}
            for it, sc in scored[:topk]
        ]

    def predict_row(self, raw_name, topk=3):
        candidates = self.predict(raw_name, topk=topk)
        pred = candidates[0] if candidates else {"item": None, "brand": None, "category": None, "score": 0.0}
        item_size, pack_count, unit = self.parse_size(raw_name)
        return {
            "raw": raw_name,
            "pred": pred,
            "candidates": candidates,
            "matched_words": self.matched_words_for(raw_name),
            "item_size": item_size,
            "pack_count": pack_count,
            "unit": unit,
        }
