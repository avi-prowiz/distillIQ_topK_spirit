"""
Spelling clean-up applied to raw item names before they are matched, so that the same product written in different
encodings or with / without accents is one product ("1800 Tequila Añejo", "1800 Tequila Anejo",
"1800 Tequila AÃ±ejo", "1800 Tequila Aï¿½ejo" and "1800 Tequila A�ejo" all become "1800 Tequila Anejo").

The raw data carries several kinds of damage:
  * UTF-8 read as Windows-1252   "AÃ±ejo", "â€™"
  * UTF-8 read as GBK            "1800庐" (®), "Jack Daniel鈥檚" (’s), "Espad帽" ... (ñ), "caf茅" (é)
  * accents lost on the way      "Aï¿½ejo", "A�ejo" (the letter is gone; it is recovered from the known words)
  * plain accents / curly quotes "Añejo", "Patrón", "Daniel’s", "•"
"""
import re
import unicodedata

_CJK = re.compile(r"[一-鿿]+")
_REPLACEMENT_WORD = re.compile(r"[A-Za-z0-9�]*�[A-Za-z0-9�]*")
_QUOTES = {ord(c): "'" for c in "‘’‛ʼ´`"}
_QUOTES.update({ord(c): '"' for c in "“”"})
_GUESS = ["n", "i", "o", "e", "a", "u", "y", ""]       # letters most often lost: n (ñ), i (í), o (ó), e (é), a (á), u (ú)


def _cp1252_mojibake(s):
    try:
        fixed = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return s if "�" in fixed else fixed


def _gbk_mojibake(s):
    """Runs of Chinese characters that are really UTF-8 bytes shown as GBK ("庐" -> "®", "帽" -> "ñ")."""
    def run(m):
        text, out, i = m.group(0), [], 0
        while i < len(text):
            for n in (4, 3, 2, 1):
                seg = text[i:i + n]
                if len(seg) != n:
                    continue
                try:
                    out.append(seg.encode("gbk").decode("utf-8"))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                i += n
                break
            else:
                out.append(text[i])
                i += 1
        return "".join(out)
    return _CJK.sub(run, s)


def fold_accents(s):
    """Accents off, curly quotes straight, symbols (®, ™, •) dropped, spaces collapsed: Añejo -> Anejo."""
    s = str(s).translate(_QUOTES)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def clean_text(s, vocab=None):
    """Repair the encodings above and fold the result to plain ASCII. `vocab` (a set of lower-case ASCII words)
    lets a letter that was lost ("A�ejo") be restored from the known words ("anejo"); without it the gap is closed."""
    s = "" if s is None else str(s)
    s = _cp1252_mojibake(s)
    s = _gbk_mojibake(s)
    s = s.replace("ï¿½", "�").replace("Ã¯Â¿Â½", "�")
    s = re.sub(r"(?<=[A-Za-z])\?(?=[A-Za-z])", "�", s)       # "A?ejo": a lost letter written as "?"
    if "�" in s:
        def fix(m):
            word = m.group(0)
            if vocab:
                for g in _GUESS:
                    cand = word.replace("�", g)
                    if fold_accents(cand).lower() in vocab:
                        return cand
            return word.replace("�", "")
        s = _REPLACEMENT_WORD.sub(fix, s)
    return fold_accents(s)
