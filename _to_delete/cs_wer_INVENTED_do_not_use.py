#!/usr/bin/env python3
"""Fine-grained CS-WER for Bangla-English intra-sentential code-switching.

Standard WER treats every error identically. For agglutinative code-switching
that hides the thing we actually care about: whether the model found the
language switch, whether it got the English root, and whether it attached the
Bangla suffix correctly.

CS-WER decomposes error into three rates, reported as E_switch / E_root / E_morph.

    E_switch  Switch-Point Error.
              Over reference switch points, the bigram
              (last Matrix-Language word, first Embedded-Language word)
              must be reproduced exactly in the hypothesis.

    E_root    Loanword Root Error.
              Over reference code-switched tokens, the Latin-script English
              root must appear, ignoring whatever suffix follows.

    E_morph   Morphological Boundary Error.
              Over reference code-switched tokens, the root AND the Bangla
              suffix must both be correct, i.e. the full surface form matches.

All three are error rates in [0, 1]; lower is better.

Usage
-----
    python cs_wer.py --ref refs.txt --hyp hyps.txt [--lexicon loanwords.tsv]
    python cs_wer.py --ref refs.txt --hyp hyps.txt --per-utt
    python cs_wer.py --self-test

ref/hyp files are parallel line-aligned plain text, one utterance per line.
Optionally each line may be "<utt_id>\\t<text>"; ids are matched when present in
both files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Script detection
# --------------------------------------------------------------------------

BENGALI_RANGE = (0x0980, 0x09FF)
LATIN_RE = re.compile(r"[A-Za-z]")
BENGALI_RE = re.compile(r"[ঀ-৿]")
PUNCT_RE = re.compile(r"[^\wঀ-৿]+", re.UNICODE)


def is_bengali_char(ch: str) -> bool:
    return BENGALI_RANGE[0] <= ord(ch) <= BENGALI_RANGE[1]


def has_latin(tok: str) -> bool:
    return bool(LATIN_RE.search(tok))


def has_bengali(tok: str) -> bool:
    return bool(BENGALI_RE.search(tok))


def is_code_switched(tok: str) -> bool:
    """A code-switched token carries a Latin-script English root.

    This covers both the bare loanword ("doctor") and the script-anchored
    agglutinated form ("college" + Bengali suffix -> "collegeti" written with a
    Bengali suffix).
    """
    return has_latin(tok)


def is_agglutinated(tok: str) -> bool:
    """Latin root with a Bengali suffix glued on, e.g. college + Bengali suffix."""
    return has_latin(tok) and has_bengali(tok)


def split_root_suffix(tok: str) -> tuple[str, str]:
    """Split a script-anchored token into (latin_root, bengali_suffix).

    The script boundary is the morpheme boundary by construction of
    Script-Anchored Loanword Injection, so we split at the first Bengali
    character following the Latin run.
    """
    for i, ch in enumerate(tok):
        if is_bengali_char(ch):
            return tok[:i], tok[i:]
    return tok, ""


# --------------------------------------------------------------------------
# Normalisation and tokenisation
# --------------------------------------------------------------------------


def normalize(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        # keep intra-token script transitions intact; only kill separators
        text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str, **kw) -> list[str]:
    return normalize(text, **kw).split()


# --------------------------------------------------------------------------
# Alignment (Levenshtein with backtrace)
# --------------------------------------------------------------------------

MATCH, SUB, INS, DEL = "match", "sub", "ins", "del"


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int | None, int | None]]:
    """Word-level Levenshtein alignment.

    Returns a list of (op, ref_index, hyp_index). Index is None where the op
    consumes nothing on that side.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append((MATCH if cost == 0 else SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((DEL, i - 1, None))
            i -= 1
            continue
        ops.append((INS, None, j - 1))
        j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# Metric accumulators
# --------------------------------------------------------------------------


@dataclass
class Counts:
    # standard WER
    ref_words: int = 0
    sub: int = 0
    ins: int = 0
    dele: int = 0
    # CS-WER numerators / denominators
    switch_total: int = 0
    switch_err: int = 0
    root_total: int = 0
    root_err: int = 0
    morph_total: int = 0
    morph_err: int = 0
    # char error rate
    ref_chars: int = 0
    char_edits: int = 0

    def merge(self, other: "Counts") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @staticmethod
    def _rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def wer(self) -> float:
        return self._rate(self.sub + self.ins + self.dele, self.ref_words)

    @property
    def cer(self) -> float:
        return self._rate(self.char_edits, self.ref_chars)

    @property
    def e_switch(self) -> float:
        return self._rate(self.switch_err, self.switch_total)

    @property
    def e_root(self) -> float:
        return self._rate(self.root_err, self.root_total)

    @property
    def e_morph(self) -> float:
        return self._rate(self.morph_err, self.morph_total)

    def as_dict(self) -> dict:
        return {
            "cer": round(self.cer * 100, 2),
            "wer": round(self.wer * 100, 2),
            "e_switch": round(self.e_switch, 3),
            "e_root": round(self.e_root, 3),
            "e_morph": round(self.e_morph, 3),
            "cs_wer": f"{self.e_switch:.2f}/{self.e_root:.2f}/{self.e_morph:.2f}",
            "ref_words": self.ref_words,
            "switch_points": self.switch_total,
            "cs_tokens": self.root_total,
        }


def _char_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_utterance(
    ref_text: str,
    hyp_text: str,
    lexicon: set[str] | None = None,
) -> Counts:
    ref = tokenize(ref_text)
    hyp = tokenize(hyp_text)
    c = Counts()

    # ---- standard WER / CER ------------------------------------------------
    ops = align(ref, hyp)
    c.ref_words = len(ref)
    for op, _, _ in ops:
        if op == SUB:
            c.sub += 1
        elif op == INS:
            c.ins += 1
        elif op == DEL:
            c.dele += 1
    ref_join, hyp_join = " ".join(ref), " ".join(hyp)
    c.ref_chars = len(ref_join)
    c.char_edits = _char_edit_distance(ref_join, hyp_join)

    # map ref index -> aligned hyp token (None if deleted)
    r2h: dict[int, str | None] = {}
    for op, ri, hi in ops:
        if ri is not None:
            r2h[ri] = hyp[hi] if hi is not None else None

    hyp_set = Counter(hyp)
    hyp_roots = Counter(split_root_suffix(t)[0] for t in hyp if is_code_switched(t))

    for ri, rtok in enumerate(ref):
        if not is_code_switched(rtok):
            continue
        if lexicon is not None:
            root_ref = split_root_suffix(rtok)[0]
            if root_ref not in lexicon:
                continue

        root_ref, suff_ref = split_root_suffix(rtok)

        # ---- E_root: is the English root present at all? -------------------
        c.root_total += 1
        aligned = r2h.get(ri)
        root_ok = False
        if aligned is not None and is_code_switched(aligned):
            root_ok = split_root_suffix(aligned)[0] == root_ref
        if not root_ok:
            # tolerate alignment slippage: root present anywhere in hyp counts
            root_ok = hyp_roots[root_ref] > 0
        if not root_ok:
            c.root_err += 1

        # ---- E_morph: root AND suffix both correct -------------------------
        c.morph_total += 1
        morph_ok = aligned == rtok or hyp_set[rtok] > 0
        if not morph_ok:
            c.morph_err += 1

        # ---- E_switch: matrix->embedded bigram preserved -------------------
        if ri > 0 and not is_code_switched(ref[ri - 1]):
            c.switch_total += 1
            prev_ok = r2h.get(ri - 1) == ref[ri - 1]
            bigram = f"{ref[ri - 1]} {rtok}"
            switch_ok = prev_ok and root_ok and bigram in hyp_join
            if not switch_ok:
                c.switch_err += 1

    return c


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def read_lines(path: Path) -> dict[str, str] | list[str]:
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if all("\t" in ln for ln in lines) and lines:
        return {ln.split("\t", 1)[0]: ln.split("\t", 1)[1] for ln in lines}
    return lines


def load_lexicon(path: Path) -> set[str]:
    roots: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 2:
            roots.add(normalize(parts[1]))
    return roots


def pair_up(ref, hyp) -> list[tuple[str, str, str]]:
    if isinstance(ref, dict) and isinstance(hyp, dict):
        keys = [k for k in ref if k in hyp]
        missing = len(ref) - len(keys)
        if missing:
            print(f"warning: {missing} reference ids missing from hypothesis", file=sys.stderr)
        return [(k, ref[k], hyp[k]) for k in keys]
    ref_l = list(ref.values()) if isinstance(ref, dict) else ref
    hyp_l = list(hyp.values()) if isinstance(hyp, dict) else hyp
    if len(ref_l) != len(hyp_l):
        raise SystemExit(f"line count mismatch: ref={len(ref_l)} hyp={len(hyp_l)}")
    return [(str(i), r, h) for i, (r, h) in enumerate(zip(ref_l, hyp_l))]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

SELF_TEST = [
    # (ref, hyp, expect_root_err, expect_morph_err)
    ("আমি collegeটি যাচ্ছি", "আমি collegeটি যাচ্ছি", 0, 0),      # perfect
    ("আমি collegeটি যাচ্ছি", "আমি collegeতে যাচ্ছি", 0, 1),      # root ok, suffix wrong
    ("আমি collegeটি যাচ্ছি", "আমি কলেজটি যাচ্ছি", 1, 1),        # transliterated away
    ("আমি collegeটি যাচ্ছি", "আমি যাচ্ছি", 1, 1),                # deleted
]


def self_test() -> int:
    failures = 0
    for ref, hyp, want_root, want_morph in SELF_TEST:
        c = score_utterance(ref, hyp)
        ok = c.root_err == want_root and c.morph_err == want_morph
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] ref={ref!r} hyp={hyp!r} "
              f"root_err={c.root_err}(want {want_root}) "
              f"morph_err={c.morph_err}(want {want_morph})")
    print(f"\n{len(SELF_TEST) - failures}/{len(SELF_TEST)} passed")
    return 1 if failures else 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-grained CS-WER for Bangla-English CS ASR")
    p.add_argument("--ref", type=Path, help="reference transcript file")
    p.add_argument("--hyp", type=Path, help="hypothesis transcript file")
    p.add_argument("--lexicon", type=Path, default=None,
                   help="loanwords.tsv; restricts CS scoring to lexicon roots")
    p.add_argument("--per-utt", action="store_true", help="print per-utterance scores")
    p.add_argument("--json", action="store_true", help="emit JSON only")
    p.add_argument("--self-test", action="store_true", help="run built-in sanity checks")
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.ref or not a.hyp:
        p.error("--ref and --hyp are required (or use --self-test)")

    lexicon = load_lexicon(a.lexicon) if a.lexicon else None
    pairs = pair_up(read_lines(a.ref), read_lines(a.hyp))

    total = Counts()
    for utt_id, r, h in pairs:
        c = score_utterance(r, h, lexicon)
        total.merge(c)
        if a.per_utt:
            print(f"{utt_id}\t{json.dumps(c.as_dict(), ensure_ascii=False)}")

    d = total.as_dict()
    if a.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        print(f"utterances   {len(pairs)}")
        print(f"ref words    {d['ref_words']}")
        print(f"CS tokens    {d['cs_tokens']}")
        print(f"switch pts   {d['switch_points']}")
        print(f"CER          {d['cer']:.2f}")
        print(f"WER          {d['wer']:.2f}")
        print(f"CS-WER       {d['cs_wer']}   (E_switch / E_root / E_morph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
