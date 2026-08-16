#!/usr/bin/env python3
"""Script-Anchored Loanword Injection.

Turns a monolingual Bangla corpus into a synthetic intra-sentential
code-switched corpus, without touching the audio.

The idea in one line: for a Bangla token w = root(w) + suff(w), if root(w) has
an English equivalent r_en in the lexicon M, rewrite the token as

    w_cs = r_en + suff(w)

The English root stays in Latin script, the Bangla suffix stays in Bengali
script. The script boundary IS the morpheme boundary, so no explicit <en>/<bn>
tag is needed. Syntax is preserved and the audio still matches, which is exactly
where naive "splice in an English clip" augmentation falls apart.

Usage
-----
    python inject_loanwords.py \
        --lexicon ../data/loanwords.tsv \
        --input   train_bn.txt \
        --output  train_cs.txt \
        --rate    0.20

Input format: plain text, one utterance per line, or "<utt_id>\\t<text>".
Output preserves whichever format came in.
"""

from __future__ import annotations

import argparse
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

BENGALI_RE = re.compile(r"[ঀ-৿]")


def load_lexicon(path: Path) -> dict[str, str]:
    """Return {bengali_root: english_root}, longest roots first at match time."""
    lex: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        bn, en = parts[0].strip(), parts[1].strip()
        if bn == "বাংলা_রুট":  # header row
            continue
        if bn and en:
            lex[unicodedata.normalize("NFC", bn)] = en
    return lex


def build_matcher(lex: dict[str, str]) -> list[tuple[str, str]]:
    """Longest-first so 'বিশ্ববিদ্যালয়' wins over any shorter prefix."""
    return sorted(lex.items(), key=lambda kv: -len(kv[0]))


def decompose(token: str, matcher: list[tuple[str, str]]) -> tuple[str, str] | None:
    """If token starts with a known Bangla root, return (english_root, suffix)."""
    tok = unicodedata.normalize("NFC", token)
    for bn_root, en_root in matcher:
        if tok.startswith(bn_root):
            return en_root, tok[len(bn_root):]
    return None


def inject_line(
    text: str,
    matcher: list[tuple[str, str]],
    rate: float,
    rng: random.Random,
    stats: Counter,
) -> str:
    toks = text.split()
    out: list[str] = []
    switched = False
    for tok in toks:
        # strip trailing punctuation, rebuild after
        m = re.match(r"^(.*?)([^\wঀ-৿]*)$", tok, flags=re.UNICODE)
        core, tail = (m.group(1), m.group(2)) if m else (tok, "")
        hit = decompose(core, matcher) if BENGALI_RE.search(core) else None
        if hit and rng.random() < rate:
            en_root, suffix = hit
            out.append(en_root + suffix + tail)
            stats["tokens_switched"] += 1
            stats[f"root:{en_root}"] += 1
            switched = True
        else:
            out.append(tok)
    if switched:
        stats["lines_switched"] += 1
    return " ".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Script-Anchored Loanword Injection")
    p.add_argument("--lexicon", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate", type=float, default=0.20,
                   help="probability a matched token is switched (default 0.20)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--report", type=Path, default=None,
                   help="optional path for a coverage report")
    a = p.parse_args()

    rng = random.Random(a.seed)
    matcher = build_matcher(load_lexicon(a.lexicon))
    if not matcher:
        raise SystemExit(f"no lexicon entries loaded from {a.lexicon}")

    stats: Counter = Counter()
    n_lines = 0
    with a.input.open(encoding="utf-8") as fin, a.output.open("w", encoding="utf-8") as fout:
        for ln in fin:
            ln = ln.rstrip("\n")
            if not ln.strip():
                fout.write("\n")
                continue
            n_lines += 1
            if "\t" in ln:
                uid, text = ln.split("\t", 1)
                fout.write(f"{uid}\t{inject_line(text, matcher, a.rate, rng, stats)}\n")
            else:
                fout.write(inject_line(ln, matcher, a.rate, rng, stats) + "\n")

    pct = 100 * stats["lines_switched"] / n_lines if n_lines else 0
    print(f"lexicon entries   {len(matcher)}")
    print(f"lines in          {n_lines}")
    print(f"lines with CS     {stats['lines_switched']} ({pct:.1f}%)")
    print(f"tokens switched   {stats['tokens_switched']}")

    if a.report:
        roots = sorted(
            ((k[5:], v) for k, v in stats.items() if k.startswith("root:")),
            key=lambda kv: -kv[1],
        )
        with a.report.open("w", encoding="utf-8") as f:
            f.write("english_root\tcount\n")
            for r, c in roots:
                f.write(f"{r}\t{c}\n")
        print(f"report written    {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
