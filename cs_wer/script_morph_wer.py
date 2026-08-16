"""Fine-Grained CS-WER for Bangla-English intra-sentential code-switching.

Reference implementation for:

    Dynamic Block-Online Streaming ASR for Low-Resource Agglutinative
    Code-Switching Speech with Morphology-Aware Evaluation
    Interspeech 2026

CS-WER decomposes error over code-switch instances into three components,
reported as the tuple E_switch / E_root / E_morph. See the README for the
definitions, the scoring unit, and input requirements.

Usage
-----
As a library:

    from script_morph_wer import ScriptMorphWER
    scorer = ScriptMorphWER()
    results = scorer.compute_batch(references, hypotheses)

From the command line, over an Excel or CSV file with `sentence` (reference)
and `outputs` (hypothesis) columns:

    python script_morph_wer.py --file results.xlsx
    python script_morph_wer.py --file a.xlsx b.xlsx --ref-col sentence --hyp-col outputs

Or over two parallel plain-text files:

    python script_morph_wer.py --ref refs.txt --hyp hyps.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class ScriptMorphWER:
    def __init__(self):
        # Regex to detect if a word contains English characters (A-Z, a-z)
        self.english_char_pattern = re.compile(r'[a-zA-Z]')
        self.valid_suffixes = [
            # --- LEVEL 1: Complex/Compound Suffixes (4+ chars) ---
            "গুলো", "গুলি", "গুলা",        # Plurals
            "গুলোর", "গুলির", "গুলার",     # Plural Possessive (of the files)
            "গুলোতে", "গুলিতে", "গুলায়",   # Plural Locative (in the files)
            "গুলোকে", "গুলিকে", "গুলাকে",  # Plural Objective (to the files)
            "দেরকে", "দের",               # Human Plural (Doctors)
            "সমূহ", "সমূহের", "সমূহে",     # Formal Plural

            # --- LEVEL 2: Determiners + Case/Emphasis (3 chars) ---
            "টাতে", "টায়", "টার",         # The + Locative/Possessive (Table-ta-te)
            "টিতে", "টির",                # The + Locative/Possessive
            "টাই", "টায়",                # The + Emphatic (Mobile-tai - That specific mobile)
            "খানা", "খানি", "খান",        # Measure words
            "টুকু", "টুকুন", "টুক",        # Amount (Water-tuku)
            "গুলোও", "গুলিও",             # Plural + Also (Files-guloo)

            # --- LEVEL 3: Adverbial & Derivational ---
            "ভাবে",                       # -ly (Automatic-bhabe)
            "গত", "ময়", "হীন", "জনিত",    # -related (Server-goto)
            "পনা", "গিরি",                # Noun modifiers (Doctor-giri)
            "ওয়ালা", "ওয়ালি",             # Holder (Rickshaw-wala)
            "কারী",                       # Doer (Support-kari)

            # --- LEVEL 4: Standard Grammatical Suffixes (2 chars) ---
            "রা", "এরা",                  # Plural Subject
            "ের", "র", "য়ের", "কার",      # Possessive (Of) - 'Internet-er'
            "কে", "রে",                   # To/For
            "তে", "এ", "য়",               # In/At (Locative) - 'Office-e'
            "সহ",                        # With (Data-shoho)

            # --- LEVEL 5: Basic Determiners (2 chars) ---
            "টা", "টি",

            # --- LEVEL 6: Single Letter Clitics (1 char) ---
            "ই",                          # Emphatic (Mobile-i -> The mobile itself)
            "ও"                           # Inclusive (Laptop-o -> Laptop also)
        ]

    def is_english(self, token):
        return bool(self.english_char_pattern.search(token))

    def normalize(self, token):
        token = token.replace("#E", "")
        token = token.lower().strip()
        return token

    def splitter(self, text):
        parts = re.findall(r'[a-zA-Z]+|[^a-zA-Z]+', text)
        return parts

    def extract_cs_triplets(self, text):
        tokens = text.split()
        triplets = []

        for i, token in enumerate(tokens):
            if self.is_english(token):
                # 1. Get Context
                if i > 0:
                    prev_word = self.normalize(tokens[i-1])
                else:
                    prev_word = "<START>"

                clean_token = self.normalize(token)
                parts = self.splitter(clean_token)

                # 2. Handle Merged Suffixes
                if len(parts) > 1:
                    triplets.append((prev_word, parts[0], parts[1]))
                    continue

                # 3. Handle Separate Suffixes
                if i+1 < len(tokens):
                    if tokens[i+1] in self.valid_suffixes:
                        next_token = self.normalize(tokens[i+1])
                        triplets.append((prev_word, clean_token, next_token))
                    else:
                        triplets.append((prev_word, clean_token, 1))
                else:
                    triplets.append((prev_word, clean_token, "<EOS>"))

        return triplets

    def compute_single(self, reference, hypothesis):
        """
        Computes counts for a SINGLE pair of strings.
        Returns raw counts (total, matches) instead of WER so we can sum them later.
        """
        ref_triplets = self.extract_cs_triplets(reference)
        total_items = len(ref_triplets)

        if total_items == 0:
            return 0, 0, 0, 0  # count, m_match, r_match, s_match

        hyp_triplets = self.extract_cs_triplets(hypothesis)

        # Prepare Data
        ref_morphs = [(t[1], t[2]) for t in ref_triplets]
        hyp_morphs = [(t[1], t[2]) for t in hyp_triplets]

        ref_roots = [t[1] for t in ref_triplets]
        hyp_roots = [t[1] for t in hyp_triplets]

        ref_switches = [(t[0], t[1]) for t in ref_triplets]
        hyp_switches = [(t[0], t[1]) for t in hyp_triplets]

        # Calculate Matches
        morph_matches = 0
        hyp_morphs_copy = list(hyp_morphs)
        for r in ref_morphs:
            if r in hyp_morphs_copy:
                morph_matches += 1
                hyp_morphs_copy.remove(r)

        root_matches = 0
        hyp_roots_copy = list(hyp_roots)
        for r in ref_roots:
            if r in hyp_roots_copy:
                root_matches += 1
                hyp_roots_copy.remove(r)

        switch_matches = 0
        hyp_switches_copy = list(hyp_switches)
        for r in ref_switches:
            if r in hyp_switches_copy:
                switch_matches += 1
                hyp_switches_copy.remove(r)

        return total_items, morph_matches, root_matches, switch_matches

    def compute_batch(self, ref_list, hyp_list):
        """
        Process a LIST of sentences and return global CS-WER.
        """
        global_total = 0
        global_morph_match = 0
        global_root_match = 0
        global_switch_match = 0

        # Iterate through the lists
        for ref, hyp in zip(ref_list, hyp_list):
            total, m_match, r_match, s_match = self.compute_single(ref, hyp)

            global_total += total
            global_morph_match += m_match
            global_root_match += r_match
            global_switch_match += s_match

        # Avoid division by zero if no English words exist in the entire set
        if global_total == 0:
            return 0.0, 0.0, 0.0, 0

        # Calculate Aggregate Metrics
        morph_wer = (global_total - global_morph_match) / global_total
        root_wer = (global_total - global_root_match) / global_total
        switch_wer = (global_total - global_switch_match) / global_total

        return {
            "CS-WER (Morph)": morph_wer,
            "CS-WER (Root)": root_wer,
            "CS-WER (Switch)": switch_wer,
            "Total CS Instances": global_total
        }


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _load_table(path: Path, ref_col: str, hyp_col: str):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("reading tables requires pandas: pip install pandas openpyxl")
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    for col in (ref_col, hyp_col):
        if col not in df.columns:
            raise SystemExit(f"{path}: column {col!r} not found. Columns: {list(df.columns)}")
    df = df[[ref_col, hyp_col]].dropna()
    return df[ref_col].astype(str).tolist(), df[hyp_col].astype(str).tolist()


def _load_text(path: Path):
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-Grained CS-WER (E_switch / E_root / E_morph)")
    p.add_argument("--file", type=Path, nargs="+",
                   help="xlsx/csv file(s) with reference and hypothesis columns")
    p.add_argument("--ref-col", default="sentence", help="reference column (default: sentence)")
    p.add_argument("--hyp-col", default="outputs", help="hypothesis column (default: outputs)")
    p.add_argument("--ref", type=Path, help="reference text file, one utterance per line")
    p.add_argument("--hyp", type=Path, help="hypothesis text file, one utterance per line")
    p.add_argument("--json", action="store_true", help="emit JSON")
    a = p.parse_args(argv)

    scorer = ScriptMorphWER()
    jobs = []

    if a.file:
        for f in a.file:
            jobs.append((f.name, *_load_table(f, a.ref_col, a.hyp_col)))
    elif a.ref and a.hyp:
        refs, hyps = _load_text(a.ref), _load_text(a.hyp)
        if len(refs) != len(hyps):
            raise SystemExit(f"line count mismatch: ref={len(refs)} hyp={len(hyps)}")
        jobs.append((f"{a.ref.name} vs {a.hyp.name}", refs, hyps))
    else:
        p.error("provide --file, or both --ref and --hyp")

    out = {}
    for name, refs, hyps in jobs:
        r = scorer.compute_batch(refs, hyps)
        out[name] = r
        if not a.json:
            print(f"\n{name}")
            print(f"  Total CS instances  {r['Total CS Instances']}")
            print(f"  E_switch            {r['CS-WER (Switch)']:.4f}")
            print(f"  E_root              {r['CS-WER (Root)']:.4f}")
            print(f"  E_morph             {r['CS-WER (Morph)']:.4f}")
            print(f"  reported as         "
                  f"{r['CS-WER (Switch)']:.2f}/{r['CS-WER (Root)']:.2f}/{r['CS-WER (Morph)']:.2f}")

    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
