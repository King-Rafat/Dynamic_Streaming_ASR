"""Sanity checks for the CS-WER implementation.

Run with:  python tests/test_cs_wer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cs_wer"))
from script_morph_wer import ScriptMorphWER  # noqa: E402

s = ScriptMorphWER()

CASES = [
    # (name, reference, hypothesis, expect_root_ok, expect_morph_ok, expect_switch_ok)
    ("exact match",
     "আমি collegeটি যাচ্ছি", "আমি collegeটি যাচ্ছি", True, True, True),
    ("root right, suffix wrong",
     "আমি collegeটি যাচ্ছি", "আমি collegeতে যাচ্ছি", True, False, True),
    ("transliterated back to Bengali",
     "আমি collegeটি যাচ্ছি", "আমি কলেজটি যাচ্ছি", False, False, False),
    ("code-switch deleted",
     "আমি collegeটি যাচ্ছি", "আমি যাচ্ছি", False, False, False),
    ("preceding word wrong breaks the switch bigram only",
     "আমি collegeটি যাচ্ছি", "তুমি collegeটি যাচ্ছি", True, True, False),
    ("#E tag is stripped",
     "আমি collegeটি যাচ্ছি", "আমি College#Eটি যাচ্ছি", True, True, True),
    ("suffix as a separate token",
     "আমি college টি যাচ্ছি", "আমি college টি যাচ্ছি", True, True, True),
]

failed = 0
for name, ref, hyp, want_root, want_morph, want_switch in CASES:
    total, m, r, sw = s.compute_single(ref, hyp)
    ok = (bool(r) == want_root) and (bool(m) == want_morph) and (bool(sw) == want_switch)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        failed += 1
        print(f"       total={total} root={r} morph={m} switch={sw}")

# batch behaviour on a corpus that DOES contain code-switches
batch = s.compute_batch([c[1] for c in CASES], [c[2] for c in CASES])
if batch["Total CS Instances"] != len(CASES):
    print(f"[FAIL] batch total = {batch['Total CS Instances']}, expected {len(CASES)}")
    failed += 1
else:
    print("[PASS] batch totals match the per-sentence counts")

print(f"\n{len(CASES) + 1 - failed}/{len(CASES) + 1} passed")
sys.exit(1 if failed else 0)
