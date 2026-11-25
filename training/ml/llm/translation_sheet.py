"""Translation check sheet: English sentence next to its on-device ML Kit translation, from app records
(docs/bench/*.json*), up to 30 per language -> docs/TRANSLATION_CHECK.md. A fluent speaker marks OK / awkward / wrong.

  python ml/llm/translation_sheet.py
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = re.compile(r"(?<=[.!?।])\s+")


def main():
    pairs = {"hi": [], "ta": []}
    for p in sorted((REPO / "docs/bench").glob("*.json*")):
        if "before" in p.name:  # archived runs from before a fix
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang, en, loc = r.get("lang"), r.get("advice_en", ""), r.get("advice_local", "")
            if lang not in pairs or not loc:
                continue
            e, l = [s for s in SPLIT.split(en) if s.strip()], [s for s in SPLIT.split(loc) if s.strip()]
            if len(e) == len(l):  # one translation per sentence, so the split lines up
                pairs[lang] += [(r.get("card", ""), a, b) for a, b in zip(e, l) if (a, b) not in {x[1:] for x in pairs[lang]}]
    out = ["# Translation check", "", "Mark each line **OK**, **awkward** or **wrong** (wrong = a farmer would misunderstand). "
           "Pest and crop names come from the glossary, not the translator; check them too.", ""]
    for lang, rows in pairs.items():
        out += [f"## {'Hindi' if lang == 'hi' else 'Tamil'} ({min(len(rows), 30)} sentences)", "",
                "| # | card | English | translation | mark |", "|---|---|---|---|---|"]
        out += [f"| {i + 1} | {c} | {a} | {b} |  |" for i, (c, a, b) in enumerate(rows[:30])] + [""]
    (REPO / "docs/TRANSLATION_CHECK.md").write_text("\n".join(out), encoding="utf-8")
    print({k: len(v) for k, v in pairs.items()})


if __name__ == "__main__":
    main()
