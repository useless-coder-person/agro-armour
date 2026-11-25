"""Knowledge-base linter. Exit 1 on any error.

Checks:
  * every card matches kb/schema.json; id == "<kind>.<key>"; file name == <key>.json
  * coverage: a card for every advisory group (ml/configs/groups.yaml) and every non-healthy disease label
    (ml/configs/classes_disease.yaml), directly or through "also_keys"
  * chemical entries: source_id exists; active not banned / excluded (kb/banned_pesticides.json, DPPQS list
    31.10.2026); not restricted on the card's crops; dose and active found VERBATIM in the cited source text
    when $AGROARMOUR_SOURCES/<local_text> exists (whitespace/case-insensitive)
  * any number written in free text must also appear in the cited source text (anti-fabrication guard)
  * severity thresholds that are filled need a source_id

  python ml/llm/kb_lint.py
"""
import json
import os
import re
import sys
from pathlib import Path

import jsonschema
import yaml

REPO = Path(__file__).resolve().parents[2]
KB = REPO.parent / "kb"  # app knowledge base at the submission root
SOURCES = Path(os.environ.get("AGROARMOUR_SOURCES", "~/cropdoc/sources")).expanduser()
NUM = re.compile(r"\d+(?:\.\d+)?")


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).replace(" %", "%").replace("@ ", "@").strip()


def required_keys():
    groups = set(yaml.safe_load(open(REPO / "ml/configs/groups.yaml")))
    cfg = yaml.safe_load(open(REPO / "ml/configs/classes_disease.yaml"))
    labels = {v for s in cfg["sources"].values() for v in s["map"].values()}
    return groups | {l for l in labels if not l.endswith("__healthy") and not l.startswith("other__")}


def lint(cards, banned, src_texts):
    errs = []
    schema = json.load(open(KB / "schema.json"))
    for path, c in cards:
        e = lambda m: errs.append(f"{path.name}: {m}")  # noqa: E731
        try:
            jsonschema.validate(c, schema)
        except jsonschema.ValidationError as ex:
            e(f"schema: {ex.message} at {list(ex.absolute_path)}")
            continue
        if c["id"] != f"{c['kind']}.{c['key']}":
            e(f"id {c['id']} != kind.key")
        if path.stem != c["key"]:
            e("file name must equal key")
        sids = {s["id"]: s for s in c["sources"]}
        text = " ".join(norm(src_texts.get(s.get("local_text"), "")) for s in c["sources"])
        crops = {x.lower() for x in c["crops"]}
        if c["severity"]["thresholds"] and any(c["severity"]["thresholds"].values()) and not c["severity"]["source_id"]:
            e("severity thresholds without source_id")
        for lvl, a in c["actions"].items():
            for ch in a["chemical"]:
                if ch["source_id"] not in sids:
                    e(f"{lvl} chemical {ch['active']}: unknown source_id {ch['source_id']}")
                act = ch["active"].lower()
                for b in banned["banned"] + list(banned["project_excluded"]):
                    if re.search(rf"\b{re.escape(b)}\b", act):
                        e(f"{lvl} chemical {ch['active']}: banned/excluded ({b})")
                for r, bad in banned["restricted_crops"].items():
                    if r in act and crops & set(bad):
                        e(f"{lvl} chemical {ch['active']}: restricted on {crops & set(bad)}")
                if text:
                    dose = norm(f"{ch['dose']} {ch['dose_unit']}")
                    if dose not in text and norm(f"{ch['dose']}{ch['dose_unit']}") not in text:
                        e(f"{lvl} chemical {ch['active']}: dose '{dose}' not found verbatim in source text")
                    if norm(act.split()[0]) not in text:
                        e(f"{lvl} chemical {ch['active']}: active not found in source text")
                else:
                    e(f"{lvl} chemical {ch['active']}: no local source text to check the dose against")
        if text:
            free = c["identify"] + [c["damage"]] + c["safety"] + list(c["region_notes"].values()) + \
                list(c["season_notes"].values()) + list(c["stage_notes"].values()) + \
                [t for a in c["actions"].values() for t in a["cultural"] + a["biological"]] + \
                [v for v in c["severity"]["thresholds"].values()]
            for t in free:
                for n in NUM.findall(t):
                    if n not in text:
                        e(f"number {n} in '{t[:60]}' not in source text")
    have = {c["key"] for _, c in cards} | {k for _, c in cards for k in c.get("also_keys", [])}
    for k in sorted(required_keys() - have):
        errs.append(f"coverage: no card for {k}")
    return errs


def main():
    banned = json.load(open(KB / "banned_pesticides.json"))
    banned["banned"] = [b.lower() for b in banned["banned"]]
    cards = [(p, json.load(open(p, encoding="utf-8"))) for p in sorted((KB / "cards").glob("*.json"))]
    src_texts = {}
    for _, c in cards:
        for s in c["sources"]:
            lt = s.get("local_text")
            if lt and lt not in src_texts and (SOURCES / lt).exists():
                src_texts[lt] = (SOURCES / lt).read_text(errors="ignore")
    if not src_texts:
        print(f"note: no source texts under {SOURCES}; dose-verbatim checks skipped")
    errs = lint(cards, banned, src_texts)
    for x in errs:
        print("ERROR", x)
    n_chem = sum(len(a["chemical"]) for _, c in cards for a in c["actions"].values())
    print(f"{len(cards)} cards, {n_chem} chemical entries, {len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
