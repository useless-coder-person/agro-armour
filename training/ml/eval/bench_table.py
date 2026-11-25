"""On-device benchmark tables from the app's metrics records (docs/bench/*.jsonl, *.json) -> docs/report/figures/bench_device.md.

Every number comes from a AgroArmourBench record written by the app on the phone (scripts/bench_device.sh,
scripts/airplane_check.sh, scripts/llm_eval_device.sh).

  python ml/eval/bench_table.py
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load():
    recs = []
    for p in sorted((REPO / "docs/bench").glob("*.json*")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("{"):
                try:
                    recs.append((p.name, json.loads(line)))
                except json.JSONDecodeError:
                    pass  # airplane_check strips backslashes; skip a record that no longer parses
    return recs


def med(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return (f"{st.median(xs):.0f}" if max(xs) > 20 else f"{st.median(xs):.2f}") + f" (n={len(xs)}, {min(xs):g}-{max(xs):g})" if xs else "-"


def main():
    recs = load()
    before = [r for _, r in recs if "threads" not in r]  # Phase 1 build: models loaded per photo, Llama on 1 thread
    after = [r for n, r in recs if "threads" in r and not n.startswith("thread_sweep")]
    sweep = [r for n, r in recs if n.startswith("thread_sweep")]

    def col(rs):
        vision = [r for r in rs if "capture_to_result" in r.get("stages_ms", {})]
        adv = [r for r in rs if r.get("ttft_ms") and not r.get("question_en")]
        fol = [r for r in rs if r.get("ttft_ms") and r.get("question_en")]
        s = lambda k: med([r["stages_ms"].get(k) for r in vision])  # noqa: E731
        return {
            "pest": s("pest"), "disease": s("disease"), "lesion": s("lesion"), "c2r": s("capture_to_result"),
            "load": med([r.get("llm_load_ms") for r in rs]),
            "prime": med([r.get("stages_ms", {}).get("prime") for r in rs if r.get("preload")]),
            "ptok": med([r.get("prompt_tokens") for r in adv]), "ttft": med([r.get("ttft_ms") for r in adv]),
            "tps": med([r.get("tok_per_s") for r in adv + fol]),
            "first": med([r.get("first_sentence_ms") for r in rs]),
            "fttft": med([r.get("ttft_ms") for r in fol]),
            "pss": med([r.get("peak_pss_mb") for r in rs if r.get("peak_pss_mb")]),
        }

    b, a = col(before), col(after)
    n_after = sorted({r["threads"] for r in after})
    md = ["Device: Samsung Galaxy M34 5G (SM-M346B), Exynos 1280 (2× Cortex-A78 + 6× Cortex-A55), 5.3 GB RAM, Tier B. "
          "ExecuTorch 1.5.1, XNNPACK, CPU only. Median (n, min-max).", "",
          "**Before** = Phase 1 build: vision models loaded and closed per photo; Llama loaded on the first photo and, "
          "as the logs show, run on 1 thread. **After** = models loaded at app start and kept, all on one threadpool of "
          f"{'/'.join(map(str, n_after)) or '-'} threads, system prompt pre-filled, follow-ups appended to the context.", "",
          "| stage | before | after | target (low-end / mid) |", "|---|---|---|---|",
          f"| pest detector, 640 px | {b['pest']} ms (load + run) | {a['pest']} ms (run) | ≤ 900 / ≤ 350 ms |",
          f"| disease classifier, 224 px | {b['disease']} ms (load + run) | {a['disease']} ms (run) | ≤ 200 / ≤ 80 ms |",
          f"| lesion estimate | {b['lesion']} ms | {a['lesion']} ms | - |",
          f"| capture → structured result | {b['c2r']} ms | {a['c2r']} ms | ≤ 2000 / ≤ 1000 ms |",
          f"| Llama load (mmap) | {b['load']} ms (on the first photo) | {a['load']} ms (at app start) | ≤ 8000 / ≤ 4000 ms |",
          f"| system prompt pre-fill (idle time) | - | {a['prime']} ms | - |",
          f"| advisory prompt tokens prefilled after the photo | {b['ptok']} | {a['ptok']} | - |",
          f"| advisory time to first token | {b['ttft']} ms | {a['ttft']} ms | ≤ 6000 / ≤ 3000 ms |",
          f"| decode speed | {b['tps']} tok/s | {a['tps']} tok/s | ≥ 6 / ≥ 15 tok/s |",
          f"| photo → first advice sentence on screen | {b['first']} ms | {a['first']} ms | - |",
          f"| follow-up time to first token | {b['fttft']} ms | {a['fttft']} ms | - |",
          f"| peak app PSS during generation | {b['pss']} MB | {a['pss']} MB | ≤ 2300 MB |"]
    if sweep:
        md += ["", "Thread sweep (`scripts/thread_sweep.sh`; per N: cold start, one photo, two fixed advisories):", "",
               "| threads | Llama load ms | prime ms | advisory TTFT ms | prefill tok/s | decode tok/s | pest detector ms | photo → first sentence ms |",
               "|---|---|---|---|---|---|---|---|"]
        for t in sorted({r["threads"] for r in sweep}):
            c = col([r for r in sweep if r["threads"] == t])
            pre = med([r["prompt_tokens"] * 1000 / r["ttft_ms"] for r in sweep
                       if r["threads"] == t and r.get("ttft_ms") and not r.get("question_en")])
            md.append(f"| {t} | {c['load']} | {c['prime']} | {c['ttft']} | {pre} | {c['tps']} | {c['pest']} | {c['first']} |")
    md += ["", "Sources: " + ", ".join(sorted({n for n, _ in recs}))]
    out = REPO / "docs/report/figures/bench_device.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
