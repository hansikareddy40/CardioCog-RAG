"""Phase 10, step 3: test the evidence assistant.

  retrieval     for each test question, is a passage from an expected source
                among the top results? Compared for keyword-only, meaning-only
                and hybrid search.
  guardrails    are requests for individual medical advice flagged, and are
                ordinary questions left alone?
  out of scope  does the assistant say "not found" for unrelated questions?
  grounding     in generative mode, does every answer cite retrieved passages,
                and how often did it have to fall back to quoting?

The test questions are in rag/eval_questions.json. They were written by hand
from the topics of the sources, so this is a check of the mechanics, not an
independent benchmark.

Output  reports/phase10_rag_results.json
Usage:  python scripts/rag_eval.py [--llm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import rag

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate generative answers (needs the local model)")
    args = ap.parse_args()
    qs = json.loads((REPO / "rag" / "eval_questions.json").read_text())
    a = rag.Assistant()
    lib = a.library
    res = {"library": {"passages": len(lib.passages), "sources": len({p["source"] for p in lib.passages})}, "n_questions": {k: len(v) for k, v in qs.items()}}

    res["retrieval"] = {}
    for mode in ["bm25", "dense", "hybrid"]:
        ranks = []
        for item in qs["retrieval"]:
            hits = lib.search(item["q"], 10, mode)
            rank = next((i + 1 for i, h in enumerate(hits) if h["source"] in item["expect"]), None)
            ranks.append(rank)
        res["retrieval"][mode] = {f"hit_at_{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in (1, 3, 5)}
        res["retrieval"][mode]["mrr"] = float(np.mean([0 if r is None else 1 / r for r in ranks]))

    res["guardrails"] = {
        "advice_flagged": float(np.mean([bool(rag.ADVICE.search(q)) for q in qs["advice"]])),
        "general_wrongly_flagged": float(np.mean([bool(rag.ADVICE.search(q)) for q in qs["general_not_advice"]])),
        "out_of_scope_refused": float(np.mean([a.ask(q, use_llm=False)["mode"] == "not_found" for q in qs["out_of_scope"]])),
        "in_scope_wrongly_refused": float(np.mean([a.ask(i["q"], use_llm=False)["mode"] == "not_found" for i in qs["retrieval"]])),
    }
    sims_in = [max(h["similarity"] for h in lib.search(i["q"], 5)) for i in qs["retrieval"]]
    sims_out = [max(h["similarity"] for h in lib.search(q, 5)) for q in qs["out_of_scope"]]
    res["similarity"] = {"threshold": rag.MIN_SIMILARITY, "in_scope_min": float(min(sims_in)), "in_scope_median": float(np.median(sims_in)),
                         "out_of_scope_max": float(max(sims_out))}

    if args.llm and a.generator.load():
        modes, cited_expected, n_sent = [], [], []
        for item in qs["retrieval"]:
            r = a.ask(item["q"])
            modes.append(r["mode"])
            cited_expected.append(any(s["short"] for s in r["sources"]) and any(
                p["source"] in item["expect"] for p in lib.passages if p["short"] in {s["short"] for s in r["sources"]}))
            n_sent.append(len(r["answer"].split(". ")))
        advice_modes = [a.ask(q)["mode"] for q in qs["advice"]]
        res["generation"] = {"model": rag.LLM_MODEL,
                             "answered_generatively": float(np.mean([m == "generative" for m in modes])),
                             "fell_back_to_quoting": float(np.mean([m == "extractive" for m in modes])),
                             "said_not_found": float(np.mean([m == "not_found" for m in modes])),
                             "cited_an_expected_source": float(np.mean(cited_expected)),
                             "advice_questions_answered_generatively": float(np.mean([m == "generative" for m in advice_modes]))}
    (REPO / "reports" / "phase10_rag_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
