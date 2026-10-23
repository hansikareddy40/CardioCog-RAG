"""Plain-language report for one estimate: risk, reasons, reliability and evidence.

The zeroth-review design ends in a clinician-readable report that combines the
model's output, its explanation and supporting published evidence. This module
writes that report from the prediction, the SHAP contributions and the
evidence library. The wording is fixed templates filled with the person's
numbers, so the report never states anything the model did not produce.
Evidence is quoted from the library (no generated text), with its source.
"""

from __future__ import annotations

import re

import numpy as np

from inference import LABELS

# Which published evidence to look up for each kind of information, and a word
# the quoted passage must contain so that it is on topic.
EVIDENCE_QUESTION = {
    "Clinical stage (CDR-SB, diagnosis)": ("progression from mild cognitive impairment to dementia", r"progress"),
    "Cognitive tests": ("progression from mild cognitive impairment to dementia", r"progress"),
    "Cardiovascular / medical": ("hypertension and blood pressure as a risk factor for dementia", r"hypertension|blood pressure"),
    "Genetics / APOE": ("APOE e4 allele and the risk of Alzheimer's disease", r"APOE"),
    "Demographics": ("modifiable risk factors for dementia across the life course", r"risk factor"),
}
GROUP_WORDS = {
    "Clinical stage (CDR-SB, diagnosis)": "the clinician's rating of daily function and cognitive status",
    "Cognitive tests": "the memory and thinking test results",
    "Cardiovascular / medical": "heart and blood-vessel history",
    "Genetics / APOE": "the APOE gene result and family history",
    "Demographics": "age and other personal details",
}


def pct(p: float) -> str:
    return "under 1%" if p < 0.01 else f"{round(p * 100)}%"


def describe_item(name: str, value: float) -> str:
    """Turn one model input and its value into a short phrase."""
    label = LABELS.get(name, name)
    label = label[:1].lower() + label[1:]
    if name.startswith("z_"):
        level = ("well below" if value <= -2 else "below" if value <= -1 else "about as" if value < 1 else "above")
        return f"{label} {level} expected for age, sex and education"
    if name == "is_mci":
        return "a current status of MCI" if value == 1 else "not having a diagnosis of MCI"
    if name == "is_impaired_not_mci":
        return "a current status of impairment without MCI" if value == 1 else "not having a diagnosis of cognitive impairment"
    if name == "CDRSUM":
        return f"a CDR Sum of Boxes of {value:g}"
    if name == "age":
        return f"age {value:g}"
    if name == "apoe_e4_count":
        return f"{int(value)} cop{'y' if value == 1 else 'ies'} of APOE e4"
    if value in (0.0, 1.0) and name.endswith(("_ever", "female", "hispanic", "lives_alone", "family_history")) or name.startswith(("med_", "race_")):
        return f"{label}: {'yes' if value == 1 else 'no'}"
    return f"{label} of {value:g}"


def first_sentences(text: str, n: int = 2, limit: int = 320) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:n])
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + " ..."


def build_report(res: dict, checks: list, similar: dict | None, used: list, missing: list, library=None, group_of: dict | None = None) -> dict:
    f, sv = res["features"], res["shap_by_feature"]
    items = [(k, v) for k, v in sv.items() if not np.isnan(f[k]) and abs(v) > 0.05]
    raising = sorted([kv for kv in items if kv[1] > 0], key=lambda kv: -kv[1])[:3]
    lowering = sorted([kv for kv in items if kv[1] < 0], key=lambda kv: kv[1])[:2]

    summary = f"The estimated chance of a dementia diagnosis within the next three years is {pct(res['risk'])}."
    if similar:
        who = f"aged {similar['age_group'].lower()}" + (f" with a status of {similar['status']}" if similar["status"] else "")
        summary += (f" For comparison, {pct(similar['rate'])} of {similar['n']:,} research participants {who} "
                    f"were diagnosed with dementia within three years.")

    reasons = []
    if raising:
        reasons.append("The estimate was raised mainly by " + "; ".join(describe_item(k, f[k]) for k, _ in raising) + ".")
    if lowering:
        reasons.append("It was lowered by " + "; ".join(describe_item(k, f[k]) for k, _ in lowering) + ".")
    groups = sorted(((g, v) for g, v in res["shap_by_group"].items() if g not in res["missing_groups"]), key=lambda gv: -abs(gv[1]))
    if groups:
        reasons.append(f"Overall, the model relied most on {GROUP_WORDS.get(groups[0][0], groups[0][0])}.")

    reliability = []
    if missing:
        reliability.append("Not available, so not used: " + ", ".join(missing).lower() + ". The estimate is less certain than with complete information.")
    for level, message in checks:
        reliability.append(("Check the entries: " if level == "check" else "Note: ") + message)
    if not reliability:
        reliability.append("All entries are within the ranges seen in the research cohort.")

    evidence = []
    if library is not None and groups:
        asked = set()
        for g, _ in groups[:2]:
            if g not in EVIDENCE_QUESTION or EVIDENCE_QUESTION[g][0] in asked:
                continue
            q, must = EVIDENCE_QUESTION[g]
            asked.add(q)
            hits = [h for h in library.search(q, 8) if h["similarity"] >= 0.35 and re.search(must, h["text"], re.I)]
            if hits:
                h = hits[0]
                text = re.sub(r"^(Purpose|Background|Introduction|Results|Methods|Conclusions?)(?=[A-Z])", "", h["text"])
                sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if re.search(must, x, re.I)]
                sentences.sort(key=lambda x: not re.search(r"\d", x))          # prefer sentences that report a result
                evidence.append({"topic": GROUP_WORDS.get(g, g), "quote": first_sentences(" ".join(sentences) or h["text"]),
                                 "source": f"{h['short']} ({h['year']})", "url": h["url"]})

    limits = ("This is a research estimate from a model trained on volunteers in the NACC research cohort. "
              "It is not a diagnosis, and it does not say what treatment would change the risk. "
              "The factors listed describe what the model used, not causes.")
    return {"summary": summary, "reasons": reasons, "reliability": reliability, "evidence": evidence, "limits": limits}
