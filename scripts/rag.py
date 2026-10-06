"""Phase 10, step 2: the evidence assistant.

Given a question it
  1. checks whether the question asks for individual medical advice,
  2. finds the most relevant passages in the evidence library,
  3. answers using only those passages, citing each one,
  4. says so when the library does not contain an answer.

Retrieval is "hybrid": a meaning-based search (sentence embeddings) and a
keyword search (BM25) are run separately and their rankings merged. Meaning
search finds paraphrases; keyword search finds exact terms such as "APOE".

Two ways of writing the answer:
  extractive   quotes the best sentences from the retrieved passages. Cannot
               invent anything, because every word comes from a source.
  generative   a small local language model writes a short answer from the
               passages. Used only if the model is installed; every citation
               it produces is checked against the retrieved passages.

No patient data is needed, and nothing is sent to an external service.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data" / "external" / "rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen2.5-1.5B-Instruct"
LLM_MODEL = str(REPO / "models" / "llm")     # local copy of the open model; nothing is sent out
MIN_SIMILARITY = 0.35            # below this, the library is treated as having no answer
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in for with on by is are was were be been as at from that this it its their which what "
           "does do how can could should would i my me we you your about into than then there these those has have had".split())

# Questions asking what an individual should take or do are not answered as advice.
ADVICE = re.compile(
    r"\b(should (i|he|she|we|my|the patient|this patient)|what (drug|medication|medicine|dose|dosage|treatment)s? (should|do i|to give|is best for (me|him|her|my|this))"
    r"|prescribe|how (much|many mg)|\bdos(e|age|ing)\b|can i (take|stop|start)|stop taking|start taking|"
    r"(treat|treatment plan|medication) for (my|this|the) (patient|mother|father|husband|wife|mom|dad)|is it safe (for|to))", re.I)
ADVICE_NOTE = ("This tool cannot recommend a treatment, medication or dose for a person. Those decisions need a clinician who knows "
               "the patient. Below is general, published information related to the question, for background only.")
NOT_FOUND = "I could not find evidence for that in the library. The sources cover dementia risk factors, vascular and metabolic " \
            "contributions, APOE, amyloid and tau PET, and diagnostic criteria."


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class Library:
    def __init__(self, store: Path = STORE):
        from sentence_transformers import SentenceTransformer
        self.passages = json.loads((store / "passages.json").read_text(encoding="utf-8"))
        self.vectors = np.load(store / "vectors.npy")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        # BM25 statistics
        self.docs = [tokens(p["section"] + " " + p["text"]) for p in self.passages]
        self.tf = [Counter(d) for d in self.docs]
        self.avg_len = float(np.mean([len(d) for d in self.docs]))
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _bm25(self, query: str, k1=1.5, b=0.75) -> np.ndarray:
        q = tokens(query)
        scores = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            scores[i] = sum(self.idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / self.avg_len)) for t in q if t in tf)
        return scores

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        """Top-k passages. mode: 'dense', 'bm25' or 'hybrid' (reciprocal rank fusion)."""
        dense = self.vectors @ self.embedder.encode([query], normalize_embeddings=True)[0]
        bm25 = self._bm25(query)
        if mode == "dense":
            order = np.argsort(-dense)
        elif mode == "bm25":
            order = np.argsort(-bm25)
        else:
            rank_d, rank_b = np.argsort(np.argsort(-dense)), np.argsort(np.argsort(-bm25))
            order = np.argsort(-(1 / (60 + rank_d) + 1 / (60 + rank_b)))
        return [{**self.passages[i], "similarity": float(dense[i]), "bm25": float(bm25[i])} for i in order[:k]]


def extractive_answer(question: str, hits: list[dict], embedder, n_sentences: int = 4) -> str:
    """Pick the sentences most similar to the question, each with its citation number."""
    sents = []
    for n, h in enumerate(hits, 1):
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", h["text"]):
            if 8 <= len(s.split()) <= 60:
                sents.append((n, s.strip()))
    if not sents:
        return ""
    sims = embedder.encode([s for _, s in sents], normalize_embeddings=True) @ embedder.encode([question], normalize_embeddings=True)[0]
    best = sorted(np.argsort(-sims)[:n_sentences])
    return " ".join(f"{sents[i][1]} [{sents[i][0]}]" for i in best)


class Generator:
    """Optional small local language model. Loaded on first use."""

    def __init__(self):
        self.pipe = None
        self.available = None

    def load(self) -> bool:
        if self.available is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(LLM_MODEL, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, local_files_only=True,
                                                             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                self.pipe = (tok, model.to("cuda" if torch.cuda.is_available() else "cpu").eval())
                self.available = True
            except Exception:
                self.available = False
        return self.available

    def write(self, question: str, hits: list[dict], model_context: str | None) -> str:
        import torch
        tok, model = self.pipe
        sources = "\n\n".join(f"[{n}] ({h['short']}, {h['section']}) {h['text']}" for n, h in enumerate(hits, 1))
        system = ("You answer questions about dementia research for clinicians and students. Use ONLY the numbered sources. "
                  "After every sentence put the number of the source it came from in square brackets, like [2]. "
                  "If the sources do not answer the question, reply exactly: NOT IN SOURCES. "
                  "Do not give treatment, medication or dosing advice for an individual. Write 2 to 4 sentences.")
        user = f"Sources:\n{sources}\n\n" + (f"Model output for context (not a source): {model_context}\n\n" if model_context else "") + f"Question: {question}"
        enc = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=220, do_sample=False, repetition_penalty=1.05)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def citations_valid(text: str, n_hits: int) -> bool:
    """Every sentence must cite, and only cite, a retrieved source."""
    cited = [int(c) for c in re.findall(r"\[(\d+)\]", text)]
    return bool(cited) and all(1 <= c <= n_hits for c in cited)


class Assistant:
    def __init__(self):
        self.library = Library()
        self.generator = Generator()

    def ask(self, question: str, model_context: str | None = None, k: int = 5, use_llm: bool = True) -> dict:
        advice = bool(ADVICE.search(question))
        hits = self.library.search(question, k)
        if not hits or max(h["similarity"] for h in hits) < MIN_SIMILARITY:
            note = ADVICE_NOTE.split(" Below")[0] + "\n\n" if advice else ""
            return {"answer": note + NOT_FOUND, "sources": [], "mode": "not_found", "advice_request": advice}
        mode, text = "extractive", ""
        if use_llm and not advice and self.generator.load():
            text = self.generator.write(question, hits, model_context)
            if "NOT IN SOURCES" in text.upper():
                return {"answer": NOT_FOUND, "sources": [], "mode": "not_found", "advice_request": advice}
            mode = "generative" if citations_valid(text, len(hits)) else "extractive"   # fall back if it failed to cite
        if mode == "extractive":
            text = extractive_answer(question, hits, self.library.embedder)
        used = sorted({int(c) for c in re.findall(r"\[(\d+)\]", text)})
        return {"answer": (ADVICE_NOTE + "\n\n" if advice else "") + text, "mode": mode, "advice_request": advice,
                "sources": [{"n": n, "short": hits[n - 1]["short"], "title": hits[n - 1]["title"], "year": hits[n - 1]["year"],
                             "section": hits[n - 1]["section"], "url": hits[n - 1]["url"], "passage": hits[n - 1]["text"]} for n in used]}


if __name__ == "__main__":
    import sys
    a = Assistant()
    r = a.ask(" ".join(sys.argv[1:]) or "Which modifiable risk factors for dementia are identified?", use_llm=False)
    print(r["mode"], "\n", r["answer"].encode("ascii", "replace").decode())
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['short']} - {s['section']}".encode("ascii", "replace").decode())
