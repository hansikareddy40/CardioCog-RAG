"""Phase 10, step 1: build the evidence library for the assistant.

  1. download the open-access full text of each source in rag/sources.json
     from Europe PMC (kept under data/external/rag, not committed),
  2. split each article into passages of about 180 words, keeping the section
     heading so a passage can be cited precisely,
  3. turn each passage into a vector with a sentence-embedding model,
  4. save the passages and vectors for searching.

Usage:  python scripts/rag_build.py
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "rag" / "sources.json"
STORE = REPO / "data" / "external" / "rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORDS_PER_PASSAGE = 180
SKIP_SECTIONS = re.compile(r"reference|acknowledg|funding|conflict|contributor|declaration|supplement|author|disclosure|data sharing", re.I)


def fetch(source: dict) -> Path | None:
    path = STORE / "xml" / f"{source['id']}.xml"
    if not path.exists():
        r = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source['pmcid']}/fullTextXML", timeout=120)
        if r.status_code != 200 or len(r.content) < 20000:
            print(f"[rag] could not fetch {source['id']} ({r.status_code})")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def text_of(node) -> str:
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def passages(source: dict, path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    meta = {"source": source["id"], "short": source["short"], "pmcid": source["pmcid"],
            "title": text_of(root.find(".//article-title")),
            "year": (root.findtext(".//pub-date/year") or ""), "journal": (root.findtext(".//journal-title") or ""),
            "url": f"https://europepmc.org/article/PMC/{source['pmcid']}"}
    blocks = []
    abstract = root.find(".//abstract")
    if abstract is not None:
        blocks.append(("Abstract", text_of(abstract)))
    body = root.find(".//body")
    for sec in ([] if body is None else body.iter("sec")):
        heading = text_of(sec.find("title")) if sec.find("title") is not None else ""
        if SKIP_SECTIONS.search(heading):
            continue
        for p in sec.findall("p"):                      # direct paragraphs only, so nested sections are not repeated
            t = re.sub(r"\[[\d,\s\-–]+\]", "", text_of(p))
            if len(t.split()) >= 25:
                blocks.append((heading, t))
    out, buf, head = [], [], None

    def flush():
        if buf:
            out.append({**meta, "section": head or "", "text": " ".join(buf)})

    for heading, t in blocks:
        if heading != head or sum(len(b.split()) for b in buf) + len(t.split()) > WORDS_PER_PASSAGE * 1.5:
            flush(); buf, head = [], heading
        words = t.split()
        while len(words) > WORDS_PER_PASSAGE * 1.5:     # very long paragraph: cut on the word limit
            buf.append(" ".join(words[:WORDS_PER_PASSAGE])); flush(); buf = []
            words = words[WORDS_PER_PASSAGE:]
        buf.append(" ".join(words))
    flush()
    return out


def main():
    from sentence_transformers import SentenceTransformer

    sources = json.loads(SOURCES.read_text())
    chunks = []
    for s in sources:
        path = fetch(s)
        if path:
            ps = passages(s, path)
            chunks += ps
            print(f"[rag] {s['id']}: {len(ps)} passages")
    for i, c in enumerate(chunks):
        c["id"] = i
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([f"{c['section']}. {c['text']}" for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "passages.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(STORE / "vectors.npy", vectors.astype(np.float32))
    print(f"[rag] {len(chunks)} passages from {len({c['source'] for c in chunks})} sources; vectors {vectors.shape}")


if __name__ == "__main__":
    main()
