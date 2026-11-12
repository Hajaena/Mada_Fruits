"""RAG sans embeddings : recherche par mots-clés (BM25) + génération Gemini."""

import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 250, 50, 3

STOPWORDS = set(
    "le la les un une des de du d l et ou a au aux en dans pour par sur avec ce ces cet cette "
    "est sont il elle on je tu vous nous que qui quoi quel quelle quels quelles y ne pas se sa "
    "son ses mon ma mes ton ta tes notre votre vos leur leurs c s j n m t qu".split()
)

SYSTEM = (
    "Tu es l'assistant client de Mada Fruits, une société de vente de fruits. "
    "Réponds en français, de façon courte et claire, uniquement à partir du CONTEXTE fourni. "
    "Si la personne te salue, salue-la et propose ton aide (fruits, livraison, retours). "
    "Fais le calcul si le client demande le 1/2, 1/3.. etc"
    "Si la réponse n'est pas dans le contexte, réponds : "
    "« On n'a pas cette information pour l'instant, contactez notre service client. » "
    "Indique la source entre crochets, par exemple [livraison.txt]."
)
NO_ANSWER = (
    "On n'a pas cette information pour l'instant, contactez notre service client."
)


# 1 Ingestion
def load_documents(folder="data"):
    """Lit les .txt / .md / .pdf du dossier -> liste de (nom_fichier, texte)."""
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in (".txt", ".md"):
            docs.append((path.name, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            pages = PdfReader(path).pages
            docs.append((path.name, "\n".join(p.extract_text() or "" for p in pages)))
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpe en chunks d'environ `size` caractères, avec chevauchement."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # coupe proprement sur un saut de ligne ou un espace
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)  # ne pas démarrer au milieu d'un mot
        if space != -1:
            start = space + 1
    return chunks


# 2 Recherche par mots-clés (BM25)
def tokenize(text):
    """Minuscules, sans accents, sans mots vides, pluriels simplifiés."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    words = re.findall(r"[a-z0-9]+", text)
    return [w.rstrip("sx") if len(w) > 3 else w for w in words if w not in STOPWORDS]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.length = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.length) / len(self.length)
        df = Counter(w for c in self.tf for w in c)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        words, out = tokenize(query), []
        for tf, dl in zip(self.tf, self.length):
            s = 0.0
            for w in words:
                if w in tf:
                    f = tf[w]
                    norm = f + self.k1 * (1 - self.b + self.b * dl / self.avg)
                    s += self.idf[w] * f * (self.k1 + 1) / norm
            out.append(s)
        return out


def build_index(folder="data"):
    chunks = []
    for source, text in load_documents(folder):
        for i, c in enumerate(chunk_text(text)):
            chunks.append({"source": source, "id": i, "text": c})
    return chunks, BM25([c["text"] for c in chunks])


def retrieve(question, chunks, bm25, k=TOP_K):
    scores = bm25.scores(question)
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    hits = [(chunks[i], scores[i]) for i in best if scores[i] > 0]

    top = hits[0][1] if hits else 1
    for chunk, score in hits:
        print(f"{score / top:.0%}  ({score:.2f})  {chunk['source']}")

    return hits


# 3 génération augmentée Gemini
def build_prompt(question, hits):
    context = (
        "\n\n".join(f"[{c['source']}]\n{c['text']}" for c, _ in hits)
        or "(aucun extrait trouvé)"
    )
    return f"CONTEXTE :\n{context}\n\nQUESTION : {question}"


def generate(question, hits):
    import time
    from google import genai
    from google.genai import errors, types

    client = genai.Client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2)
    models = [
        m
        for m in (
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
        )
        if m
    ]
    for model in models:
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=model, contents=build_prompt(question, hits), config=config
                ).text
            except errors.APIError as e:
                if e.code in (
                    404,
                    429,
                ):
                    break
                if e.code in (500, 503):  # surcharge
                    time.sleep(2**attempt)
                    continue
                raise
    extraits = "\n".join(f"- [{c['source']}] {c['text'][:150]}" for c, _ in hits)
    return (
        "Gemini est indisponible (quota ou surcharge). Extraits trouvés :\n" + extraits
    )
