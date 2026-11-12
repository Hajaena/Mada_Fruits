"""Assistant Mada Fruits : python main.py [--context] [--no-llm]"""

import argparse
import os
import sys

from dotenv import load_dotenv

from rag import build_index, generate, retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context", action="store_true", help="afficher les extraits trouvés"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="recherche seule, sans Gemini"
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    use_llm = not args.no_llm
    if use_llm and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Clé API absente (voir .env) : mode recherche seule.\n")
        use_llm = False

    chunks, bm25 = build_index("data")
    print(f"{len(chunks)} chunks indexés. Tapez 'q' pour quitter.\n\n")
    print("Bienvenu sur Mada Fruit votre assistant pour faire vos achats")

    while True:
        question = input("Vous > ").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        hits = retrieve(question, chunks, bm25)
        if args.context or not use_llm:
            for chunk, score in hits:
                print(
                    f"  [{chunk['source']} #{chunk['id']} | score {score:.2f}] {chunk['text'][:90]}..."
                )
        if use_llm:
            print(f"\nAssistant > {generate(question, hits)}\n")
        print()


if __name__ == "__main__":
    main()
