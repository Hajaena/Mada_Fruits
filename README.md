# Mada Fruits — RAG avec Gemini (sans embeddings)

Petit assistant client pour une société fictive de vente de fruits.
Les données dans `data/` (prix, livraison, retours, FAQ) sont inventées pour la démo.

## Principe

1. **Ingestion** : lecture des fichiers de `data/` (.txt, .md, .pdf) puis découpage en chunks.
2. **Recherche** : classement des chunks par mots-clés avec BM25 (pas de modèle d'embedding).
3. **Génération** : les 3 meilleurs chunks sont envoyés à Gemini avec la question.

Si aucun chunk n'est pertinent, Gemini n'est pas appelé et l'assistant répond « Je ne sais pas ».

## Installation

    python -m venv .venv
    .venv\Scripts\activate        # Windows  (Linux/Mac : source .venv/bin/activate)
    pip install -r requirements.txt
    copy .env.example .env        # Linux/Mac : cp .env.example .env

Éditez `.env` et ajoutez votre clé API (Google AI Studio : https://aistudio.google.com).
Vérifiez aussi le nom du modèle (`GEMINI_MODEL`) dans la documentation Gemini.

## Utilisation

    python main.py                # questions / réponses avec Gemini
    python main.py --context      # affiche aussi les extraits trouvés
    python main.py --no-llm       # recherche seule, sans clé API

Exemples de questions :
- Combien coûte le litchi ?
- Livrez-vous à Antsirabe ?
- Mon ananas est abîmé, que faire ?
- Comment conserver les fraises ?
- Vendez-vous des pommes ? (hors données : « Je ne sais pas »)

## Structure

    data/        documents de l'entreprise (ajoutez vos PDF ici)
    rag.py       ingestion, chunking, BM25, prompt, appel Gemini
    main.py      interface en ligne de commande

## Limite de BM25

BM25 compare des mots, pas du sens : « tarif » ne retrouve pas « prix » si le mot « tarif »
n'apparaît pas dans le texte. Les embeddings règlent ce problème (suite possible du workshop).
