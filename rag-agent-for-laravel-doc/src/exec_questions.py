"""
Lance les 20 questions et loggue automatiquement :
- temps total par question
- web search déclenché ou non
- retry_count
- grounded / addresses_question

"""

import time
import json
from langchain_core.messages import HumanMessage

import sys
sys.path.insert(0, ".")
from src.graph import build_graph

QUESTIONS = [
    ("S1",  "simple",   "Comment définir une route GET dans Laravel ?"),
    ("S2",  "simple",   "Qu'est-ce qu'un Eloquent Model et comment en créer un ?"),
    ("S3",  "simple",   "Comment créer une migration avec Artisan ?"),
    ("S4",  "simple",   "À quoi sert le middleware auth dans Laravel ?"),
    ("S5",  "simple",   "Comment valider une requête entrante avec Form Request ?"),
    ("S6",  "simple",   "Comment définir une relation hasMany dans Eloquent ?"),
    ("S7",  "simple",   "Comment retourner une réponse JSON depuis un contrôleur ?"),
    ("S8",  "simple",   "Comment stocker et récupérer une valeur dans le cache Laravel ?"),
    ("S9",  "simple",   "Comment dispatcher un job dans une queue ?"),
    ("S10", "simple",   "Qu'est-ce qu'un Service Provider et à quoi sert-il ?"),
    ("C1",  "complexe", "Quelle est la différence entre eager loading et lazy loading dans Eloquent, et comment éviter le problème N+1 ?"),
    ("C2",  "complexe", "Comment mettre en place une authentification API avec Sanctum en gérant les tokens et les middlewares ?"),
    ("C3",  "complexe", "Quelle est la différence entre Laravel Sanctum et Laravel Passport, et quand choisir l'un plutôt que l'autre ?"),
    ("C4",  "complexe", "Comment fonctionne le système d'événements et de listeners, et comment les combiner avec les queues ?"),
    ("C5",  "complexe", "Comment tester un contrôleur qui dépend d'un service externe en utilisant les mocks Laravel ?"),
    ("C6",  "complexe", "Comment optimiser les performances d'une application Laravel souffrant de requêtes lentes en base de données ?"),
    ("C7",  "complexe", "Comment structurer une application Laravel avec le pattern Repository et le Service Layer ?"),
    ("C8",  "complexe", "Comment implémenter une stratégie de cache pour une API Laravel à fort trafic ?"),
    ("C9",  "complexe", "Comment configurer et utiliser les jobs chainés et les batches de jobs dans Laravel ?"),
    ("C10", "complexe", "Comment fonctionne le container IoC de Laravel et comment l'utiliser pour l'injection de dépendances ?"),
]

def run():
    app = build_graph(with_memory=False)
    results = []

    print(f"\n{'ID':<5} {'Type':<8} {'Temps':>7} {'Docs':>5} {'Web':>5} {'Grnd':>5} {'AdrQ':>5} {'Retry':>6}")
    print("-" * 55)

    for qid, qtype, question in QUESTIONS:
        start = time.perf_counter()
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=question)],
                "question": question,
                "documents": [],
                "web_results": [],
                "needs_web_search": False,
                "generation": "",
                "retry_count": 0,
                "answer_is_grounded": False,
                "answer_addresses_question": False,
            })
            elapsed = round(time.perf_counter() - start, 1)

            row = {
                "id": qid,
                "type": qtype,
                "question": question,
                "temps_s": elapsed,
                "docs_pertinents": len(result.get("documents", [])),
                "web_search": bool(result.get("web_results")),
                "grounded": result.get("answer_is_grounded"),
                "addresses_question": result.get("answer_addresses_question"),
                "retry_count": result.get("retry_count", 1) - 1,
                "generation": result.get("generation", "")[:120],
            }
        except Exception as e:
            elapsed = round(time.perf_counter() - start, 1)
            row = {
                "id": qid, "type": qtype, "question": question,
                "temps_s": elapsed, "erreur": str(e),
                "docs_pertinents": "?", "web_search": "?",
                "grounded": "?", "addresses_question": "?", "retry_count": "?",
            }

        results.append(row)
        print(
            f"{row['id']:<5} {row['type']:<8} {row['temps_s']:>6}s "
            f"{str(row['docs_pertinents']):>5} "
            f"{'oui' if row['web_search'] else 'non':>5} "
            f"{'✓' if row['grounded'] else '✗':>5} "
            f"{'✓' if row['addresses_question'] else '✗':>5} "
            f"{str(row['retry_count']):>6}"
        )

    # Export JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Résumé
    valid = [r for r in results if "erreur" not in r]
    simples  = [r for r in valid if r["type"] == "simple"]
    complexes = [r for r in valid if r["type"] == "complexe"]

    print("\n ----------- RÉSUMÉ -----------------------------")
    if simples:
        print(f"Simples/ temps moyen : {sum(r['temps_s'] for r in simples)/len(simples):.1f}s")
    if complexes:
        print(f"Complexes / temps moyen : {sum(r['temps_s'] for r in complexes)/len(complexes):.1f}s")
    print(f"Web search déclenché   : {sum(1 for r in valid if r['web_search'])}/{len(valid)}")
    print(f"Grounded               : {sum(1 for r in valid if r['grounded'])}/{len(valid)}")
    print(f"Addresses question     : {sum(1 for r in valid if r['addresses_question'])}/{len(valid)}")
    print(f"Résultats : evaluation_results.json")

if __name__ == "__main__":
    run()