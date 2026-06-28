# Outils utilisés par l'agent RAG pour récupérer et évaluer des documents.
# Trois outils principaux :
#   - retrieve_docs  : recherche vectorielle locale dans ChromaDB
#   - web_search     : recherche web via Tavily si la base locale est insuffisante
#   - grade_document : évalue si un extrait est pertinent pour la question posée

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.llm_config import get_llm

# Charge les variables d'environnement (.env) — notamment TAVILY_API_KEY
load_dotenv()

# Chemin vers la base ChromaDB persistée par chunking.py
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")

# Nom de la collection ChromaDB (doit correspondre à celui utilisé lors de l'indexation)
COLLECTION_NAME = "laravel_docs"

# Nombre de documents retournés par la recherche vectorielle par défaut
TOP_K = 3


def _get_vectorstore() -> Chroma:
    """Ouvre la connexion au vectorstore ChromaDB existant (lecture seule).
    Le modèle d'embedding doit être identique à celui utilisé lors de l'indexation."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve_docs(query: str, k: int = TOP_K) -> list[dict]:
    """Recherche les k documents locaux les plus similaires à la requête.

    Retourne une liste de dicts avec : content, source (topic), url et score de similarité.
    Le score permet de juger la qualité de la correspondance avant d'envoyer au LLM.
    """
    vs = _get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "url": doc.metadata.get("url", ""),
            "score": round(score, 3),
        }
        for doc, score in results
    ]


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Effectue une recherche web via l'API Tavily en ciblant Laravel.

    Utilisé en fallback quand la base locale ne contient pas de réponse pertinente.
    Le préfixe 'Laravel' est ajouté à la query pour concentrer les résultats.
    """
    tavily = TavilySearch(max_results=max_results)
    results = tavily.invoke({"query": f"Laravel {query}"})
    # Tavily peut retourner un dict avec clé "results" ou directement une liste
    items = results.get("results", []) if isinstance(results, dict) else results
    return [
        {
            "content": item.get("content", ""),
            "source": item.get("title", "web"),
            "url": item.get("url", ""),
            "score": item.get("score", None),
        }
        for item in items
    ]


# Schéma de sortie structurée pour le grading — force le LLM à répondre par True/False
class DocumentGrade(BaseModel):
    relevant: bool = Field(description="True si le document permet de répondre à la question")


# Prompt système pour l'évaluateur de pertinence — volontairement tolérant
# pour éviter de filtrer des documents partiellement utiles
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un évaluateur qui détermine si un extrait de documentation "
     "permet de répondre à une question technique sur Laravel. "
     "Sois tolérant : si l'extrait contient une partie utile, considère-le pertinent."),
    ("human", "Question : {question}\n\nExtrait :\n{document}"),
])


def grade_document(question: str, document: str) -> bool:
    """Évalue si un extrait de document est pertinent pour répondre à la question.

    Utilise le LLM avec une sortie structurée (DocumentGrade) pour obtenir
    une décision binaire fiable plutôt qu'un texte libre.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(DocumentGrade)
    chain = GRADE_PROMPT | structured_llm
    result: DocumentGrade = chain.invoke({"question": question, "document": document})
    return result.relevant


def grade_documents_batch(question: str, documents: list[dict]) -> list[dict]:
    """Filtre une liste de documents en ne gardant que ceux jugés pertinents.

    En cas d'erreur sur un document individuel (ex: LLM timeout), le document
    est conservé par défaut pour ne pas perdre de contexte potentiellement utile.
    """
    graded = []
    for doc in documents:
        try:
            if grade_document(question, doc["content"]):
                graded.append(doc)
        except Exception as e:
            # Conserve le document si le grading échoue — principe de précaution
            print(f"[WARN] grading échoué ({e}), conservé par défaut")
            graded.append(doc)
    return graded
