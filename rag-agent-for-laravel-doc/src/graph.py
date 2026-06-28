# Définition du graphe LangGraph pour l'agent RAG sur la documentation Laravel.
#
# Flux principal :
#   START → retrieve → grade_documents → [web_search?] → generate → grade_generation → END
#
# Si les documents locaux ne sont pas pertinents, le nœud web_search est activé.
# Si la réponse générée est insuffisante, une boucle de retry relance retrieve (max MAX_RETRIES).

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.llm_config import get_llm
from src.tools import retrieve_docs, web_search, grade_documents_batch

# Nombre maximum de tentatives de génération avant de forcer la fin du graphe
MAX_RETRIES = 2


# --- Nœuds du graphe --------------------------------------------------------

def node_retrieve(state: AgentState) -> dict:
    """Nœud 1 — Recherche vectorielle locale.
    Interroge ChromaDB pour trouver les chunks les plus proches de la question."""
    docs = retrieve_docs(state["question"])
    print(f"[retrieve] {len(docs)} documents trouvés")
    return {"documents": docs}


def node_grade_documents(state: AgentState) -> dict:
    """Nœud 2 — Filtrage des documents récupérés.
    Évalue chaque document avec le LLM et ne conserve que les pertinents.
    Positionne needs_web_search=True si aucun document local n'est retenu."""
    relevant_docs = grade_documents_batch(state["question"], state["documents"])
    print(f"[grade_documents] {len(relevant_docs)}/{len(state['documents'])} pertinents")
    return {"documents": relevant_docs, "needs_web_search": len(relevant_docs) == 0}


def node_web_search(state: AgentState) -> dict:
    """Nœud 3 (conditionnel) — Recherche web via Tavily.
    Activé uniquement si node_grade_documents n'a retenu aucun document local."""
    try:
        results = web_search(state["question"])
    except Exception as e:
        print(f"[web_search] échec ({e})")
        results = []
    return {"web_results": results}


# Prompt de génération — le LLM répond uniquement à partir du contexte fourni
# pour éviter les hallucinations sur des détails Laravel non vérifiés
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant expert Laravel. Réponds UNIQUEMENT à partir du contexte "
     "fourni. Si le contexte ne suffit pas, dis-le. Réponds en français.\n\nContexte :\n{context}"),
    ("human", "{question}"),
])


def _format_context(state: AgentState) -> str:
    """Assemble le contexte à injecter dans le prompt de génération.

    Fusionne les documents locaux (ChromaDB) et les résultats web (Tavily)
    en un seul bloc texte avec des séparateurs clairs pour aider le LLM
    à distinguer les sources. Filtre les valeurs non-dict parasites du state.
    """
    # Récupération défensive — gère dict et object avec attributs
    documents = state.get("documents") if isinstance(state, dict) else getattr(state, "documents", [])
    web_results = state.get("web_results") if isinstance(state, dict) else getattr(state, "web_results", [])

    # Filtre les valeurs non-dict (bool, None) qui peuvent arriver du state initial
    documents = [d for d in (documents or []) if isinstance(d, dict)]
    web_results = [d for d in (web_results or []) if isinstance(d, dict)]

    parts = [f"[Local - {d['source']}]\n{d['content']}" for d in documents]
    parts += [f"[Web - {d['source']}]\n{d['content']}" for d in web_results]
    return "\n\n---\n\n".join(parts) if parts else "Aucun contexte disponible."


def node_generate(state: AgentState) -> dict:
    """Nœud 4 — Génération de la réponse finale.
    Invoque le LLM avec le contexte assemblé et incrémente le compteur de retry."""
    llm = get_llm()
    chain = GENERATE_PROMPT | llm
    response = chain.invoke({"context": _format_context(state), "question": state["question"]})
    return {
        "generation": response.content,
        "retry_count": state.get("retry_count", 0) + 1,
        "messages": [AIMessage(content=response.content)],
    }


# Schéma de sortie structurée pour l'évaluation de la réponse générée
class AnswerGrade(BaseModel):
    grounded: bool = Field(description="Réponse fondée sur le contexte, sans invention")
    addresses_question: bool = Field(description="Réponse au point")


# Prompt d'évaluation de la réponse — vérifie à la fois la fidélité au contexte
# et la pertinence par rapport à la question initiale
GRADE_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Évalue : (1) la réponse est-elle fondée sur le contexte ? (2) répond-elle à la question ?"),
    ("human", "Question : {question}\n\nContexte :\n{context}\n\nRéponse :\n{generation}"),
])


def node_grade_generation(state: AgentState) -> dict:
    """Nœud 5 — Évaluation de la réponse générée.
    Le LLM juge si la réponse est (1) ancrée dans le contexte et (2) répond à la question.
    Ces deux flags pilotent le routage conditionnel vers END ou vers un retry."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnswerGrade)
    chain = GRADE_ANSWER_PROMPT | structured_llm
    result: AnswerGrade = chain.invoke({
        "question": state["question"],
        "context": _format_context(state),
        "generation": state["generation"],
    })
    print(f"[grade_generation] grounded={result.grounded}, addresses={result.addresses_question}")

    return {"answer_is_grounded": result.grounded, "answer_addresses_question": result.addresses_question}


# --- Edges conditionnels ----------------------------------------------------

def route_after_grading_docs(state: AgentState) -> str:
    """Si aucun document local pertinent -> recherche web. Sinon -> génération."""
    return "web_search" if state.get("needs_web_search") else "generate"


def route_after_answer_grading(state: AgentState) -> str:
    """Boucle de retry si la réponse n'est pas satisfaisante (max MAX_RETRIES).
    Force la fin si MAX_RETRIES atteint pour éviter une boucle infinie."""
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"
    if state.get("answer_is_grounded") and state.get("answer_addresses_question"):
        return "end"
    return "retry"


# --- Construction du graphe -------------------------------------------------

def build_graph(with_memory: bool = True):
    """Construit et compile le graphe LangGraph de l'agent RAG.

    with_memory=True : active MemorySaver pour la persistance de la conversation
                       (multi-turn, utilisé en production et en CLI).
    with_memory=False : sans checkpointer, utilisé pour LangGraph Studio et les tests.
    """
    workflow = StateGraph(AgentState)

    # Enregistrement des nœuds
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("web_search", node_web_search)
    workflow.add_node("generate", node_generate)
    workflow.add_node("grade_generation", node_grade_generation)

    # Arêtes fixes du flux principal
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Arête conditionnelle : web_search si pas de docs pertinents, sinon generate directement
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading_docs,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_generation")

    # Arête conditionnelle : retry (retour à retrieve) ou fin selon la qualité de la réponse
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_answer_grading,
        {"retry": "retrieve", "end": END},
    )

    if with_memory:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# Graphe compilé exposé pour LangGraph Studio (langgraph dev)
graph = build_graph(with_memory=False)


if __name__ == "__main__":
    # Génère une visualisation PNG du graphe dans data/graph.png
    app = build_graph()
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("data/graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graphe sauvegardé dans data/graph.png")
    except Exception as e:
        print(f"[WARN] Impossible de générer le PNG ({e}). Affichage mermaid brut :")
        print(app.get_graph().draw_mermaid())
