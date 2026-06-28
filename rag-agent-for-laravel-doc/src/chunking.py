# Pipeline de découpage (chunking) de la documentation Laravel.
# Lit les fichiers Markdown bruts, les divise d'abord par sections Markdown
# puis par taille de caractères, et indexe les chunks dans une base vectorielle ChromaDB.

import os
import glob

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Dossier contenant les fichiers Markdown téléchargés par scrape_laravel_docs.py
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Dossier où ChromaDB va persister les vecteurs sur disque
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

# Nom de la collection ChromaDB (identifiant logique du vectorstore)
COLLECTION_NAME = "laravel_docs"

# Niveaux de titres Markdown utilisés comme frontières de découpage sémantique
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Taille maximale d'un chunk en caractères et chevauchement entre chunks consécutifs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_and_chunk() -> list[Document]:
    """Charge tous les fichiers .md du dossier RAW_DIR et les découpe en chunks.

    Étape 1 — MarkdownHeaderTextSplitter : découpe sur les titres H1/H2/H3
               pour conserver la cohérence sémantique des sections.
    Étape 2 — RecursiveCharacterTextSplitter : redécoupe les sections trop longues
               en sous-chunks de CHUNK_SIZE caractères avec CHUNK_OVERLAP de chevauchement.
    Chaque chunk reçoit des métadonnées : topic (nom du fichier), chunk_id unique, url Laravel.
    """
    # Splitter par titres Markdown pour respecter la structure des sections
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False,
    )
    # Splitter par taille pour limiter les chunks trop longs pour le modèle d'embedding
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[Document] = []
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.md")))
    print(f"{len(files)} fichiers trouvés")

    for filepath in files:
        # Le nom du fichier (sans extension) sert d'identifiant de topic
        topic = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Double découpage : par sections Markdown puis par taille
        section_docs = md_splitter.split_text(text)
        sub_docs = char_splitter.split_documents(section_docs)

        # Enrichissement des métadonnées pour la traçabilité et les citations
        for i, doc in enumerate(sub_docs):
            doc.metadata["source"] = topic
            doc.metadata["chunk_id"] = f"{topic}-{i}"
            doc.metadata["url"] = f"https://laravel.com/docs/11.x/{topic}"
            all_chunks.append(doc)

    print(f"{len(all_chunks)} chunks générés")
    return all_chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Génère les embeddings pour chaque chunk et les persiste dans ChromaDB.

    Utilise le modèle nomic-embed-text via Ollama (local, sans API externe).
    Le vectorstore est sauvegardé sur disque dans PERSIST_DIR pour être réutilisé
    lors des requêtes RAG sans avoir à re-embedder.
    """
    print("Génération des embeddings via Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Vectorstore persistée dans {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    # Exécution standalone : charge, découpe, affiche un aperçu et indexe
    chunks = load_and_chunk()
    for c in chunks[:3]:
        print("---", c.metadata, c.page_content[:150])
    build_vectorstore(chunks)
