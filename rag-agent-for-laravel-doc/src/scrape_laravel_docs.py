# Script de scraping de la documentation Laravel depuis GitHub.
# Télécharge chaque page en format Markdown et la sauvegarde localement
# pour être utilisée ensuite par le pipeline RAG.

import os
import time
import requests

# Branche de la documentation Laravel à télécharger (ex: "11.x", "10.x")
BRANCH = "11.x"

# URL de base pour accéder aux fichiers bruts sur GitHub
RAW_BASE = f"https://raw.githubusercontent.com/laravel/docs/{BRANCH}"

# Dossier de sortie où les fichiers Markdown seront sauvegardés
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Liste des pages (slugs) à télécharger — chaque slug correspond à un fichier .md
PAGES = [
   "routing", "middleware", "controllers", "requests", "responses",
    "views", "blade","validation", "errors", "artisan",
    "eloquent", "eloquent-relationships", "eloquent-resources",
     "pagination", "migrations", "seeding",
    "authentication", "authorization", "sanctum", "passport",
    "queues", "events", "broadcasting", "scheduling", "mail",
    "notifications", "filesystem", "cache", "collections",
     "facades", "providers", "container",
    "testing", "configuration", "structure",
]


def fetch_page(slug: str) -> str | None:
    """Télécharge le contenu Markdown d'une page de la doc Laravel via son slug.
    Retourne le texte brut si succès, None sinon."""
    url = f"{RAW_BASE}/{slug}.md"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        return resp.text
    print(f"  [!] {slug}.md -> HTTP {resp.status_code}")
    return None


def main():
    # Crée le dossier de sortie s'il n'existe pas encore
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ok, fail = 0, 0

    for slug in PAGES:
        content = fetch_page(slug)
        if content:
            out_path = os.path.join(OUTPUT_DIR, f"{slug}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                # Ajoute un commentaire d'en-tête pour tracer la source du fichier
                f.write(f"<!-- source: laravel/docs ({BRANCH}) - {slug}.md -->\n\n")
                f.write(content)
            print(f"  [OK] {slug}.md ({len(content)} chars)")
            ok += 1
        else:
            fail += 1
        # Pause courte pour éviter de surcharger l'API GitHub
        time.sleep(0.1)

    print(f"\nTerminé : {ok} pages, {fail} échecs.")


if __name__ == "__main__":
    main()
