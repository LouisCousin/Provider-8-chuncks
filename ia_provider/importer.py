from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import docx
from docx.document import Document as DocumentObject
from docx.section import _Header, _Footer
from docx.opc.exceptions import OpcError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
import fitz  # PyMuPDF

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _extraire_style_run(run) -> Dict[str, Any]:
    """Extrait les informations de style d'un segment de texte (run)."""
    font = run.font
    color = font.color.rgb if font.color and font.color.rgb else None
    return {
        "text": run.text,
        "style": {
            "font_name": font.name,
            "font_size": font.size.pt if font.size else None,
            "is_bold": font.bold,
            "is_italic": font.italic,
            "font_color_rgb": str(color) if color else None,
        },
    }


def _analyser_contenu_block(parent: Union[DocumentObject, _Header, _Footer, _Cell]) -> List[Dict[str, Any]]:
    """Analyse un conteneur (document, header, cell, etc.) et retourne la structure des blocs."""

    block_items = []
    # Logique qui s'adapte au type de "parent"
    if hasattr(parent, "element"):
        # Cas pour le corps du document et les cellules de tableau
        parent_element = parent._tc if isinstance(parent, _Cell) else parent.element.body
        for child in parent_element.iterchildren():
            if isinstance(child, CT_P):
                block_items.append(Paragraph(child, parent))
            elif isinstance(child, CT_Tbl):
                block_items.append(Table(child, parent))
    elif hasattr(parent, "paragraphs"):
        # Cas pour les en-têtes (_Header) et pieds de page (_Footer)
        # Note: L'ordre n'est pas garanti si les paragraphes et tableaux sont mélangés
        block_items.extend(parent.paragraphs)
        block_items.extend(parent.tables)

    contenu_structure: List[Dict[str, Any]] = []
    for block in block_items:
        if isinstance(block, Paragraph):
            if not block.text.strip():
                continue

            style_name = block.style.name.lower() if block.style and block.style.name else ""

            # Gestion des listes
            if "list" in style_name or "liste" in style_name:
                if contenu_structure and contenu_structure[-1]["type"] == "list":
                    contenu_structure[-1]["items"].append(block.text)
                else:
                    contenu_structure.append({"type": "list", "items": [block.text]})
                continue

            # Gestion des titres et paragraphes
            block_type = "paragraph"
            if style_name.startswith("heading 1") or style_name.startswith("titre 1"):
                block_type = "heading_1"
            elif style_name.startswith("heading 2") or style_name.startswith("titre 2"):
                block_type = "heading_2"

            runs_data = [_extraire_style_run(run) for run in block.runs if run.text.strip()]
            if runs_data:
                contenu_structure.append({"type": block_type, "runs": runs_data})

        elif isinstance(block, Table):
            table_data: List[List[Dict[str, Any]]] = []
            for row in block.rows:
                row_data = [_analyser_contenu_block(cell) for cell in row.cells]
                table_data.append(row_data)
            if table_data:
                contenu_structure.append({"type": "table", "rows": table_data})

    return contenu_structure


def analyser_docx(
    file_stream,
) -> Tuple[Dict[str, List[Dict[str, Any]]], None]:
    """Extrait le contenu structuré d'un DOCX, y compris en-têtes et pieds de page."""
    try:
        file_stream.seek(0)
        document = docx.Document(file_stream)

        # 1. Analyser le corps du document
        corps_structure = _analyser_contenu_block(document)

        # 2. Analyser l'en-tête et le pied de page (simplifié à la première section)
        header_structure = []
        footer_structure = []
        if document.sections:
            section = document.sections[0]
            if section.header:
                header_structure = _analyser_contenu_block(section.header)
            if section.footer:
                footer_structure = _analyser_contenu_block(section.footer)

        document_complet = {
            "header": header_structure,
            "body": corps_structure,
            "footer": footer_structure,
        }
        return document_complet, None

    except OpcError as e:
        logging.error(f"Fichier DOCX corrompu : {e}")
        return {"header": [], "body": [], "footer": []}, None
    except Exception as e:
        logging.error(f"Erreur inattendue sur DOCX : {e}", exc_info=True)
        return {"header": [], "body": [], "footer": []}, None


def analyser_pdf(file_stream) -> Tuple[str, None]:
    """Extrait le contenu textuel brut d'un PDF."""
    try:
        file_stream.seek(0)
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        return full_text, None
    except Exception as e:
        logging.error(f"Erreur inattendue sur PDF : {e}", exc_info=True)
        return "", None


def analyser_document(
    fichier,
) -> Tuple[Union[str, Dict[str, List[Dict[str, Any]]]], None]:
    """Analyse un fichier importé et choisit la méthode appropriée."""
    filename = fichier.name.lower()
    if filename.endswith(".docx"):
        return analyser_docx(fichier)
    if filename.endswith(".pdf"):
        return analyser_pdf(fichier)
    return "", None


def decouper_document_en_chunks(
    document_structure: Dict[str, List[Dict[str, Any]]],
    seuil_blocs: int = 150,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Découpe une structure de document en plusieurs chunks si elle dépasse un seuil.

    Args:
        document_structure: Structure complète du document analysé.
        seuil_blocs: Nombre maximum de blocs autorisés par chunk.

    Returns:
        Liste de structures de document découpées. Si le document
        ne dépasse pas ``seuil_blocs``, la liste contiendra une seule
        structure correspondant au document original.
    """

    corps = document_structure.get("body", [])

    # Si le document est sous le seuil, aucune découpe nécessaire
    if len(corps) <= seuil_blocs:
        return [document_structure]

    chunks: List[Dict[str, List[Dict[str, Any]]]] = []
    chunk_courant: List[Dict[str, Any]] = []

    for bloc in corps:
        # RÈGLE 1 : Si le chunk courant atteint le seuil, on le coupe de force.
        if len(chunk_courant) >= seuil_blocs:
            # RÈGLE 3 : Avant de couper, on vérifie s'il y a un titre orphelin.
            if chunk_courant and chunk_courant[-1].get("type", "").startswith("heading"):
                titre_orphelin = chunk_courant.pop()
                nouveau_chunk_structure = {
                    "header": document_structure.get("header", []),
                    "body": chunk_courant,
                    "footer": document_structure.get("footer", []),
                }
                chunks.append(nouveau_chunk_structure)

                # Le nouveau chunk commence avec le titre qui était orphelin
                chunk_courant = [titre_orphelin, bloc]
                continue
            else:
                nouveau_chunk_structure = {
                    "header": document_structure.get("header", []),
                    "body": chunk_courant,
                    "footer": document_structure.get("footer", []),
                }
                chunks.append(nouveau_chunk_structure)
                chunk_courant = []

        # Découpage sémantique sur les titres de chapitre
        if bloc.get("type") == "heading_1" and chunk_courant:
            nouveau_chunk_structure = {
                "header": document_structure.get("header", []),
                "body": chunk_courant,
                "footer": document_structure.get("footer", []),
            }
            chunks.append(nouveau_chunk_structure)
            chunk_courant = []

        chunk_courant.append(bloc)

    # Ajouter le dernier chunk restant
    if chunk_courant:
        dernier_chunk_structure = {
            "header": document_structure.get("header", []),
            "body": chunk_courant,
            "footer": document_structure.get("footer", []),
        }
        chunks.append(dernier_chunk_structure)

    return chunks
