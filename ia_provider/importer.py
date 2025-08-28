from __future__ import annotations

import logging
from typing import Tuple

import docx
from docx.opc.exceptions import OpcError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
import fitz  # PyMuPDF

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _convertir_docx_en_markdown(document: docx.Document) -> str:
    """Convertit un ``Document`` DOCX en texte Markdown."""

    markdown_lines: list[str] = []

    for element in document.element.body:
        if isinstance(element, CT_P):
            paragraphe = Paragraph(element, document)
            if not paragraphe.text.strip():
                markdown_lines.append("")
                continue

            style_name = (
                paragraphe.style.name
                if getattr(paragraphe, "style", None) and getattr(paragraphe.style, "name", None)
                else ""
            )
            style_lower = style_name.lower()
            style_normalized = (
                style_lower.replace(" ", "").replace("-", "").replace("_", "")
            )

            contenu = ""
            for run in paragraphe.runs:
                texte = run.text
                if not texte:
                    continue
                if run.bold and run.italic:
                    texte = f"***{texte}***"
                elif run.bold:
                    texte = f"**{texte}**"
                elif run.italic:
                    texte = f"*{texte}*"
                contenu += texte

            heading_level = None
            for i in range(1, 5):
                if f"heading{i}" in style_normalized or f"titre{i}" in style_normalized:
                    heading_level = i
                    break

            if heading_level:
                markdown_lines.append("#" * heading_level + f" {contenu}")
            elif "list bullet" in style_lower:
                markdown_lines.append(f"* {contenu}")
            else:
                markdown_lines.append(contenu)

            markdown_lines.append("")

        elif isinstance(element, CT_Tbl):
            logging.info("Tableau détecté dans le document source.")
            table = Table(element, document)
            lignes_tableau: list[list[str]] = []
            for row in table.rows:
                cellules = []
                for cell in row.cells:
                    textes_cellule: list[str] = []
                    for para in cell.paragraphs:
                        contenu_para = ""
                        for run in para.runs:
                            texte = run.text
                            if not texte:
                                continue
                            if run.bold and run.italic:
                                texte = f"***{texte}***"
                            elif run.bold:
                                texte = f"**{texte}**"
                            elif run.italic:
                                texte = f"*{texte}*"
                            contenu_para += texte
                        textes_cellule.append(contenu_para)
                    if not textes_cellule:
                        textes_cellule.append("")
                    cellules.append("<br>".join(textes_cellule))
                lignes_tableau.append(cellules)

            if lignes_tableau:
                entete = "| " + " | ".join(lignes_tableau[0]) + " |"
                separateur = "| " + " | ".join(["---"] * len(lignes_tableau[0])) + " |"
                markdown_lines.append(entete)
                markdown_lines.append(separateur)
                for ligne in lignes_tableau[1:]:
                    markdown_lines.append("| " + " | ".join(ligne) + " |")
                markdown_lines.append("")
                logging.info(
                    f"Tableau converti en Markdown :\n{entete}\n{separateur}\n..."
                )

    return "\n".join(markdown_lines).strip()


def analyser_docx(file_stream) -> Tuple[str, None]:
    """Analyse un fichier DOCX et retourne son contenu Markdown."""

    try:
        file_stream.seek(0)
        document = docx.Document(file_stream)
        markdown_content = _convertir_docx_en_markdown(document)
        return markdown_content, None
    except OpcError as e:
        logging.error(f"Fichier DOCX corrompu : {e}")
        return "", None
    except Exception as e:  # pragma: no cover - erreurs inattendues
        logging.error(f"Erreur inattendue sur DOCX : {e}", exc_info=True)
        return "", None


def analyser_pdf(file_stream) -> Tuple[str, None]:
    """Extrait le contenu textuel brut d'un PDF."""

    try:
        file_stream.seek(0)
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        return full_text, None
    except Exception as e:  # pragma: no cover - erreurs inattendues
        logging.error(f"Erreur inattendue sur PDF : {e}", exc_info=True)
        return "", None


def analyser_document(fichier) -> Tuple[str, None]:
    """Analyse un fichier importé et choisit la méthode appropriée."""

    filename = fichier.name.lower()
    if filename.endswith(".docx"):
        return analyser_docx(fichier)
    if filename.endswith(".pdf"):
        return analyser_pdf(fichier)
    return "", None

