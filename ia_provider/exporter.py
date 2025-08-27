from __future__ import annotations
import io
import json
import logging
from typing import Any, Dict, List, Union

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from docx.section import _Header, _Footer
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.text.run import Run
import markdown as md
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MarkdownToDocxConverter:
    """Convertit du texte Markdown en éléments DOCX."""

    def __init__(self, document: Document, styles: Dict[str, Dict[str, Any]]):
        """Initialise le convertisseur avec un document et un dictionnaire de styles."""

        self.doc = document
        self.styles = styles or {}

    def _apply_style(
        self,
        run,
        style_overrides: Dict[str, Any] | None = None,
        *,
        style_name: str = "response",
    ) -> None:
        """Applique un style au ``run`` donné.

        ``style_name`` permet de sélectionner un style de base dans ``self.styles``.
        ``style_overrides`` peut être utilisé pour modifier certains attributs.
        """

        style = {**self.styles.get(style_name, {}), **(style_overrides or {})}

        if font_name := style.get("font_name"):
            run.font.name = font_name
            run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
        if size := style.get("font_size"):
            run.font.size = Pt(size)
        if color := style.get("font_color_rgb"):
            try:
                if isinstance(color, str):
                    run.font.color.rgb = RGBColor.from_string(color)
                else:
                    run.font.color.rgb = RGBColor(*color)
            except Exception:
                pass
        run.bold = style.get("is_bold", False)
        run.italic = style.get("is_italic", False)

    def _add_inline(self, paragraph, node) -> None:
        """Ajoute récursivement les noeuds inline à un paragraphe."""

        if isinstance(node, NavigableString):
            text = str(node)
            if text:
                run = paragraph.add_run(text)
                self._apply_style(run)
            return

        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if text:
                    run = paragraph.add_run(text)
                    self._apply_style(run)
            elif child.name in {"strong", "b"}:
                run = paragraph.add_run(child.get_text())
                self._apply_style(run, {"is_bold": True})
            elif child.name in {"em", "i"}:
                run = paragraph.add_run(child.get_text())
                self._apply_style(run, {"is_italic": True})
            elif child.name == "code":
                run = paragraph.add_run(child.get_text())
                self._apply_style(run)
                run.font.name = "Consolas"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            elif child.name == "a":
                text = child.get_text()
                href = child.get("href")
                if href:
                    run = self._add_hyperlink(paragraph, href, text)
                    self._apply_style(run)
                else:
                    run = paragraph.add_run(text)
                    self._apply_style(run)
            else:
                self._add_inline(paragraph, child)

    def _add_hyperlink(self, paragraph, url: str, text: str) -> Run:
        """Ajoute un hyperlien cliquable au paragraphe."""

        part = paragraph.part
        r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

        hyperlink = OxmlElement('w:hyperlink')
        hyperlink.set(qn('r:id'), r_id)

        new_run = OxmlElement('w:r')
        r_pr = OxmlElement('w:rPr')
        r_style = OxmlElement('w:rStyle')
        r_style.set(qn('w:val'), 'Hyperlink')
        r_pr.append(r_style)
        new_run.append(r_pr)

        text_elem = OxmlElement('w:t')
        text_elem.text = text
        new_run.append(text_elem)
        hyperlink.append(new_run)
        paragraph._p.append(hyperlink)

        return Run(new_run, paragraph)

    def _process_element(self, elem, list_style: str | None = None) -> None:
        """Traite les éléments HTML convertis depuis le Markdown."""

        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            if text:
                paragraph = (
                    self.doc.add_paragraph(style=list_style)
                    if list_style
                    else self.doc.add_paragraph()
                )
                run = paragraph.add_run(text)
                self._apply_style(run)
            return

        tag = elem.name
        if tag in {"p", "li"}:
            paragraph = (
                self.doc.add_paragraph(style=list_style)
                if list_style
                else self.doc.add_paragraph()
            )
            self._add_inline(paragraph, elem)
            for child in elem.find_all(["ul", "ol"], recursive=False):
                self._process_element(
                    child,
                    "List Bullet" if child.name == "ul" else "List Number",
                )
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            paragraph = self.doc.add_heading(level=level)
            self._add_inline(paragraph, elem)
        elif tag == "ul":
            for li in elem.find_all("li", recursive=False):
                self._process_element(li, "List Bullet")
        elif tag == "ol":
            for li in elem.find_all("li", recursive=False):
                self._process_element(li, "List Number")
        elif tag == "pre":
            code_text = "".join(elem.strings).strip()
            paragraph = self.doc.add_paragraph()
            run = paragraph.add_run(code_text)
            self._apply_style(run)
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        elif tag == "table":
            rows = elem.find_all("tr", recursive=False)
            if rows:
                first_row_cells = rows[0].find_all(["th", "td"], recursive=False)
                cols = len(first_row_cells)
                table = self.doc.add_table(rows=len(rows), cols=cols)
                for r_idx, row in enumerate(rows):
                    cells = row.find_all(["th", "td"], recursive=False)
                    for c_idx, cell in enumerate(cells):
                        paragraph = table.cell(r_idx, c_idx).paragraphs[0]
                        self._add_inline(paragraph, cell)
        else:
            text = elem.get_text(strip=True)
            if text:
                paragraph = self.doc.add_paragraph()
                self._add_inline(paragraph, elem)

    def add_markdown(self, text: str) -> None:
        """Convertit un texte Markdown et l'ajoute au document avec un fallback."""

        try:
            if not text:
                return

            md_converter = md.Markdown(extensions=["fenced_code", "tables"])
            html = md_converter.convert(text)

            soup = BeautifulSoup(html, "lxml")
            if soup.body:
                for elem in soup.body.find_all(recursive=False):
                    self._process_element(elem)
        except Exception as e:  # pragma: no cover - fallback branch
            warning_p = self.doc.add_paragraph()
            warning_run = warning_p.add_run(
                f"[Le formatage de ce bloc a échoué. Contenu original ci-dessous. Erreur : {e}]"
            )
            warning_run.font.italic = True
            warning_run.font.color.rgb = RGBColor(120, 120, 120)

            self.doc.add_paragraph(text)

def _appliquer_style_run(run, style: Dict):
    """Applique un dictionnaire de style à un objet Run."""
    if not style:
        return
    run.bold = style.get("is_bold")
    run.italic = style.get("is_italic")
    if style.get("font_name"):
        run.font.name = style["font_name"]
    if style.get("font_size"):
        run.font.size = Pt(int(style["font_size"]))
    if style.get("font_color_rgb"):
        try:
            run.font.color.rgb = RGBColor.from_string(style["font_color_rgb"])
        except ValueError:
            pass  # Ignore les couleurs mal formatées


def _reconstruire_blocs(parent: Union[Document, _Header, _Footer, _Cell], blocs_structure: List[Dict]):
    """Peuple un conteneur (document, header, etc.) avec le contenu structuré."""
    for bloc in blocs_structure:
        type_bloc = bloc.get("type", "paragraph")

        if type_bloc.startswith("heading"):
            niveau = int(type_bloc.split("_")[-1])
            p = parent.add_heading(level=niveau)
        elif type_bloc == "list":
            for item in bloc.get("items", []):
                parent.add_paragraph(item, style='List Bullet')
            continue
        elif type_bloc == "table":
            table_data = bloc.get("rows", [])
            if table_data:
                table = parent.add_table(rows=len(table_data), cols=len(table_data[0]))
                for i, row_data in enumerate(table_data):
                    for j, cell_structure in enumerate(row_data):
                        _reconstruire_blocs(table.cell(i, j), cell_structure)
            continue
        else:  # paragraph
            p = parent.add_paragraph()

        for run_data in bloc.get("runs", []):
            run = p.add_run(run_data.get("text", ""))
            _appliquer_style_run(run, run_data.get("style"))


def generer_export_docx(document_structure: Dict, styles_interface: Dict) -> io.BytesIO:
    """Génère un DOCX à partir d'une structure de document complète."""
    document = Document()

    # 1. Reconstruire l'en-tête et le pied de page
    if document_structure.get("header"):
        _reconstruire_blocs(document.sections[0].header, document_structure["header"])
    if document_structure.get("footer"):
        _reconstruire_blocs(document.sections[0].footer, document_structure["footer"])

    # 2. Insérer la Table des Matières
    p_tdm = document.add_paragraph()
    run = p_tdm.add_run()
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar)
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'
    run._r.append(instrText)
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar)

    # 3. Reconstruire le corps du document
    _reconstruire_blocs(document, document_structure.get("body", []))

    output = io.BytesIO()
    document.save(output)
    output.seek(0)
    return output


def generer_export_docx_markdown(texte_markdown: str, styles_interface: Dict) -> io.BytesIO:
    """Génère un DOCX à partir d'un texte Markdown (cas de repli pour PDF/erreurs)."""
    document = Document()
    converter = MarkdownToDocxConverter(document, styles_interface)
    converter.add_markdown(texte_markdown)

    output = io.BytesIO()
    document.save(output)
    output.seek(0)
    return output

