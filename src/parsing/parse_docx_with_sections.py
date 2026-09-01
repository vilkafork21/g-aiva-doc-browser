from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _local_name(element: Any) -> str:
    """Return an XML element's local name without its namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _math_child(element: Any, name: str) -> Optional[Any]:
    """Return the first direct OMML child with the requested local name."""
    for child in element:
        if child.tag.startswith(f"{{{MATH_NS}}}") and _local_name(child) == name:
            return child
    return None


def _math_property_value(element: Any, property_name: str, default: str = "") -> str:
    """Read m:val from a property nested inside an OMML property node."""
    for descendant in element.iter():
        if (
            descendant.tag.startswith(f"{{{MATH_NS}}}")
            and _local_name(descendant) == property_name
        ):
            return descendant.get(f"{{{MATH_NS}}}val", default)
    return default


def _omml_to_text(element: Any) -> str:
    """Convert an Office Math (OMML) node into readable linear text.

    ``python-docx`` intentionally omits ``m:oMath`` content from
    ``Paragraph.text``. The output deliberately resembles an ordinary text
    formula (``F1``, ``a/b``, ``x^2``), because that is the format already
    handled by the existing downstream logic.
    """
    if element is None:
        return ""

    name = _local_name(element)

    if name == "t":
        return element.text or ""

    if name == "r":
        return "".join(
            descendant.text or ""
            for descendant in element.iter()
            if descendant.tag == f"{{{MATH_NS}}}t"
        )

    if name == "f":
        numerator = _omml_to_text(_math_child(element, "num"))
        denominator = _omml_to_text(_math_child(element, "den"))
        return f"({numerator})/({denominator})"

    if name in {"sSub", "sSup", "sSubSup"}:
        base = _omml_to_text(_math_child(element, "e"))
        subscript = _omml_to_text(_math_child(element, "sub"))
        superscript = _omml_to_text(_math_child(element, "sup"))
        result = base
        if subscript:
            result += subscript
        if superscript:
            result += f"^{superscript}"
        return result

    if name == "rad":
        degree = _omml_to_text(_math_child(element, "deg"))
        radicand = _omml_to_text(_math_child(element, "e"))
        return f"root({degree}, {radicand})" if degree else f"sqrt({radicand})"

    if name == "d":
        content = _omml_to_text(_math_child(element, "e"))
        begin = _math_property_value(element, "begChr", "(")
        end = _math_property_value(element, "endChr", ")")
        return f"{begin}{content}{end}"

    if name == "func":
        function_name = _omml_to_text(_math_child(element, "fName"))
        argument = _omml_to_text(_math_child(element, "e"))
        if argument.startswith("(") and argument.endswith(")"):
            return f"{function_name}{argument}"
        return f"{function_name}({argument})"

    if name == "nary":
        operator = _math_property_value(element, "chr", "∑")
        operand = _omml_to_text(_math_child(element, "e"))
        lower = _omml_to_text(_math_child(element, "sub"))
        upper = _omml_to_text(_math_child(element, "sup"))
        limits = f"_{lower}" if lower else ""
        limits += f"^{upper}" if upper else ""
        return f"{operator}{limits}({operand})"

    if name in {"limLow", "limUpp"}:
        base = _omml_to_text(_math_child(element, "e"))
        limit = _omml_to_text(_math_child(element, "lim"))
        marker = "_" if name == "limLow" else "^"
        return f"{base}{marker}{limit}"

    if name == "acc":
        content = _omml_to_text(_math_child(element, "e"))
        accent = _math_property_value(element, "chr", "^")
        return f"{accent}({content})"

    if name == "bar":
        content = _omml_to_text(_math_child(element, "e"))
        position = _math_property_value(element, "pos", "top")
        return f"underline({content})" if position == "bot" else f"overline({content})"

    if name == "m":
        rows = []
        for row in element:
            if _local_name(row) == "mr":
                cells = [
                    _omml_to_text(cell)
                    for cell in row
                    if _local_name(cell) == "e"
                ]
                rows.append(", ".join(cells))
        return f"matrix({'; '.join(rows)})"

    if name == "eqArr":
        return "\n".join(
            _omml_to_text(child)
            for child in element
            if _local_name(child) == "e"
        )

    # Property nodes only describe visual formatting and must not leak into
    # formula text. All other containers are concatenated in document order.
    if name.endswith("Pr") or name == "ctrlPr":
        return ""

    return "".join(_omml_to_text(child) for child in element)


def _word_xml_to_text(element: Any) -> str:
    """Extract Word text and OMML formulas while preserving document order."""
    if element is None:
        return ""

    if element.tag in {
        f"{{{MATH_NS}}}oMath",
        f"{{{MATH_NS}}}oMathPara",
    }:
        return _omml_to_text(element)

    if element.tag in {
        f"{{{WORD_NS}}}t",
        f"{{{WORD_NS}}}instrText",
    }:
        return element.text or ""

    if element.tag == f"{{{WORD_NS}}}tab":
        return "\t"
    if element.tag in {
        f"{{{WORD_NS}}}br",
        f"{{{WORD_NS}}}cr",
    }:
        return "\n"

    return "".join(_word_xml_to_text(child) for child in element)


def _contains_omml(element: Any) -> bool:
    """Return True only when the Word XML node contains an equation."""
    return any(
        descendant.tag in {
            f"{{{MATH_NS}}}oMath",
            f"{{{MATH_NS}}}oMathPara",
        }
        for descendant in element.iter()
    )


def _extract_paragraph_text(paragraph_element: Any) -> str:
    """Extract normal text and Word equations from a paragraph XML node."""
    return _word_xml_to_text(paragraph_element).strip()


class DocumentSection:
    def __init__(self, title: str = "", level: int = 0):
        self.id: str = str(uuid.uuid4())
        self.title: str = title
        self.level: int = level
        self.content: List[Union[str, Dict[str, Any]]] = []
        self.subsections: List[DocumentSection] = []
        self.tags: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "tags": self.tags,
            "content": self.content,
            "subsections": [sub.to_dict() for sub in self.subsections],
        }
        if self.level == 0:
            data["full_plain_text"] = self.get_all_plain_text()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(json_str: str) -> DocumentSection:
        def _from_dict(data: Dict[str, Any]) -> DocumentSection:
            section = DocumentSection(title=data.get("title", ""), level=data.get("level", 0))
            section.id = data.get("id", str(uuid.uuid4()))
            section.tags = data.get("tags", [])
            section.content = data.get("content", [])
            for sub_data in data.get("subsections", []):
                section.subsections.append(_from_dict(sub_data))
            return section

        data = json.loads(json_str)
        return _from_dict(data)

    def flatten(self, parent_titles: List[str] = None) -> List[Dict[str, Any]]:
        if parent_titles is None:
            parent_titles = []
        full_title = (
            " > ".join(parent_titles + [self.title]) if self.title else " > ".join(parent_titles)
        )
        flattened = [
            {
                "id": self.id,
                "full_title": full_title,
                "level": self.level,
                "tags": self.tags,
                "content": self.content,
            }
        ]
        for sub in self.subsections:
            flattened.extend(sub.flatten(parent_titles + [self.title]))
        return flattened

    def get_tree(self, indent: int = 0) -> str:
        result = "  " * indent + f"{self.title} (Level {self.level}, id: {self.id})\n"
        for sub in self.subsections:
            result += sub.get_tree(indent + 1)
        return result

    def print_tree(self, indent: int = 0):
        print(self.get_tree(indent))

    def get_full_text(self) -> str:
        texts = []
        for item in self.content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "table_text" in item:
                texts.append(item["table_text"])
        return "\n".join(texts)

    def get_all_plain_text(self) -> str:
        """
        Рекурсивно собирает полный текст из текущей секции и всех вложенных подразделов.
        """
        texts = [self.get_full_text()]
        for sub in self.subsections:
            texts.append(sub.get_all_plain_text())
        return "\n".join([txt for txt in texts if txt]).strip()

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        results = []
        lower_kw = keyword.lower()
        if lower_kw in self.title.lower() or lower_kw in self.get_full_text().lower():
            results.append(self.to_dict())
        for sub in self.subsections:
            results.extend(sub.search(keyword))
        return results

    def search_by_regex(self, pattern: str) -> List[Dict[str, Any]]:
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return results
        if regex.search(self.title) or regex.search(self.get_full_text()):
            results.append(self.to_dict())
        for sub in self.subsections:
            results.extend(sub.search_by_regex(pattern))
        return results

    def filter_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        results = []
        if tag in self.tags:
            results.append(self.to_dict())
        for sub in self.subsections:
            results.extend(sub.filter_by_tag(tag))
        return results

    def find_by_id(self, search_id: str) -> Optional[DocumentSection]:
        if self.id == search_id:
            return self
        for sub in self.subsections:
            found = sub.find_by_id(search_id)
            if found:
                return found
        return None


def iter_block_items(parent: Any):
    if isinstance(parent, docx.document.Document):
        document = parent
        parent_element = parent.element.body
    else:
        document = None
        parent_element = parent

    for child in parent_element.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def parse_table_with_colspan(table: Table) -> List[List[tuple]]:
    tbl_xml = table._tbl
    rows_data = []
    tr_elements = tbl_xml.xpath(".//w:tr")
    for tr in tr_elements:
        row_cells = []
        tc_elements = tr.xpath(".//w:tc")
        for tc in tc_elements:
            tcPr = tc.xpath("./w:tcPr")
            colspan = 1
            if tcPr:
                grid_span_el = tcPr[0].xpath("./w:gridSpan")
                if grid_span_el and grid_span_el[0].get("w:val"):
                    try:
                        colspan = int(grid_span_el[0].get("w:val"))
                    except ValueError:
                        colspan = 1
            paragraphs = tc.xpath(".//w:p")
            texts = []
            for p in paragraphs:
                if _contains_omml(p):
                    paragraph_text = _extract_paragraph_text(p)
                    if paragraph_text:
                        texts.append(paragraph_text)
                else:
                    # Preserve the original table parsing behavior exactly
                    # for paragraphs that do not contain Word equations.
                    text_nodes = p.xpath(".//w:t")
                    texts.extend([tn.text for tn in text_nodes if tn.text])
            cell_text = "\n".join(texts).strip()
            row_cells.append((cell_text, colspan))
        rows_data.append(row_cells)
    return rows_data


def expand_merged_cells(rows_data: List[List[tuple]]) -> List[List[str]]:
    max_cols = max(sum(colspan for _, colspan in row) for row in rows_data)
    expanded_table = []
    for row in rows_data:
        expanded_row = []
        for text, colspan in row:
            expanded_row.append(text)
            expanded_row.extend([""] * (colspan - 1))
        expanded_row.extend([""] * (max_cols - len(expanded_row)))
        expanded_table.append(expanded_row)
    return expanded_table


def parse_table(table: Table) -> List[List[str]]:
    rows_data = parse_table_with_colspan(table)
    return expand_merged_cells(rows_data)


def parse_table_as_dict(table: Table) -> List[Dict[str, Any]]:
    table_data = parse_table(table)
    if not table_data or len(table_data) < 2:
        return []
    headers = table_data[0]
    dict_rows = []
    for row in table_data[1:]:
        row_dict = {
            headers[i] if headers[i] else f"Column {i+1}": (row[i] if i < len(row) else "")
            for i in range(len(headers))
        }
        dict_rows.append(row_dict)
    return dict_rows


def _parse_docx_sections(doc: "docx.Document") -> DocumentSection:
    root = DocumentSection(title="", level=0)
    stack = [(0, root)]

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            # Keep the original python-docx path unless Word Math is present.
            text = (
                _extract_paragraph_text(block._p)
                if _contains_omml(block._p)
                else block.text.strip()
            )
            if not text:
                continue
            style_name = block.style.name if block.style else ""
            if style_name.startswith("Heading"):
                try:
                    level = int(style_name.split()[1])
                except (IndexError, ValueError):
                    level = 1
                new_section = DocumentSection(title=text, level=level)
                if level == 1:
                    new_section.tags.append("главный раздел")
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent_section = stack[-1][1]
                parent_section.subsections.append(new_section)
                stack.append((level, new_section))
            else:
                current_section = stack[-1][1]
                if current_section.content and isinstance(current_section.content[-1], str):
                    current_section.content[-1] += "\n" + text
                else:
                    current_section.content.append(text)
        elif isinstance(block, Table):
            table_data = parse_table(block)
            table_text = "\n".join(["\t".join(row) for row in table_data])
            table_dict = parse_table_as_dict(block)
            current_section = stack[-1][1]
            current_section.content.append({"table_text": table_text, "table_dict": table_dict})

    return root


def parse_docx_to_sections(file_path: str) -> DocumentSection:
    try:
        doc = docx.Document(file_path)
    except Exception:
        raise
    return _parse_docx_sections(doc)


def parse_obj_to_sections(doc: "docx.document.Document") -> DocumentSection:
    return _parse_docx_sections(doc)
