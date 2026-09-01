from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union

import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """
        Gathers all strings from this section's 'content' (ignoring nested sections).
        """
        texts = []
        for item in self.content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "table_text" in item:
                # Example if we had tables, but in PDF we won't parse tables by default
                texts.append(item["table_text"])
        return "\n".join(texts)

    def get_all_plain_text(self) -> str:
        """
        Recursively gathers text from this section and all nested subsections.
        """
        texts = [self.get_full_text()]
        for sub in self.subsections:
            texts.append(sub.get_all_plain_text())
        return "\n".join([txt for txt in texts if txt]).strip()

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        results = []
        lower_kw = keyword.lower()
        # If keyword is in title or content, add this entire section
        if lower_kw in self.title.lower() or lower_kw in self.get_full_text().lower():
            results.append(self.to_dict())
        # Then search subsections
        for sub in self.subsections:
            results.extend(sub.search(keyword))
        return results

    def search_by_regex(self, pattern: str) -> List[Dict[str, Any]]:
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Regex error: {e}")
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


def parse_pdf_to_sections(file_path: str) -> DocumentSection:
    pdf_reader = PyPDF2.PdfReader(file_path)

    root = DocumentSection(title="PDF Document Root", level=0)

    # 2) Iterate right away
    for page_index, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue

        page_section = DocumentSection(title=f"Page {page_index+1}", level=1)
        page_section.tags.append("page_section")

        lines = page_text.split("\n")
        for line in lines:
            line = line.strip()
            if line:
                page_section.content.append(line)

        root.subsections.append(page_section)

    return root