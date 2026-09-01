import io
import docx
from typing import Union, Optional
from src.parsing.parse_docx_with_sections import parse_docx_to_sections, parse_obj_to_sections
from src.parsing.parse_pdf_with_sections import parse_pdf_to_sections
from src.parsing.parse_xlsx_with_sections import parse_xlsx_to_sections


def parse_file(
    source: Union[str, io.BytesIO, docx.document.Document],
    ext: Optional[str] = None
):
    if isinstance(source, docx.document.Document):
        return parse_obj_to_sections(source)

    ext = ext.lower()
    if ext == '.docx' or ext == '.doc':
        return parse_docx_to_sections(source)
    elif ext == '.pdf':
        return parse_pdf_to_sections(source)
    elif ext == '.xlsx' or ext == '.xls':
        return parse_xlsx_to_sections(source)
    else:
        raise ValueError(f"Unsupported file format: {source}")