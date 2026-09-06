"""Layout-aware multimodal extraction for every PDF in a folder."""

import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import pdfplumber
from langchain_core.documents import Document


def _clean(value):
    """Normalize extracted PDF text and repair known arrow glyph corruption."""
    # Normalize whitespace and a common PDF symbol-font extraction error:
    # arrow glyphs in the supplied workflow PDF are decoded as standalone "fi".
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"(?<!\w)fi(?!\w)", "->", text)


def _table_to_markdown(table):
    """Preserve table row/column relationships in embedding-friendly Markdown."""
    rows = [[_clean(cell) for cell in row] for row in table if row]
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _page_lines(page):
    """Reconstruct visual text lines from positioned PDF words."""
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["size", "fontname"],
    )
    grouped = defaultdict(list)
    for word in words:
        # A six-point tolerance keeps symbol-font arrows on the same baseline
        # as their neighboring flow labels despite small vertical offsets.
        grouped[round(float(word["top"]) / 6) * 6].append(word)
    lines = []
    for top in sorted(grouped):
        line_words = sorted(grouped[top], key=lambda word: float(word["x0"]))
        text = _clean(" ".join(word["text"] for word in line_words))
        if text:
            lines.append({
                "text": text,
                "size": max(float(word.get("size", 0)) for word in line_words),
                "bold": any(
                    "bold" in word.get("fontname", "").lower() for word in line_words
                ),
            })
    return lines


def _heading_level(line, body_size):
    """Classify probable document titles and section headings by typography."""
    if len(line["text"]) > 120:
        return 0
    if line["size"] >= body_size * 1.55:
        return 1
    if line["size"] >= body_size * 1.22 or (
        line["bold"] and len(line["text"]) <= 80
    ):
        return 2
    return 0


def _text_sections(page, source, page_number):
    """Turn one PDF page into layout-aware header and section chunks."""
    lines = _page_lines(page)
    if not lines:
        return []
    body_size = median(line["size"] for line in lines if line["size"] > 0)
    document_title = Path(source).stem.replace("_", " ")
    header = document_title
    section = f"Page {page_number}"
    content = []
    documents = []

    def flush():
        """Emit accumulated body lines under the current header and section."""
        if content:
            text = "\n".join(content).strip()
            if text:
                documents.append(Document(
                    page_content=(
                        f"Document: {document_title}\nHeader: {header}\n"
                        f"Section: {section}\nPage: {page_number}\n\n{text}"
                    ),
                    metadata={
                        "header": header,
                        "section": section,
                        "source": str(source),
                        "page": page_number,
                        "content_type": "text",
                    },
                ))
            content.clear()

    for line in lines:
        level = _heading_level(line, body_size)
        if level:
            flush()
            if level == 1:
                header = line["text"]
                section = f"Page {page_number}"
            else:
                section = line["text"]
        else:
            content.append(line["text"])
    flush()
    return documents


def _visual_context(page_text):
    """Select captions when available, otherwise retain bounded nearby text."""
    lines = [_clean(line) for line in page_text.splitlines()]
    captions = [
        line for line in lines
        if re.search(r"\b(figure|fig\.?|chart|diagram|image|graph)\b", line, re.I)
    ]
    return "\n".join(captions) or _clean(page_text)[:1200]


def _flow_documents(page, source, page_number):
    """Extract ordered diagram nodes into high-confidence flow documents."""
    # Flow diagrams often expose their node labels and arrows as one text line.
    # Store that line separately so retrieval does not confuse surrounding prose
    # (business rules) with the ordered workflow shown by the diagram.
    lines = _page_lines(page)
    headings = [
        line["text"] for line in lines
        if re.match(r"^\d+\.\s+", line["text"])
    ]
    header = headings[0] if headings else Path(source).stem.replace("_", " ")
    documents = []
    # Layout-preserving extraction is better for diagrams because separated
    # boxes can have slightly different baselines even when visually aligned.
    flow_lines = [
        _clean(line) for line in (page.extract_text(layout=True) or "").splitlines()
    ]
    for flow_line in flow_lines:
        if flow_line.count("->") < 2:
            continue
        steps = [step.strip() for step in flow_line.split("->") if step.strip()]
        if len(steps) < 3:
            continue
        canonical_flow = " -> ".join(steps)
        documents.append(Document(
            page_content=(
                f"Document: {Path(source).stem}\nHeader: {header}\n"
                f"Section: Process flow\nPage: {page_number}\n\n"
                f"Exact flow from the diagram:\n{canonical_flow}\n"
                f"Ordered steps: {', '.join(steps)}."
            ),
            metadata={
                "header": header,
                "section": "Process flow",
                "source": str(source),
                "page": page_number,
                "content_type": "flow",
                "flow_steps": steps,
            },
        ))
    return documents


def load_multimodal_pdf(pdf_path):
    """Extract text, tables, flows, raster images, and vector charts from a PDF."""
    pdf_path = Path(pdf_path)
    documents = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            documents.extend(_text_sections(page, pdf_path, page_number))
            documents.extend(_flow_documents(page, pdf_path, page_number))

            # Tables become their own chunks so column/value relationships do
            # not get flattened into ambiguous page-level prose.
            for table_number, table in enumerate(page.extract_tables(), start=1):
                markdown = _table_to_markdown(table)
                if markdown:
                    documents.append(Document(
                        page_content=(
                            f"Document: {pdf_path.stem}\n"
                            f"Table {table_number}, page {page_number}:\n{markdown}"
                        ),
                        metadata={
                            "header": pdf_path.stem,
                            "section": f"Page {page_number} table {table_number}",
                            "source": str(pdf_path),
                            "page": page_number,
                            "content_type": "table",
                        },
                    ))

            visual_text = _visual_context(page_text)
            # Raster images are represented by searchable captions and nearby
            # text because the configured generation model is text-only.
            for image_number, image in enumerate(page.images, start=1):
                documents.append(Document(
                    page_content=(
                        f"Image {image_number}, page {page_number}, "
                        f"document {pdf_path.stem}. Dimensions: "
                        f"{image.get('width', 0):.0f} by {image.get('height', 0):.0f}. "
                        f"Caption or nearby explanation:\n{visual_text}"
                    ),
                    metadata={
                        "header": pdf_path.stem,
                        "section": f"Page {page_number} image {image_number}",
                        "source": str(pdf_path),
                        "page": page_number,
                        "content_type": "image",
                    },
                ))

            drawing_count = len(page.curves) + len(page.lines) + len(page.rects)
            has_visual_label = re.search(
                r"\b(chart|figure|graph|diagram|workflow|process|journey|trend|plot)\b",
                page_text,
                re.I,
            )
            # Many business PDFs construct diagrams and charts from vector
            # lines/rectangles, so they do not appear in page.images.
            if drawing_count >= 8 or has_visual_label:
                documents.append(Document(
                    page_content=(
                        f"Vector chart, workflow, or diagram on page {page_number} "
                        f"of {pdf_path.stem}. It contains {drawing_count} drawing "
                        f"objects. Caption or nearby explanation:\n{visual_text}"
                    ),
                    metadata={
                        "header": pdf_path.stem,
                        "section": f"Page {page_number} chart",
                        "source": str(pdf_path),
                        "page": page_number,
                        "content_type": "chart",
                    },
                ))
    return documents


def load_pdf_folder(folder):
    """Recursively load all PDFs so adding a document requires no code change."""
    folder = Path(folder)
    pdf_files = sorted(folder.rglob("*.pdf")) if folder.exists() else []
    documents = []
    for pdf_path in pdf_files:
        extracted = load_multimodal_pdf(pdf_path)
        documents.extend(extracted)
    return documents
