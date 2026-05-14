"""
chunking.py  —  IMPROVED VERSION
----------------------------------
User Stories covered:
  US-04 : Document Chunking

Additional features added (add to Excel as extras):
  EXTRA-05 : Legal-aware chunk separators (respects numbered clauses)
  EXTRA-06 : Richer metadata per chunk (timestamp, word count, has_amounts)
  EXTRA-07 : Chunk preview in metadata (first 100 chars for debugging)
"""

import logging
import re
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


def split_into_chunks(
    text: str,
    source_filename: str = "unknown",
    file_type: str = "unknown",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split raw legal document text into overlapping LangChain Document chunks.

    WHY chunk_size=1000?
        ~150-200 words per chunk — enough to capture a full legal clause.

    WHY chunk_overlap=200?
        20% overlap prevents clauses that span chunk boundaries from being lost.

    WHY legal-aware separators? (EXTRA-05)
        Legal docs use numbered clauses like "4.1", "4.2.1" and ALL CAPS headings.
        Standard splitters cut through these. Legal separators respect boundaries.

    Args:
        text            : Full extracted text from parser.py
        source_filename : Original filename stored in metadata
        file_type       : 'pdf' or 'docx' — stored in metadata
        chunk_size      : Max characters per chunk (default 1000)
        chunk_overlap   : Shared characters between chunks (default 200)

    Returns:
        list[Document]: LangChain Documents with rich metadata
    """

    # ── Input validation ──────────────────────────────────────────────
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text. Check that extraction succeeded.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be >= 0 and < chunk_size ({chunk_size})"
        )

    logger.info(
        f"Chunking '{source_filename}' | "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    # ── Legal-aware separators  ────────────────────────────
    # Tried in order — splits on the most natural boundary first
    legal_separators = [
        "\n\n",             # paragraph break — strongest boundary
        "\n(?=\\d+\\.)",    # before numbered clause: "4." "4.1." "4.1.2."
        "\n(?=[A-Z]{2,})",  # before ALL CAPS headings: "TERMINATION", "WHEREAS"
        "\n",               # any line break
        ". ",               # sentence boundary
        " ",                # word boundary
        "",                 # character fallback
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=legal_separators,
        length_function=len,
        is_separator_regex=True,   
    )

    raw_chunks: list[str] = splitter.split_text(text)

    # ── Build Documents with rich metadata ──────
    upload_time = datetime.now().isoformat()

    documents: list[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    # Core metadata
                    "source": source_filename,
                    "file_type": file_type,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),

                    # Size info
                    "chunk_size_chars": len(chunk),
                    "word_count": len(chunk.split()),

                    # Content flags — useful for filtering
                    "has_monetary_amounts": bool(
                        re.search(r"₹|Rs\.?|\$|EUR|lakh|crore", chunk, re.IGNORECASE)
                    ),
                    "has_dates": bool(
                        re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}", chunk)
                    ),
                    "has_numbered_clause": bool(
                        re.search(r"^\d+\.\d*", chunk, re.MULTILINE)
                    ),

                    # Traceability
                    "upload_timestamp": upload_time,
                    "chunk_preview": chunk[:100].replace("\n", " "),  
                },
            )
        )

    avg_chars = sum(len(d.page_content) for d in documents) // len(documents)
    logger.info(
        f"Chunking complete: {len(documents)} chunks "
        f"(avg {avg_chars} chars/chunk)"
    )

    return documents