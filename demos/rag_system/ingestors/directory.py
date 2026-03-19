"""Directory ingestor.

Recursively loads all supported documents from a local directory and
splits them into chunks suitable for embedding and vector storage.

Supported formats:
    .pdf   — via PyPDFLoader
    .docx  — via Docx2txtLoader
    .md, .txt, .text  — via TextLoader

When an LLM is supplied, each document's text is passed through a
metadata-extraction chain to populate: title, author, summary, filename.
"""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseIngestor

# Project root: demos/rag_system/ingestors/ -> up 3 levels
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
    ".text": TextLoader,
}

# Characters of document text to feed into the extraction prompt.
# Enough to find title/author in a header without overloading the context.
_EXTRACT_CHARS = 3000

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a metadata extraction assistant. "
        "Extract structured metadata from the document text provided and return "
        "valid JSON only — no explanation, no markdown, no code fences. "
        "If a field cannot be determined, use null.",
    ),
    (
        "human",
        "Extract the following fields from this document:\n"
        "- title: the document title\n"
        "- author: the author name(s) as a single string\n"
        "- summary: a 1-2 sentence summary of the document\n\n"
        "Document text (first portion):\n{text}\n\n"
        "Return ONLY a JSON object with keys: title, author, summary.",
    ),
])


class DirectoryIngestor(BaseIngestor):
    """Ingestor that loads all supported files from a local directory."""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._llm = llm
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if llm is not None:
            self._extract_chain = _EXTRACT_PROMPT | llm | JsonOutputParser()
        else:
            self._extract_chain = None

    @property
    def source_type(self) -> str:
        return "Directory"

    def _extract_metadata(self, file: Path, docs: list[Document]) -> dict:
        """Run LLM extraction on the first portion of the document text."""
        combined = " ".join(d.page_content for d in docs)[:_EXTRACT_CHARS]
        try:
            result = self._extract_chain.invoke({"text": combined})
            return {
                "title": result.get("title") or None,
                "author": result.get("author") or None,
                "summary": result.get("summary") or None,
                "filename": file.name,
            }
        except Exception as exc:
            print(f"Warning: metadata extraction failed for {file.name}: {exc}")
            return {"filename": file.name}

    def load(self, source: str) -> list[Document]:
        """Load all supported documents under *source* and return chunks.

        Args:
            source: Path to a local directory containing documents.

        Returns:
            List of text chunks as LangChain Documents.
        """
        path = Path(source).expanduser()
        # Resolve relative paths against the project root, not the working directory
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        directory = path.resolve()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_chunks: list[Document] = []

        for file in sorted(directory.rglob("*")):
            loader_cls = _LOADERS.get(file.suffix.lower())
            if loader_cls is None:
                continue
            try:
                file_docs = loader_cls(str(file)).load()
            except Exception as exc:
                print(f"Warning: could not load {file.name}: {exc}")
                continue

            # Extract metadata via LLM if available, otherwise just add filename
            if self._extract_chain is not None:
                extra_meta = self._extract_metadata(file, file_docs)
            else:
                extra_meta = {"filename": file.name}

            for doc in file_docs:
                doc.metadata.update(extra_meta)

            chunks = self._splitter.split_documents(file_docs)

            # Prepend title and author to each chunk's text so they are baked
            # into the embedding and influence similarity search.
            title = extra_meta.get("title")
            author = extra_meta.get("author")
            if title or author:
                prefix_parts = []
                if title:
                    prefix_parts.append(f"Title: {title}")
                if author:
                    prefix_parts.append(f"Author: {author}")
                prefix = "\n".join(prefix_parts) + "\n\n"
                for chunk in chunks:
                    chunk.page_content = prefix + chunk.page_content

            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError(
                f"No supported documents found in {directory}. "
                f"Supported extensions: {', '.join(_LOADERS)}"
            )

        return all_chunks
