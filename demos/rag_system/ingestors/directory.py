"""Directory ingestor.

Recursively loads all supported documents from a local directory and
splits them into chunks suitable for embedding and vector storage.

Supported formats:
    .pdf   — via PyPDFLoader
    .docx  — via Docx2txtLoader
    .md, .txt, .text  — via TextLoader
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseIngestor

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
    ".text": TextLoader,
}


class DirectoryIngestor(BaseIngestor):
    """Ingestor that loads all supported files from a local directory."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def source_type(self) -> str:
        return "Directory"

    def load(self, source: str) -> list[Document]:
        """Load all supported documents under *source* and return chunks.

        Args:
            source: Path to a local directory containing documents.

        Returns:
            List of text chunks as LangChain Documents.
        """
        directory = Path(source).expanduser().resolve()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        docs: list[Document] = []
        for file in sorted(directory.rglob("*")):
            loader_cls = _LOADERS.get(file.suffix.lower())
            if loader_cls is None:
                continue
            try:
                loader = loader_cls(str(file))
                docs.extend(loader.load())
            except Exception as exc:
                # Skip unreadable files rather than aborting the whole batch
                print(f"Warning: could not load {file.name}: {exc}")

        if not docs:
            raise ValueError(
                f"No supported documents found in {directory}. "
                f"Supported extensions: {', '.join(_LOADERS)}"
            )

        return self._splitter.split_documents(docs)
