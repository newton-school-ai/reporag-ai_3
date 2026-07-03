import logging
from dataclasses import dataclass

import tiktoken
from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
    parent_symbol: str
    language: str
    token_count: int


class SemanticChunker:
    """AST-aware semantic chunker."""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.parser = ASTParser()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk_source(
        self,
        source: str,
        file_path: str = "<memory>",
        language: str = "python",
    ) -> list[Chunk]:
        """Split source code into semantic chunks."""

        if not source.strip():
            return []

        tree = self.parser.parse(source, language)

        chunks: list[Chunk] = []

        # Process top-level nodes rather than the module root.
        for child in tree.root_node.children:
            self._walk(
                node=child,
                source=source,
                chunks=chunks,
                file_path=file_path,
                language=language,
                parent_symbol="",
            )

        return chunks

    def _walk(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Recursively split oversized AST nodes."""

        text = self._node_text(node, source)
        tokens = self.count_tokens(text)

        # Small enough -> emit directly
        if tokens <= self.max_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent_symbol=parent_symbol,
                    language=language,
                    token_count=tokens,
                )
            )
            return

        symbol = parent_symbol

        # Record enclosing function/class name.
        if node.type in (
            "function_definition",
            "class_definition",
            "decorated_definition",
        ):
            core = node

            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in (
                        "function_definition",
                        "class_definition",
                    ):
                        core = child
                        break

            name = self._child(core, "identifier")
            if name:
                symbol = self._node_text(name, source)

        # Leaf node -> fallback to line-based splitting.
        if not node.children:
            self._split_text(
                node=node,
                source=source,
                chunks=chunks,
                file_path=file_path,
                language=language,
                parent_symbol=symbol,
            )
            return

        # Otherwise recurse into children.
        for child in node.children:
            self._walk(
                node=child,
                source=source,
                chunks=chunks,
                file_path=file_path,
                language=language,
                parent_symbol=symbol,
            )

    def _split_text(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Fallback splitter for very large leaf nodes."""

        text = self._node_text(node, source)

        lines = text.splitlines()

        current: list[str] = []
        start_line = node.start_point.row + 1

        current_tokens = 0

        for line in lines:
            line_tokens = self.count_tokens(line)

            if current and current_tokens + line_tokens > self.max_tokens:
                chunk_text = "\n".join(current)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=start_line + len(current) - 1,
                        parent_symbol=parent_symbol,
                        language=language,
                        token_count=current_tokens,
                    )
                )

                start_line += len(current)
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

        if current:
            chunk_text = "\n".join(current)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=node.end_point.row + 1,
                    parent_symbol=parent_symbol,
                    language=language,
                    token_count=current_tokens,
                )
            )

    def _node_text(self, node: Node, source: str) -> str:
        """Extract node text using Tree-sitter byte offsets."""
        return source[node.start_byte : node.end_byte]

    def _child(self, node: Node, kind: str) -> Node | None:
        """Return the first child of the requested type."""
        for child in node.children:
            if child.type == kind:
                return child
        return None
