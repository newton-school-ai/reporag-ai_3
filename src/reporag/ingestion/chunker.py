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

    DEFINITION_TYPES = (
        "function_definition",
        "class_definition",
        "decorated_definition",
    )

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

        # named_children skips punctuation/anonymous nodes ("(", ":", etc.)
        for child in tree.root_node.named_children:
            self._process_node(
                node=child,
                source=source,
                chunks=chunks,
                file_path=file_path,
                language=language,
                parent_symbol="",
            )

        return chunks

    def _process_node(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Emit a node as-is if it fits, otherwise split it semantically."""

        text = self._node_text(node, source)
        tokens = self.count_tokens(text)

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

        # Functions/classes are semantic units: never crawl into
        # identifier / parameters / punctuation. Split their BODY instead.
        if node.type in self.DEFINITION_TYPES:
            self._chunk_definition(
                node, source, chunks, file_path, language, parent_symbol
            )
            return

        # Non-definition oversized node (e.g. a big top-level if/try block):
        # descend into its named children rather than raw children.
        named = node.named_children
        if named:
            for child in named:
                self._process_node(
                    child, source, chunks, file_path, language, parent_symbol
                )
            return

        # No named children to descend into -> line-based fallback.
        self._split_text(node, source, chunks, file_path, language, parent_symbol)

    def _chunk_definition(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Split an oversized function/class by its body, keeping the signature intact."""

        core = node
        if node.type == "decorated_definition":
            unwrapped = self._unwrap_decorated(node)
            if unwrapped is not None:
                core = unwrapped

        name = self._child(core, "identifier")
        own_symbol = self._node_text(name, source) if name else ""
        full_symbol = (
            f"{parent_symbol}.{own_symbol}"
            if parent_symbol and own_symbol
            else (own_symbol or parent_symbol)
        )

        body = core.child_by_field_name("body")
        if body is None or not body.named_children:
            # Nothing meaningful to split on (e.g. empty body) -> fall back to lines.
            self._split_text(node, source, chunks, file_path, language, parent_symbol)
            return

        # Everything from the start of the definition up to the first body
        # statement is the "signature" (decorators, def/class line, bases, etc.)
        signature_end = body.named_children[0].start_byte
        signature_text = source[node.start_byte : signature_end].rstrip("\n")

        if core.type == "class_definition":
            # Class members (methods, nested classes, class vars) are the
            # semantic units here; recurse with the class name as context.
            for member in body.named_children:
                self._process_node(
                    member, source, chunks, file_path, language, full_symbol
                )
            return

        # function_definition: accumulate body statements under the token
        # budget, repeating the signature in every continuation chunk.
        self._chunk_body_statements(
            body_statements=body.named_children,
            signature_text=signature_text,
            source=source,
            chunks=chunks,
            file_path=file_path,
            language=language,
            parent_symbol=full_symbol,
        )

    def _chunk_body_statements(
        self,
        body_statements: list[Node],
        signature_text: str,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Accumulate whole statements until the budget is hit, then emit."""

        signature_tokens = self.count_tokens(signature_text)
        budget = max(self.max_tokens - signature_tokens, 1)

        current: list[Node] = []
        current_tokens = 0

        def emit(stmts: list[Node]) -> None:
            if not stmts:
                return
            body_text = "\n\n".join(self._node_text(s, source) for s in stmts)
            chunk_text = (
                f"{signature_text}\n\n{body_text}" if signature_text else body_text
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    file_path=file_path,
                    start_line=stmts[0].start_point.row + 1,
                    end_line=stmts[-1].end_point.row + 1,
                    parent_symbol=parent_symbol,
                    language=language,
                    token_count=self.count_tokens(chunk_text),
                )
            )

        for stmt in body_statements:
            stmt_tokens = self.count_tokens(self._node_text(stmt, source))

            # A single statement bigger than the whole budget: flush what we
            # have, then split that statement's own lines (still under the
            # repeated signature) instead of losing context.
            if stmt_tokens > budget:
                emit(current)
                current = []
                current_tokens = 0
                self._split_statement(
                    stmt,
                    source,
                    chunks,
                    file_path,
                    language,
                    parent_symbol,
                    signature_text,
                )
                continue

            if current and current_tokens + stmt_tokens > budget:
                emit(current)
                current = []
                current_tokens = 0

            current.append(stmt)
            current_tokens += stmt_tokens

        emit(current)

    def _split_statement(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
        signature_text: str,
    ) -> None:
        """Line-based fallback for a single oversized statement, signature repeated."""

        text = self._node_text(node, source)
        lines = text.splitlines()

        signature_tokens = self.count_tokens(signature_text)
        budget = max(self.max_tokens - signature_tokens, 1)

        current: list[str] = []
        start_line = node.start_point.row + 1
        current_tokens = 0

        def emit(batch: list[str], s_line: int, e_line: int) -> None:
            if not batch:
                return
            body_text = "\n".join(batch)
            chunk_text = (
                f"{signature_text}\n\n{body_text}" if signature_text else body_text
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    file_path=file_path,
                    start_line=s_line,
                    end_line=e_line,
                    parent_symbol=parent_symbol,
                    language=language,
                    token_count=self.count_tokens(chunk_text),
                )
            )

        for line in lines:
            line_tokens = self.count_tokens(line)

            if current and current_tokens + line_tokens > budget:
                emit(current, start_line, start_line + len(current) - 1)
                start_line += len(current)
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

        if current:
            emit(current, start_line, node.end_point.row + 1)

    def _split_text(
        self,
        node: Node,
        source: str,
        chunks: list[Chunk],
        file_path: str,
        language: str,
        parent_symbol: str,
    ) -> None:
        """Generic line-based fallback for non-definition leaf nodes (no signature to repeat)."""

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

    def _unwrap_decorated(self, node: Node) -> Node | None:
        """Return the function/class node wrapped by a decorated_definition."""
        for child in node.named_children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None

    def _node_text(self, node: Node, source: str) -> str:
        """Extract node text using Tree-sitter byte offsets."""
        return source[node.start_byte : node.end_byte]

    def _child(self, node: Node, kind: str) -> Node | None:
        """Return the first child of the requested type."""
        for child in node.children:
            if child.type == kind:
                return child
        return None
