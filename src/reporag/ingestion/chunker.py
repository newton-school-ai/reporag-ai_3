import logging
from dataclasses import dataclass

import tiktoken
from tree_sitter import Node

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of source code for vector embedding."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    parent_symbol: str
    language: str
    token_count: int


class SemanticChunker:
    """AST-aware code chunker that respects logical boundaries like functions and classes."""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.parser = ASTParser()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk_file(self, file_path: str, language: str | None = None) -> list[Chunk]:
        """Parse a file and split it into semantic chunks."""
        try:
            result = self.parser.parse_file(file_path, language=language)
            with open(file_path, encoding="utf-8") as f:
                source_lines = f.read().splitlines()
            return self._chunk_ast(
                result.root_node, source_lines, file_path, result.language
            )
        except Exception as e:
            logger.error(f"Failed to chunk {file_path}: {e}")
            return []

    def chunk_source(
        self, source: str, file_path: str = "<memory>", language: str = "python"
    ) -> list[Chunk]:
        """Parse source code string and split it into semantic chunks."""
        result = self.parser.parse(source, language=language)
        source_lines = source.splitlines()
        return self._chunk_ast(result.root_node, source_lines, file_path, language)

    def _chunk_ast(
        self, root_node: Node, source_lines: list[str], file_path: str, language: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # We will process the top level children of the module
        current_chunk_lines: list[str] = []
        current_start_line = -1
        current_end_line = -1
        current_tokens = 0

        def flush_current_chunk():
            nonlocal current_chunk_lines, current_start_line, current_end_line, current_tokens
            if current_chunk_lines:
                text = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        text=text,
                        file_path=file_path,
                        start_line=current_start_line,
                        end_line=current_end_line,
                        parent_symbol="",
                        language=language,
                        token_count=current_tokens,
                    )
                )
                current_chunk_lines = []
                current_start_line = -1
                current_end_line = -1
                current_tokens = 0

        for child in root_node.children:
            text = self._get_node_text(child, source_lines)
            tokens = self.count_tokens(text)

            # If this single node exceeds the max tokens, we need to split it
            if tokens > self.max_tokens:
                flush_current_chunk()
                # Split the large node
                self._split_large_node(child, source_lines, file_path, language, chunks)
            else:
                # If adding it exceeds max_tokens (plus a 10% leeway), flush the current chunk
                if (
                    current_tokens + tokens > self.max_tokens * 1.1
                    and current_chunk_lines
                ):
                    flush_current_chunk()

                if not current_chunk_lines:
                    current_start_line = child.start_point.row + 1
                current_chunk_lines.append(text)
                current_end_line = child.end_point.row + 1
                current_tokens += tokens

        flush_current_chunk()
        return chunks

    def _split_large_node(
        self,
        node: Node,
        source_lines: list[str],
        file_path: str,
        language: str,
        chunks: list[Chunk],
    ):
        """Split a large function/class by iterating over its internal block statements."""
        # Find the name and signature of this function/class
        parent_symbol = ""
        signature_lines = []

        if node.type in ("function_definition", "class_definition", "decorated_definition"):
            # For decorated, extract the core function/class inside
            core_node = node
            if node.type == "decorated_definition":
                for c in node.children:
                    if c.type in ("function_definition", "class_definition"):
                        core_node = c
                        break

            name_node = self._get_child_by_type(core_node, "identifier")
            if name_node:
                parent_symbol = self._get_node_text(name_node, source_lines)

            # Extract the signature (everything before the block)
            block_node = self._get_child_by_type(core_node, "block")
            if block_node:
                # The signature is from the start of the node to the start of the block
                sig_start = node.start_point.row
                sig_end = block_node.start_point.row - 1
                # Handle single line function where block is on the same line
                if sig_end < sig_start:
                    sig_end = sig_start
                signature_lines = source_lines[sig_start : sig_end + 1]
            else:
                block_node = node # fallback to just breaking up the node's children directly
        else:
            block_node = node

        # Now chunk the children of the block node
        current_chunk_lines = list(signature_lines) if signature_lines else []
        current_start_line = node.start_point.row + 1
        current_end_line = current_start_line

        # Determine the initial tokens from the signature
        base_tokens = self.count_tokens("\n".join(current_chunk_lines)) if current_chunk_lines else 0
        current_tokens = base_tokens

        def flush_large_chunk():
            nonlocal current_chunk_lines, current_start_line, current_end_line, current_tokens
            if current_chunk_lines and len(current_chunk_lines) > len(signature_lines):
                text = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        text=text,
                        file_path=file_path,
                        start_line=current_start_line,
                        end_line=current_end_line,
                        parent_symbol=parent_symbol,
                        language=language,
                        token_count=current_tokens,
                    )
                )
                current_chunk_lines = list(signature_lines) if signature_lines else []
                current_start_line = current_end_line + 1 # Approximate for the next chunk
                current_tokens = base_tokens

        children_to_process = block_node.children if block_node else []

        for child in children_to_process:
            text = self._get_node_text(child, source_lines)
            tokens = self.count_tokens(text)

            # If a single sub-statement is still too large, we just text-chunk it
            if tokens > self.max_tokens:
                flush_large_chunk()
                # Text fallback chunking
                self._text_fallback_chunk(
                    child, source_lines, file_path, language, parent_symbol, chunks, signature_lines
                )
                current_start_line = child.end_point.row + 2
                continue

            if current_tokens + tokens > self.max_tokens * 1.1 and len(current_chunk_lines) > len(signature_lines):
                flush_large_chunk()
                current_start_line = child.start_point.row + 1

            current_chunk_lines.append(text)
            current_end_line = child.end_point.row + 1
            current_tokens += tokens

        flush_large_chunk()

    def _text_fallback_chunk(
        self,
        node: Node,
        source_lines: list[str],
        file_path: str,
        language: str,
        parent_symbol: str,
        chunks: list[Chunk],
        signature_lines: list[str]
    ):
        """Naive text chunking as a last resort for massive single statements."""
        text = self._get_node_text(node, source_lines)
        lines = text.split("\n")

        current_chunk_lines = list(signature_lines) if signature_lines else []
        current_start_line = node.start_point.row + 1
        base_tokens = self.count_tokens("\n".join(current_chunk_lines)) if current_chunk_lines else 0
        current_tokens = base_tokens

        for line in lines:
            line_tokens = self.count_tokens(line)
            if current_tokens + line_tokens > self.max_tokens and len(current_chunk_lines) > len(signature_lines):
                c_text = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        text=c_text,
                        file_path=file_path,
                        start_line=current_start_line,
                        end_line=current_start_line + len(current_chunk_lines) - len(signature_lines) - 1,
                        parent_symbol=parent_symbol,
                        language=language,
                        token_count=current_tokens,
                    )
                )
                current_chunk_lines = list(signature_lines) if signature_lines else []
                current_start_line = current_start_line + len(current_chunk_lines) - len(signature_lines)
                current_tokens = base_tokens

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if len(current_chunk_lines) > len(signature_lines):
             chunks.append(
                 Chunk(
                     text="\n".join(current_chunk_lines),
                     file_path=file_path,
                     start_line=current_start_line,
                     end_line=node.end_point.row + 1,
                     parent_symbol=parent_symbol,
                     language=language,
                     token_count=current_tokens,
                 )
             )


    def _get_node_text(self, node: Node, source_lines: list[str]) -> str:
        """Extract the exact text for a node from the source lines."""
        start_row = node.start_point.row
        end_row = node.end_point.row

        if start_row == end_row:
            return source_lines[start_row][node.start_point.column:node.end_point.column]

        lines = []
        lines.append(source_lines[start_row][node.start_point.column:])
        for r in range(start_row + 1, end_row):
            lines.append(source_lines[r])
        lines.append(source_lines[end_row][:node.end_point.column])

        return "\n".join(lines)

    def _get_child_by_type(self, node: Node, child_type: str) -> Node | None:
        for child in node.children:
            if child.type == child_type:
                return child
        return None
