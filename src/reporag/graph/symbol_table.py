"""Global symbol table / registry.

Central lookup index. Given a symbol name, returns the defining file,
line range, type, and signature. Supports lookup by exact name, fully
qualified name, regex pattern, and file path.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from src.reporag.ingestion.symbol_extractor import Symbol

logger = logging.getLogger(__name__)


@dataclass
class SymbolRecord:
    symbol: Symbol
    qualified_name: str

    def to_dict(self) -> dict:
        return {"symbol": asdict(self.symbol), "qualified_name": self.qualified_name}

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolRecord":
        return cls(
            symbol=Symbol(**data["symbol"]), qualified_name=data["qualified_name"]
        )


class SymbolTable:
    def __init__(self):
        self._by_id: dict[str, SymbolRecord] = {}
        self._by_name: dict[str, list[SymbolRecord]] = defaultdict(list)
        self._by_file: dict[str, list[SymbolRecord]] = defaultdict(list)

    def register_symbols(self, symbols: list[Symbol]):
        for sym in symbols:
            module_name = self._file_to_module(sym.file_path)

            # Construct qualified name
            if sym.parent_class:
                qualified_name = f"{module_name}.{sym.parent_class}.{sym.name}"
            else:
                qualified_name = f"{module_name}.{sym.name}"

            # Disambiguate same-name symbols (e.g., multiple functions with same name in a file)
            base_qname = qualified_name
            collision_count = 1
            while qualified_name in self._by_id:
                # If there's a collision, disambiguate with line number
                qualified_name = f"{base_qname}_L{sym.start_line}"
                # If STILL collision (same line?), use counter
                if qualified_name in self._by_id:
                    qualified_name = f"{base_qname}_{collision_count}"
                    collision_count += 1

            record = SymbolRecord(symbol=sym, qualified_name=qualified_name)

            self._by_id[qualified_name] = record
            self._by_name[sym.name].append(record)
            self._by_file[sym.file_path].append(record)

    def lookup(self, name: str) -> list[SymbolRecord]:
        return self._by_name.get(name, [])

    def lookup_qualified(self, qualified_name: str) -> SymbolRecord | None:
        return self._by_id.get(qualified_name)

    def lookup_regex(self, pattern: str) -> list[SymbolRecord]:
        regex = re.compile(pattern)
        return [record for qname, record in self._by_id.items() if regex.search(qname)]

    def lookup_by_file(self, file_path: str) -> list[SymbolRecord]:
        return self._by_file.get(file_path, [])

    def to_json(self, file_path: str):
        data = {qname: record.to_dict() for qname, record in self._by_id.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def from_json(self, file_path: str):
        self._by_id.clear()
        self._by_name.clear()
        self._by_file.clear()

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        for qname, record_dict in data.items():
            record = SymbolRecord.from_dict(record_dict)
            self._by_id[qname] = record
            self._by_name[record.symbol.name].append(record)
            self._by_file[record.symbol.file_path].append(record)

    def _file_to_module(self, file_path: str) -> str:
        clean_path = file_path
        if clean_path.endswith("/__init__.py"):
            clean_path = clean_path[:-12]
        elif clean_path.endswith(".py"):
            clean_path = clean_path[:-3]

        return clean_path.replace("/", ".")
