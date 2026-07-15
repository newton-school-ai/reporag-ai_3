import json
import re
from dataclasses import asdict, dataclass

from src.reporag.ingestion.symbol_extractor import Symbol


@dataclass
class SymbolRecord:
    symbol: Symbol
    qualified_name: str

    def to_dict(self):
        return {
            "symbol": asdict(self.symbol),
            "qualified_name": self.qualified_name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            symbol=Symbol(**data["symbol"]),
            qualified_name=data["qualified_name"],
        )


class SymbolTable:
    def __init__(self):
        self._records = {}
        self._name_index = {}
        self._file_index = {}

    def register_symbols(self, symbols):
        for sym in symbols:
            qname = self._build_qualified_name(sym)

            if qname in self._records:
                qname = f"{qname}_L{sym.start_line}"

            record = SymbolRecord(sym, qname)

            self._records[qname] = record
            self._name_index.setdefault(sym.name, []).append(record)
            self._file_index.setdefault(sym.file_path, []).append(record)

    def lookup(self, name):
        return self._name_index.get(name, [])

    def lookup_qualified(self, qualified_name):
        return self._records.get(qualified_name)

    def lookup_regex(self, pattern):
        regex = re.compile(pattern)
        return [
            record for qname, record in self._records.items() if regex.search(qname)
        ]

    def lookup_by_file(self, file_path):
        return self._file_index.get(file_path, [])

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {q: r.to_dict() for q, r in self._records.items()},
                f,
                indent=2,
            )

    def from_json(self, path):
        self.__init__()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for qname, value in data.items():
            record = SymbolRecord.from_dict(value)
            self._records[qname] = record
            self._name_index.setdefault(record.symbol.name, []).append(record)
            self._file_index.setdefault(record.symbol.file_path, []).append(record)

    def _build_qualified_name(self, symbol):
        module = symbol.file_path.removesuffix(".py").replace("/", ".")

        if module.endswith(".__init__"):
            module = module[:-9]

        if symbol.parent_class:
            return f"{module}.{symbol.parent_class}.{symbol.name}"

        return f"{module}.{symbol.name}"
