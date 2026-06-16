"""Token counting and text padding utilities.

Uses the same len(text)//4 heuristic as inference-perf's fallback estimator.
"""

import random
import string

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def tokens_to_chars(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


_CODE_WORDS = [
    "def", "class", "self", "return", "import", "from", "if", "else", "elif",
    "for", "while", "try", "except", "raise", "with", "as", "yield", "async",
    "await", "None", "True", "False", "lambda", "pass", "break", "continue",
    "not", "and", "or", "in", "is", "global", "nonlocal", "assert", "del",
    "print", "len", "range", "list", "dict", "set", "tuple", "str", "int",
    "float", "bool", "bytes", "type", "super", "property", "staticmethod",
    "classmethod", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "any", "all",
    "min", "max", "sum", "abs", "round", "hash", "id", "repr", "format",
    "open", "close", "read", "write", "append", "extend", "insert", "remove",
    "pop", "clear", "copy", "update", "keys", "values", "items", "get",
    "join", "split", "strip", "replace", "find", "index", "count", "lower",
    "upper", "title", "startswith", "endswith", "encode", "decode",
    "request", "response", "handler", "middleware", "router", "controller",
    "service", "repository", "model", "schema", "validator", "serializer",
    "config", "settings", "logger", "metric", "trace", "span", "context",
    "session", "token", "user", "auth", "permission", "role", "cache",
    "queue", "worker", "task", "job", "event", "signal", "hook", "plugin",
    "module", "package", "dependency", "version", "release", "deploy",
    "database", "table", "column", "row", "query", "index", "migration",
    "connection", "pool", "transaction", "commit", "rollback", "cursor",
]

_PROSE_WORDS = [
    "the", "function", "method", "should", "handle", "error", "cases",
    "properly", "when", "input", "is", "invalid", "or", "missing", "this",
    "implementation", "uses", "pattern", "to", "ensure", "that", "all",
    "edge", "are", "covered", "including", "null", "values", "empty",
    "strings", "negative", "numbers", "and", "overflow", "conditions",
    "module", "provides", "abstraction", "layer", "over", "underlying",
    "storage", "mechanism", "allowing", "easy", "swapping", "between",
    "different", "backends", "without", "changing", "business", "logic",
    "configuration", "loaded", "from", "environment", "variables", "with",
    "fallback", "defaults", "for", "local", "development", "each", "setting",
    "can", "be", "overridden", "via", "command", "line", "arguments",
    "performance", "critical", "section", "uses", "caching", "strategy",
    "reduce", "database", "queries", "cache", "invalidation", "happens",
    "through", "event", "driven", "approach", "using", "message", "queue",
    "authentication", "middleware", "validates", "JWT", "tokens", "against",
    "configured", "signing", "keys", "supports", "both", "symmetric",
    "asymmetric", "algorithms", "token", "refresh", "handled", "transparently",
]


def pad_to_tokens(text: str, target_tokens: int, rng: random.Random | None = None) -> str:
    current = estimate_tokens(text)
    if current >= target_tokens:
        return text
    needed_chars = (target_tokens - current) * CHARS_PER_TOKEN
    filler = _generate_filler(needed_chars, rng or random.Random(42))
    return text + "\n" + filler


def truncate_to_tokens(text: str, target_tokens: int) -> str:
    target_chars = tokens_to_chars(target_tokens)
    if len(text) <= target_chars:
        return text
    return text[:target_chars]


def generate_random_code(tokens: int, rng: random.Random, language: str = "python") -> str:
    target_chars = tokens_to_chars(tokens)
    lines = []
    chars = 0
    indent_level = 0
    while chars < target_chars:
        indent = "    " * indent_level
        kind = rng.choice(["func", "class", "assign", "call", "comment", "if", "for", "return", "blank"])
        if kind == "func":
            name = f"_{rng.choice(_CODE_WORDS)}_{rng.randint(0, 999)}"
            params = ", ".join(rng.choices(_CODE_WORDS, k=rng.randint(1, 4)))
            line = f"{indent}def {name}({params}):"
            indent_level = min(indent_level + 1, 3)
        elif kind == "class":
            name = f"{''.join(w.capitalize() for w in rng.choices(_CODE_WORDS, k=2))}"
            line = f"{indent}class {name}:"
            indent_level = min(indent_level + 1, 3)
        elif kind == "assign":
            var = f"_{rng.choice(_CODE_WORDS)}"
            val = rng.choice([f'"{rng.choice(_CODE_WORDS)}"', str(rng.randint(0, 9999)), "None", "True", "False", "[]", "{}"])
            line = f"{indent}{var} = {val}"
        elif kind == "call":
            func = rng.choice(_CODE_WORDS)
            args = ", ".join(rng.choices(_CODE_WORDS, k=rng.randint(0, 3)))
            line = f"{indent}{func}({args})"
        elif kind == "comment":
            words = " ".join(rng.choices(_PROSE_WORDS, k=rng.randint(4, 12)))
            line = f"{indent}# {words}"
        elif kind == "if":
            cond = f"{rng.choice(_CODE_WORDS)} {rng.choice(['==', '!=', '>', '<', 'is', 'in'])} {rng.choice(_CODE_WORDS)}"
            line = f"{indent}if {cond}:"
            indent_level = min(indent_level + 1, 3)
        elif kind == "for":
            var = rng.choice(_CODE_WORDS)
            iter_ = rng.choice(_CODE_WORDS)
            line = f"{indent}for {var} in {iter_}:"
            indent_level = min(indent_level + 1, 3)
        elif kind == "return":
            val = rng.choice(_CODE_WORDS)
            line = f"{indent}return {val}"
            indent_level = max(indent_level - 1, 0)
        else:
            line = ""
            indent_level = max(indent_level - 1, 0)

        lines.append(line)
        chars += len(line) + 1

    return "\n".join(lines)[:target_chars]


def generate_prose(tokens: int, rng: random.Random) -> str:
    target_chars = tokens_to_chars(tokens)
    return _generate_filler(target_chars, rng)


def _generate_filler(chars: int, rng: random.Random) -> str:
    words = []
    total = 0
    while total < chars:
        word = rng.choice(_PROSE_WORDS)
        words.append(word)
        total += len(word) + 1
        if rng.random() < 0.08:
            words.append("\n")
    return " ".join(words)[:chars]
