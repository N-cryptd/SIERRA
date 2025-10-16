"""Minimal YAML loader supporting the configuration file used in the tests."""

from __future__ import annotations

from typing import Any


def _split_items(text: str) -> list[str]:
    items = []
    depth = 0
    current = []
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == ',' and depth == 0:
            items.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        items.append(''.join(current).strip())
    return [item for item in items if item]


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'null':
        return None
    try:
        if '.' in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item) for item in _split_items(inner)]
    if text.startswith('{') and text.endswith('}'):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        result = {}
        for item in _split_items(inner):
            if ':' not in item:
                continue
            key, value = item.split(':', 1)
            result[_parse_scalar(key)] = _parse_value(value)
        return result
    return _parse_scalar(text)


def safe_load(stream: Any) -> Any:
    if hasattr(stream, 'read'):
        content = stream.read()
    else:
        content = stream
    lines = content.splitlines()
    stack: list[tuple[Any, int]] = [({}, -1)]

    for raw_line in lines:
        stripped = raw_line.split('#', 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(' '))
        while stack and indent <= stack[-1][1]:
            stack.pop()
        parent, _ = stack[-1]
        if ':' in stripped:
            key, value_text = stripped.split(':', 1)
            key = key.strip()
            value_text = value_text.strip()
            if not value_text:
                new_container = {}
                parent[key] = new_container
                stack.append((new_container, indent))
            else:
                parent[key] = _parse_value(value_text)
        else:
            raise ValueError(f"Unsupported YAML syntax: {raw_line}")

    return stack[0][0]


__all__ = ["safe_load"]
