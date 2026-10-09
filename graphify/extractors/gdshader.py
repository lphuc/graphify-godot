"""Godot shader extractor (.gdshader, .gdshaderinc).

Godot's shading language is GLSL-shaped but its own dialect, and no grammar
targets it, so the file is read by regex. What the graph needs from a shader is
small:

* each function it defines -> a node, ``contains`` from the file node;
* ``#include "res://…"`` (or a path relative to the file) -> ``imports`` to the
  included file's node. It is the only way one shader file reaches another;
* a call to a function of this file, or of a file it includes (directly or
  through another include) -> ``calls``. Anything else called is the language's
  own (``mix``, ``texture``, ``vec3``) and yields no edge.

A shader's file node is minted like a scene's (``_resource_nid``), and its
functions are qualified by the suffix too: ``fog.gdshader`` usually sits beside
``fog.gd``, and both declare things the pipeline would otherwise file under the
one extensionless id ``fog``. Scripts reach the node through ``preload`` /
``load`` (``extractors/gdscript.py``), scenes and materials through
``[ext_resource type="Shader"]`` (``extractors/godot_resource.py``).
"""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import _file_stem, _make_id
from graphify.extractors.gdscript import _project_for, _resource_nid

# a string is matched first so the `//` of "res://…" is not read as a comment
_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|/\*[\s\S]*?\*/|//[^\n]*')
_INCLUDE_RE = re.compile(r'^[ \t]*#include[ \t]+"([^"\n]+)"', re.M)
# `vec3 tint(vec3 colour, float i) {` — the parameter list may wrap, but holds no
# `;`, `{` or `)`, so the match never runs across a statement into a later function
_FUNC_RE = re.compile(
    r"^[ \t]*(?:(?:lowp|mediump|highp)[ \t]+)?([A-Za-z_]\w*)[ \t]+([A-Za-z_]\w*)\s*\([^;{)]*\)\s*\{",
    re.M)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# `else if (…) {` reads as `<type> <name> (…) {`
_NOT_A_FUNCTION = frozenset({"if", "else", "for", "while", "switch", "return", "do"})


def _strip_comments(text: str) -> str:
    """Blank out comments, keeping every newline so line numbers still hold."""
    def blank(m: re.Match) -> str:
        token = m.group(0)
        return token if token.startswith('"') else "\n" * token.count("\n")
    return _COMMENT_RE.sub(blank, text)


def _read(path: Path) -> str | None:
    try:
        return _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def _functions(text: str) -> list[tuple[str, int, int]]:
    """(name, offset of the definition, offset of its opening brace), in file order."""
    return [(m.group(2), m.start(), m.end() - 1) for m in _FUNC_RE.finditer(text)
            if m.group(1) not in _NOT_A_FUNCTION and m.group(2) not in _NOT_A_FUNCTION]


def _includes(text: str, path: Path, root: Path | None) -> list[tuple[Path, int]]:
    """(included file, offset of the directive) for every include that exists."""
    found: list[tuple[Path, int]] = []
    for m in _INCLUDE_RE.finditer(text):
        raw = m.group(1)
        if raw.startswith("res://"):
            if root is None:
                continue
            target = root / raw[len("res://"):]
        else:
            target = path.parent / raw
        try:
            target = target.resolve()
            if not target.is_file():
                continue
        except OSError:
            continue
        found.append((target, m.start()))
    return found


def _included_functions(text: str, path: Path, root: Path | None, seen: set[Path]) -> dict[str, Path]:
    """Function name -> the file defining it, over everything ``path`` includes;
    a nearer include wins, as the first definition does for the compiler."""
    known: dict[str, Path] = {}
    for target, _at in _includes(text, path, root):
        if target in seen:
            continue
        seen.add(target)
        included = _read(target)
        if included is None:
            continue
        for name, _start, _brace in _functions(included):
            known.setdefault(name, target)
        for name, owner in _included_functions(included, target, root, seen).items():
            known.setdefault(name, owner)
    return known


def _body_end(text: str, open_brace: int) -> int:
    """Offset just past the ``}`` closing the block that opens at ``open_brace``."""
    depth = 0
    for at in range(open_brace, len(text)):
        if text[at] == "{":
            depth += 1
        elif text[at] == "}":
            depth -= 1
            if depth == 0:
                return at + 1
    return len(text)


def extract_gdshader(path: Path) -> dict:
    """Extract functions, their calls and ``#include`` edges from a Godot shader."""
    text = _read(path)
    if text is None:
        return {"nodes": [], "edges": [], "error": f"cannot read {path}"}

    project = _project_for(path)
    root = project.root if project is not None else None
    str_path = str(path)
    suffix = path.suffix.lstrip(".")
    file_nid = _resource_nid(path)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def line_at(offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    def add_edge(src: str, tgt: str, relation: str, offset: int,
                 target_file: Path | None = None) -> None:
        key = (src, tgt, relation)
        if key in seen_edges or src == tgt:
            return
        seen_edges.add(key)
        edge = {"source": src, "target": tgt, "relation": relation,
                "confidence": "EXTRACTED", "source_file": str_path,
                "source_location": f"L{line_at(offset)}", "weight": 1.0}
        if target_file is not None:
            # transient hint the pipeline reads to rewrite an id minted from
            # another file's absolute path into that file's canonical id
            edge["target_file"] = str(target_file)
        edges.append(edge)

    for target, at in _includes(text, path, root):
        add_edge(file_nid, _resource_nid(target), "imports", at, target)

    functions = _functions(text)
    own: dict[str, str] = {}
    for name, start, _brace in functions:
        if name in own:
            continue
        own[name] = nid = _make_id(_file_stem(path), suffix, name)
        nodes.append({"id": nid, "label": f"{name}()", "file_type": "code", "type": "function",
                      "source_file": str_path, "source_location": f"L{line_at(start)}"})
        add_edge(file_nid, nid, "contains", start)

    try:
        included = _included_functions(text, path, root, {path.resolve()})
    except OSError:
        included = {}
    for name, _start, brace in functions:
        for call in _CALL_RE.finditer(text, brace, _body_end(text, brace)):
            callee = call.group(1)
            if callee in own:
                add_edge(own[name], own[callee], "calls", call.start())
            elif callee in included:
                owner = included[callee]
                add_edge(own[name], _make_id(_file_stem(owner), owner.suffix.lstrip("."), callee),
                         "calls", call.start(), owner)

    return {"nodes": nodes, "edges": edges}
