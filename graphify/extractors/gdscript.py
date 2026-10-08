"""GDScript extractor (tree-sitter-gdscript).

Godot 4 scripts. One file is one class: the node graph is the file node plus its
``func`` / ``signal`` / ``const`` / ``enum`` / inner ``class`` members, and edges
follow how a Godot project is actually wired together:

* ``extends`` -> ``inherits`` to the parent SCRIPT (a ``class_name`` or a
  ``res://`` path). An engine class (``Node``, ``RefCounted``) has no script and
  yields no edge.
* ``preload(...)`` / ``load(...)`` of a ``.gd`` -> ``imports`` to that script's
  file node. This is how a script with no ``class_name`` is reached at all. A
  preloaded ``.tscn`` / ``.tres`` -> ``imports`` to the scene / resource node
  ``extractors/godot_resource.py`` emits for that file.
* ``sig.connect(cb)`` / ``sig.emit(...)`` / ``emit_signal("sig", ...)`` ->
  ``uses`` from the enclosing function to the signal node; a connect also
  ``references`` its callback when that is a function of this file.
* a function used as a value — handed to any call (``tween_callback(_on_done)``),
  bound (``_on_done.bind(1)``), stored, or named in ``Callable(obj, "name")`` ->
  ``references`` to that function; so is a method of a typed receiver
  (``hud.show_x``, ``Sfx.play.bind("x")``). ``call("name")`` / ``call_deferred("name")``
  with a literal name is a ``calls`` edge like the call it stands for.
* calls: same-file functions and functions of an ancestor script resolve
  EXTRACTED; ``Autoload.method()`` resolves through the project's
  ``[autoload]`` table to that script's method (the receiver names the script
  in source, so it is exact); ``Alias.method()`` through a ``preload`` const
  alias; ``ClassName.method()`` / ``ClassName.new()`` through the project's
  ``class_name`` index. A call on a statically typed receiver resolves through
  the script its type names: an annotated member, parameter or local
  (``var hud: Hud``), one initialised with ``Hud.new()`` or cast with
  ``as Hud``, the element of an ``Array[Hud]``, the keys and values of a
  ``Dictionary[K, V]``, and any chain of typed members and typed function
  results (``hud.panel.refresh()``, ``get_hud().show()``). A type may be an
  inner class, nested or not (``Row``, ``Grid.Cell``, ``Deck.Hand.Card``), and a
  method of an inner class is walked in that class's own scope. A node path
  (``$Layer/Hud``, ``%Boss``, ``get_node("Hud")``, also through a string
  constant or an exported ``NodePath`` the scene sets) is typed by the script
  the project's scenes put on that node, reached through instanced and
  inherited scenes; ``get_parent()``, ``owner`` and ``find_child("Name")``
  move through the same trees, on self or chained (``owner.get_node("Boss")``).
  ``get_node("/root/Name")`` and ``get_tree().root.get_node("Name")`` are the
  autoload of that name; ``get_tree().current_scene`` is the main scene or one
  a scene-changing script names. A member with no script type of its own takes
  the type of what the class assigns to it (``_hud = get_node(hud_path)``), and
  a bare exported node reference the node a scene fills it with.
  An edge is EXTRACTED when there is no doubt what the receiver is, and INFERRED
  — one edge per candidate — when the scenes a script runs in differ, when
  something untyped is also assigned to the member, or for ``current_scene``.
  Locals follow lexical scope, and an
  untyped local hides a typed member of the same name. A call resolved by none
  of those is an engine builtin or a
  dynamic dispatch and is NOT handed to the shared name-matching resolver —
  counted in ``unresolved_calls`` instead.
* ``<page>.md §N.N`` in a comment -> ``references`` from the enclosing function
  (or the file) to a section node of that documentation page. A bare ``§N.N``
  inherits the last page a comment in the same file named (INFERRED).

The project index (autoloads, ``class_name`` -> script, ``uid://`` -> script,
per-script ``func`` names) is built once per ``project.godot`` root and cached
per process; a file outside any Godot project still extracts, with only the
same-file resolutions.
"""
from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Any, NamedTuple

from graphify.extractors.base import _file_stem, _make_id, _read_text

_RESOURCE_SUFFIXES = (".tscn", ".tres")
_ENGINE_BUILTIN_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_CITATION_RE = re.compile(r"([A-Za-z0-9_./-]+\.md)`?\s*§\s*(\d+(?:\.\d+)*[a-z]?)")
_BARE_CITATION_RE = re.compile(r"§\s*(\d+(?:\.\d+)*[a-z]?)")
_CLASS_NAME_RE = re.compile(r"^class_name\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_EXTENDS_RE = re.compile(      # `extends X`, or the one-line `class_name Foo extends X`
    r'^(?:class_name\s+[A-Za-z_][A-Za-z0-9_]*\s+)?extends\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))',
    re.M)
_FUNC_RE = re.compile(r"^(?:static\s+)?func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
_SIGNAL_RE = re.compile(r"^signal\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_AUTOLOAD_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="\*?((?:res|uid)://[^"]+)"', re.M)
_UID_RE = re.compile(r"^(uid://[a-z0-9]+)\s*$", re.M)

# What a class declares about types, read by regex like the rest of the index:
# top-level `var` lines, `func ... -> T:` signatures, preload consts, inner classes.
_MEMBER_RE = re.compile(      # name, then the rest of its line — and of the next, after an open `(`
    r"^(?:@\w+(?:\([^)\n]*\))?[ \t]+)*(?:static[ \t]+)?var[ \t]+([A-Za-z_]\w*)[ \t]*"
    r"((?:[^\n]*\([ \t]*\n)?[^\n]*)", re.M)
_ANNOTATION_RE = re.compile(r":[ \t]*([A-Za-z_][\w.]*(?:\[[\w., ]+\])?)")
_NEW_RE = re.compile(r":?=[ \t]*([A-Za-z_][\w.]*)\.new\([^()\n]*\)[ \t]*(?:#.*)?$")
_CAST_RE = re.compile(r"\bas[ \t]+([A-Za-z_][\w.]*)[ \t]*(?:#.*)?$")
# a value that is exactly one node of the scene: `$A/B`, `$"A/B"`, `%Name`,
# `get_node("A/B")` or `get_node(name)` (kept as `@name`), optionally cast
_NODE_REF_RE = re.compile(
    r'=[ \t]*(?:\$"([^"\n]+)"|\$([\w/%]+)|(%\w+)|get_node(?:_or_null)?\(\s*\^?"([^"\n]+)"\s*\)'
    r"|get_node(?:_or_null)?\(\s*(?P<name>[A-Za-z_]\w*)\s*\))"
    r"[ \t]*(?:as[ \t]+[\w.]+[ \t]*)?(?:#.*)?$")
# a path literal a `var` starts as: the default of an exported NodePath
_PATH_DEFAULT_RE = re.compile(r'=[ \t]*(?:[\^&]|NodePath\()?"([^"\n]*)"\)?[ \t]*(?:#.*)?$')
# `name = value` inside a function: what a member is given after its declaration
_ASSIGN_RE = re.compile(r"^[ \t]+(?:self\.)?([A-Za-z_]\w*)[ \t]*(=(?!=)[^\n]*)", re.M)
_NULL_RE = re.compile(r"=[ \t]*null[ \t]*(?:#.*)?$")
# a call depending on which scene the script runs in names at most this many targets
_MAX_CANDIDATES = 8
# a string constant: what a `get_node(SOME_PATH)` is given
_STRING_CONST_RE = re.compile(
    r'^const[ \t]+([A-Za-z_]\w*)[ \t]*(?::[ \t]*\w+[ \t]*)?:?=[ \t]*(?:[\^&]|NodePath\()?"([^"\n]*)"\)?'
    r"[ \t]*(?:#.*)?$", re.M)
# the parameter list may span lines and nest parentheses three deep
# (`at := Vector2(maxf(1.0, absf(x)), 0.0)`); it ends at its own `)`, so a
# signature with no return type never borrows the `-> T` of the function after it
_PARAMETERS = r"[^()]"
for _depth in range(3):
    _PARAMETERS = rf"(?:[^()]|\({_PARAMETERS}*\))"
_RETURN_RE = re.compile(
    r"^(?:static[ \t]+)?func[ \t]+([A-Za-z_]\w*)[ \t]*\(" + _PARAMETERS + r"*\)[ \t]*->[ \t]*"
    r"([A-Za-z_][\w.]*(?:\[[\w., ]+\])?)[ \t]*:", re.M)
_ALIAS_RE = re.compile(
    r"^const[ \t]+([A-Za-z_]\w*)[ \t]*(?::[ \t]*\w+[ \t]*)?:?=[ \t]*(?:preload|load)"
    r"""\([ \t]*["']([^"'\n]+\.gd)["'][ \t]*\)""", re.M)
# `class Name [extends Base]:` and the indented block under it
_INNER_RE = re.compile(
    r'^class[ \t]+([A-Za-z_]\w*)(?:[ \t]+extends[ \t]+("[^"\n]+"|[\w.]+))?[ \t]*:[^\n]*\n'
    r"((?:[ \t]+[^\n]*(?:\n|$)|[ \t]*\n|#[^\n]*\n)*)", re.M)
# like _EXTENDS_RE, but keeping a dotted base (`extends Grid.Cell`) whole
_CLASS_EXTENDS_RE = re.compile(
    r'^(?:class_name[ \t]+\w+[ \t]+)?extends[ \t]+(?:"([^"\n]+)"|([A-Za-z_][\w.]*))', re.M)
_CONTAINER_RE = re.compile(r"^(Array|Dictionary)\[(.+)\]$")
# methods that name the method they call in a string: obj.call("name", ...)
_DYNAMIC_CALLS = ("call", "call_deferred", "callv")

_PROJECT_CACHE: dict[Path, "_GodotProject | None"] = {}
_FILE_INDEX_CACHE: dict[Path, tuple[frozenset[str], frozenset[str], str | None]] = {}
_TYPE_INDEX_CACHE: dict[tuple[Path, str], "_ClassIndex"] = {}


def _resource_nid(path: Path) -> str:
    """File node id of a scene / resource / project file (``.tscn`` / ``.tres`` /
    ``project.godot``), as ``extractors/godot_resource.py`` mints it.

    Deliberately NOT the plain file id (``_make_id(str(path))``): the pipeline
    rewrites that to an extensionless id, and ``hud.tscn`` usually sits beside
    ``hud.gd``. Both would claim ``hud``, the pipeline would salt the two apart
    by path, and every ``inherits`` / ``imports`` edge aimed at the script would
    be left dangling. Qualified like a symbol of the file, it is rewritten to a
    root-relative id of its own (``scene/hud.tscn`` -> ``scene_hud_tscn_file``).
    """
    return _make_id(_file_stem(path), path.suffix.lstrip("."), "file")


def _project_files(root: Path, suffixes: tuple[str, ...]):
    """Every file with one of ``suffixes`` under ``root`` that belongs to THIS project.

    The engine does not look inside a directory holding ``.gdignore`` (its own
    ``.godot`` cache, the Android build template, a staged addon update), and a
    nested ``project.godot`` starts another project. A copy of a script left in
    either must not claim a ``class_name`` or a ``uid://`` of the real one.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here != root and (".gdignore" in filenames or "project.godot" in filenames):
            dirnames[:] = []
            continue
        if ".godot" in dirnames:
            dirnames.remove(".godot")
        for name in filenames:
            if name.endswith(suffixes):
                yield here / name


class _GodotProject:
    """What the extractor knows about the Godot project a script belongs to."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.autoloads: dict[str, Path] = {}
        self.class_names: dict[str, Path] = {}
        # always present: a project.godot that cannot be read returns early below,
        # and resolve() must still answer None for a uid:// rather than raise
        self._uids: dict[str, Path] = {}
        uids = self._uids
        try:
            for script in sorted(_project_files(root, (".gd", ".gd.uid"))):
                try:
                    head = script.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if script.suffix == ".uid":
                    m = _UID_RE.search(head)
                    if m:
                        uids[m.group(1)] = script.with_suffix("")
                    continue
                m = _CLASS_NAME_RE.search(head)
                if m and m.group(1) not in self.class_names:
                    self.class_names[m.group(1)] = script
            text = (root / "project.godot").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        section = text.split("[autoload]", 1)
        if len(section) == 2:
            body = section[1].split("\n[", 1)[0]
            for m in _AUTOLOAD_RE.finditer(body):
                target = self.resolve(m.group(2), uids)
                if target is not None:
                    self.autoloads[m.group(1)] = target

    def resolve(self, ref: str, uids: dict[str, Path] | None = None) -> Path | None:
        """A ``res://`` or ``uid://`` reference as an absolute path, or None."""
        if ref.startswith("res://"):
            return self.root / ref[len("res://"):]
        if ref.startswith("uid://"):
            return (uids if uids is not None else self._uids).get(ref)
        return None


def _project_for(path: Path) -> _GodotProject | None:
    """The nearest enclosing ``project.godot``, indexed once per process."""
    try:
        start = path.resolve().parent
    except OSError:
        start = path.parent
    for candidate in (start, *start.parents):
        if candidate in _PROJECT_CACHE:
            return _PROJECT_CACHE[candidate]
        if (candidate / "project.godot").is_file():
            project = _GodotProject(candidate)
            _PROJECT_CACHE[candidate] = project
            return project
    _PROJECT_CACHE[start] = None
    return None


def _file_index(script: Path) -> tuple[frozenset[str], frozenset[str], str | None]:
    """(func names, signal names, extends spec) of a script, by regex, cached."""
    # looked up thousands of times per file: answer a path already seen without
    # touching the filesystem (Path.resolve() is slow, on Windows above all)
    hit = _FILE_INDEX_CACHE.get(script)
    if hit is not None:
        return hit
    key = script.resolve() if script.exists() else script
    hit = _FILE_INDEX_CACHE.get(key)
    if hit is not None:
        _FILE_INDEX_CACHE[script] = hit
        return hit
    try:
        text = script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    m = _EXTENDS_RE.search(text)
    ext = (m.group(1) or m.group(2)) if m else None
    result = (frozenset(_FUNC_RE.findall(text)), frozenset(_SIGNAL_RE.findall(text)), ext)
    _FILE_INDEX_CACHE[key] = _FILE_INDEX_CACHE[script] = result
    return result


class _ClassIndex(NamedTuple):
    """What one class — a script, or an inner class of it — declares."""
    members: dict[str, tuple[str | None, str | None]]   # var -> (type name, scene node path)
    returns: dict[str, str]         # func -> return type name
    aliases: dict[str, str]         # const -> preloaded script reference
    inners: dict[str, str]          # inner class -> its body, dedented
    funcs: frozenset[str]
    signals: frozenset[str]
    extends: str | None
    strings: dict[str, str]         # const -> the string it holds
    defaults: dict[str, str]        # var -> the path literal it starts as
    # var -> (type name, node reference, nothing else is ever assigned to it)
    assigned: dict[str, tuple[str | None, str | None, bool]]


def _index_text(text: str) -> _ClassIndex:
    def value_of(rest: str, annotation: bool) -> tuple[str | None, str | None]:
        """(type name, node reference) a declaration or assignment gives a variable."""
        typed = ((_ANNOTATION_RE.match(rest) if annotation else None)
                 or _NEW_RE.match(rest) or _CAST_RE.search(rest))
        node = _NODE_REF_RE.search(rest)
        ref = None
        if node:
            ref = "@" + node.group("name") if node.group("name") else next(g for g in node.groups() if g)
        return (typed.group(1) if typed else None, ref)

    members: dict[str, tuple[str | None, str | None]] = {}
    defaults: dict[str, str] = {}
    for name, rest in _MEMBER_RE.findall(text):
        members[name] = value_of(rest, True)
        default = _PATH_DEFAULT_RE.search(rest)
        if default:
            defaults[name] = default.group(1)
    # one thing assigned to a member, wherever it is assigned, types it; two
    # different ones do not. `= null` clears it and says nothing.
    given: dict[str, set] = {}
    for name, rest in _ASSIGN_RE.findall(text):
        if name in members and not _NULL_RE.match(rest):
            given.setdefault(name, set()).add(value_of(rest, False))
    assigned = {}
    for name, values in given.items():
        typed = values - {(None, None)}
        if len(typed) == 1:
            assigned[name] = (*next(iter(typed)), len(values) == 1)
    inners: dict[str, str] = {}
    for name, base, body in _INNER_RE.findall(text):
        # a base named on the `class` line reads like a leading `extends` statement
        inners[name] = (f"extends {base}\n" if base else "") + textwrap.dedent(body)
    m = _CLASS_EXTENDS_RE.search(text)
    return _ClassIndex(
        members, dict(_RETURN_RE.findall(text)), dict(_ALIAS_RE.findall(text)), inners,
        frozenset(_FUNC_RE.findall(text)), frozenset(_SIGNAL_RE.findall(text)),
        (m.group(1) or m.group(2)) if m else None, dict(_STRING_CONST_RE.findall(text)),
        defaults, assigned)


def _type_index(script: Path, inner: str = "") -> _ClassIndex:
    """The declarations of ``script``, or of its inner class ``inner`` (``A``, or
    ``A/B`` for a class nested in it), by regex, cached. Type names are kept as
    written: what they stand for depends on the class that wrote them.

    A member is typed by its annotation (``var hud: Hud``), or else by a
    ``:= Hud.new()`` initialiser or a trailing ``as Hud`` cast; one initialised
    with a node path (``= $Layer/Hud``) also records that path.
    """
    # looked up thousands of times per file: answer a path already seen without
    # touching the filesystem (Path.resolve() is slow, on Windows above all)
    hit = _TYPE_INDEX_CACHE.get((script, inner))
    if hit is not None:
        return hit
    key = script.resolve() if script.exists() else script
    hit = _TYPE_INDEX_CACHE.get((key, inner))
    if hit is None:
        if inner:
            outer, _, name = inner.rpartition("/")
            text = _type_index(key, outer).inners.get(name, "")
        else:
            try:
                text = key.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        hit = _TYPE_INDEX_CACHE[(key, inner)] = _index_text(text)
    _TYPE_INDEX_CACHE[(script, inner)] = hit
    return hit


def _nested_class(script: Path, inner: str, names: list[str]) -> tuple[Path, str] | None:
    """The class reached from ``(script, inner)`` by the inner class names
    ``names`` in turn, or None when one of them is not declared there."""
    for name in names:
        if name not in _type_index(script, inner).inners:
            return None
        inner = f"{inner}/{name}" if inner else name
    return (script, inner)


def _named_class(name: str, ctx: tuple[Path, str],
                 project: "_GodotProject | None") -> tuple[Path, str] | None:
    """The class — (script, inner class path or "") — a name written inside class
    ``ctx`` stands for: an inner class of ``ctx`` or of a class enclosing it, an
    inner class or a preload const of ``ctx``'s script or of a script it
    extends, else a project ``class_name``; ``Outer.Inner`` names an inner class
    of whatever ``Outer`` is. None for an engine class."""
    script, scope = ctx
    if name.startswith(("res://", "uid://")) or name.endswith(".gd"):
        target = _script_of(name, script, project)
        return (target, "") if target is not None else None
    head, *tail = name.split(".")
    while scope:
        # visible from inside an inner class: what it and its enclosing classes nest
        if head in _type_index(script, scope).inners:
            return _nested_class(script, scope, [head, *tail])
        scope = scope.rpartition("/")[0]
    target = None
    for holder in (script, *_ancestors(script, project)):
        index = _type_index(holder)
        if head in index.aliases:
            target = _script_of(index.aliases[head], holder, project)
            break
        if head in index.inners:
            return _nested_class(holder, "", [head, *tail])
    else:
        if project is not None:
            target = project.class_names.get(head)
    return _nested_class(target, "", tail) if target is not None else None


def _class_chain(cls: tuple[Path, str], project: "_GodotProject | None",
                 limit: int = 16) -> list[tuple[Path, str]]:
    """``cls`` and the classes it extends, nearest first."""
    out: list[tuple[Path, str]] = []
    current: tuple[Path, str] | None = cls
    while current is not None and current not in out and len(out) < limit:
        out.append(current)
        base = _type_index(*current).extends
        current = _named_class(base, current, project) if base else None
    return out


def _declared(cls: tuple[Path, str], project: "_GodotProject | None", table: str, name: str):
    """(declaring class, entry) of ``name`` in ``table`` (a _ClassIndex field) of
    ``cls`` or of the nearest class it extends; (None, None) when undeclared."""
    for holder in _class_chain(cls, project):
        entries = getattr(_type_index(*holder), table)
        if name in entries:
            return holder, (entries[name] if isinstance(entries, dict) else name)
    return None, None


def _script_of(spec: str, script: Path, project: _GodotProject | None) -> Path | None:
    """The script an ``extends`` / reference spec names, or None (engine class)."""
    if spec.startswith("res://") or spec.startswith("uid://"):
        return project.resolve(spec) if project else None
    if spec.endswith(".gd"):
        return (script.parent / spec).resolve()
    if project and spec in project.class_names:
        return project.class_names[spec]
    return None


def _ancestors(script: Path, project: _GodotProject | None, limit: int = 16) -> list[Path]:
    """Ancestor scripts of ``script``, nearest first, following ``extends``."""
    out: list[Path] = []
    seen = {script}
    current = script
    while len(out) < limit:
        _funcs, _signals, spec = _file_index(current)
        if not spec:
            break
        parent = _script_of(spec, current, project)
        if parent is None or parent in seen:
            break
        seen.add(parent)
        out.append(parent)
        current = parent
    return out


def extract_gdscript(path: Path) -> dict:
    """Extract classes, functions, signals, constants, enums, imports, signal
    wiring, calls and documentation citations from a .gd file."""
    try:
        import tree_sitter_gdscript as tsgd
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_gdscript not installed"}

    try:
        language = Language(tsgd.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    project = _project_for(path)
    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    unresolved_calls: list[dict] = []

    def add_node(nid: str, label: str, line: int, kind: str, **extra: Any) -> None:
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        node = {"id": nid, "label": label, "file_type": "code", "type": kind,
                "source_file": str_path, "source_location": f"L{line}"}
        node.update(extra)
        nodes.append(node)

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", context: str | None = None) -> None:
        key = (src, tgt, relation)
        if key in seen_edges or src == tgt:
            return
        seen_edges.add(key)
        edge = {"source": src, "target": tgt, "relation": relation,
                "confidence": confidence, "source_file": str_path,
                "source_location": f"L{line}", "weight": 1.0}
        if context:
            edge["context"] = context
        if tgt in foreign:
            # transient hint the pipeline reads to rewrite an id minted from
            # another file's absolute path into that file's canonical id — also
            # when the file is not part of this run (an incremental update),
            # where the edge would otherwise keep the checkout path and dangle
            edge["target_file"] = str(foreign[tgt])
        edges.append(edge)

    foreign: dict[str, Path] = {}           # id minted for another file -> that file

    def other_file_nid(target: Path) -> str:
        if target.suffix.lower() in _RESOURCE_SUFFIXES:
            nid = _resource_nid(target)
        else:
            nid = _make_id(str(target))
        foreign[nid] = target
        return nid

    def other_symbol_nid(script: Path, name: str) -> str:
        nid = _make_id(_file_stem(script), name)
        foreign[nid] = script
        return nid

    def line_of(node) -> int:
        return node.start_point[0] + 1

    def string_value(node) -> str | None:
        if node.type != "string":
            return None
        return _read_text(node, source).strip("\"'")

    file_nid = _make_id(str(path))
    class_label = path.name
    if project is not None:
        # An autoload with no class_name is known to every other script by its
        # autoload name, so that is the label the graph carries for it.
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        for auto_name, auto_path in project.autoloads.items():
            if auto_path == resolved or auto_path == path:
                class_label = auto_name
                break
    add_node(file_nid, class_label, 1, "class")

    # --- pass 1: declarations --------------------------------------------------
    func_nids: dict[str, str] = {}          # top-level func name -> nid
    signal_nids: dict[str, str] = {}        # signal name -> nid
    preload_alias: dict[str, Path] = {}     # const NAME := preload("...gd") -> script
    inner_funcs: dict[str, dict[str, str]] = {}     # inner class nid -> its func name -> nid
    inner_names: dict[str, str] = {}                # inner class nid -> its name
    function_bodies: list[tuple[str, Any, Any, str]] = []   # (nid, body node, def node, owner nid)
    top_funcs: frozenset[str] = frozenset()

    def declare_member(node, owner_nid: str, owner_stem: str, inner: bool) -> None:
        t = node.type
        if t in ("function_definition", "constructor_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _read_text(name_node, source)
            elif t == "constructor_definition":
                name = "_init"      # `func _init(...)` parses with no name field
            else:
                return
            nid = _make_id(owner_stem, name)
            add_node(nid, f".{name}()" if inner else f"{name}()", line_of(node), "function")
            add_edge(owner_nid, nid, "method" if inner else "contains", line_of(node))
            if inner:
                inner_funcs.setdefault(owner_nid, {})[name] = nid
            else:
                func_nids[name] = nid
            body = node.child_by_field_name("body")
            if body:
                function_bodies.append((nid, body, node, owner_nid))
            return
        if t == "signal_statement":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _read_text(name_node, source)
            nid = _make_id(owner_stem, name)
            add_node(nid, f"signal {name}", line_of(node), "signal")
            add_edge(owner_nid, nid, "contains", line_of(node))
            if not inner:
                signal_nids[name] = nid
            return
        if t == "const_statement":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _read_text(name_node, source)
            # kind-qualified: make_id case-folds, so a `const Helper` and a
            # `func helper()` would otherwise claim one id
            nid = _make_id(owner_stem, "const", name)
            add_node(nid, name, line_of(node), "constant")
            add_edge(owner_nid, nid, "contains", line_of(node))
            for child in node.children:
                if child.type == "call":
                    target = _load_target(child)
                    if target is not None and target.suffix == ".gd":
                        preload_alias[name] = target
            return
        if t == "enum_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _read_text(name_node, source)
            nid = _make_id(owner_stem, "enum", name)
            add_node(nid, f"enum {name}", line_of(node), "enum")
            add_edge(owner_nid, nid, "contains", line_of(node))
            return
        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _read_text(name_node, source)
            nid = _make_id(owner_stem, name)
            add_node(nid, name, line_of(node), "class")
            add_edge(owner_nid, nid, "contains", line_of(node))
            # a class nested in an inner class gets a name no index holds: untyped
            inner_names[nid] = f"{inner_names[owner_nid]}/{name}" if inner else name
            for child in node.children:
                if child.type == "extends_statement":
                    emit_extends(child, nid)
                elif child.type == "class_body":
                    for member in child.children:
                        declare_member(member, nid, f"{owner_stem}/{name}", True)
            return

    def _load_target(call_node) -> Path | None:
        """The script, scene or resource a ``preload("…")`` / ``load("…")`` call
        names, or None."""
        fn = None
        args = None
        for c in call_node.children:
            if c.type == "identifier" and fn is None:
                fn = _read_text(c, source)
            elif c.type == "arguments":
                args = c
        if fn not in ("preload", "load") or args is None:
            return None
        for a in args.children:
            value = string_value(a)
            if value is not None:
                if value.endswith(".gd"):
                    return _script_of(value, path, project)
                if value.endswith(_RESOURCE_SUFFIXES) and project is not None:
                    return project.resolve(value)
                return None
        return None

    def emit_extends(node, owner_nid: str) -> None:
        spec = None
        for c in node.children:
            if c.type == "type":
                spec = _read_text(c, source)
            elif c.type == "string":
                spec = string_value(c)
        if not spec:
            return
        target = _script_of(spec, path, project)
        if target is not None:
            add_edge(owner_nid, other_file_nid(target), "inherits", line_of(node))

    for child in root.children:
        if child.type == "extends_statement":
            emit_extends(child, file_nid)
        elif child.type == "class_name_statement":
            name_node = child.child_by_field_name("name")
            if name_node:
                class_label = _read_text(name_node, source)
                for n in nodes:
                    if n["id"] == file_nid:
                        n["label"] = class_label
            for sub in child.children:
                if sub.type == "extends_statement":     # `class_name Foo extends Bar`
                    emit_extends(sub, file_nid)
        else:
            declare_member(child, file_nid, stem, False)
    top_funcs = frozenset(func_nids)

    # imports: every preload/load of a .gd / .tscn / .tres anywhere in the file
    # (const aliases, locals, inline arguments) -> the target's file node.
    def walk_loads(node) -> None:
        if node.type == "call":
            target = _load_target(node)
            if target is not None:
                add_edge(file_nid, other_file_nid(target), "imports", line_of(node))
        for c in node.children:
            walk_loads(c)
    walk_loads(root)

    # --- pass 2: calls and signal wiring inside function bodies -------------------
    ancestors = _ancestors(path, project)
    try:
        self_script = path.resolve()
    except OSError:
        self_script = path
    scope: dict[str, str] = {}      # funcs of the inner class whose method is being walked
    # the class whose method is being walked: (script, inner class name or "")
    self_cls: tuple[Path, str] = (self_script, "")

    def member_nid(cls: tuple[Path, str], name: str) -> str:
        """Id of function / signal ``name`` declared by class ``cls``."""
        script, inner = cls
        base = stem if script == self_script else _file_stem(script)
        nid = _make_id(f"{base}/{inner}" if inner else base, name)
        if script != self_script:
            foreign[nid] = script
        return nid

    def class_nid(cls: tuple[Path, str]) -> str:
        script, inner = cls
        if not inner:
            return file_nid if script == self_script else other_file_nid(script)
        nid = _make_id(stem if script == self_script else _file_stem(script), inner)
        if script != self_script:
            foreign[nid] = script
        return nid

    def resolve_bare(name: str) -> str | None:
        if name in scope:
            return scope[name]
        if self_cls[1]:
            # inherited by the inner class from the class it extends
            holder, _ = _declared(self_cls, project, "funcs", name)
            if holder is not None:
                return member_nid(holder, name)
        if name in func_nids:
            return func_nids[name]
        for anc in ancestors:
            funcs, _signals, _ext = _file_index(anc)
            if name in funcs:
                return other_symbol_nid(anc, name)
        return None

    def resolve_receiver(receiver: str) -> Path | None:
        if receiver in preload_alias:
            return preload_alias[receiver]
        if project is not None:
            if receiver in project.autoloads:
                return project.autoloads[receiver]
            if receiver in project.class_names:
                return project.class_names[receiver]
        return None

    def signal_nid_for(cls: tuple[Path, str] | None, name: str) -> str | None:
        """Signal ``name`` of class ``cls``, or of the class being walked for None."""
        if cls is None and not self_cls[1]:
            if name in signal_nids:
                return signal_nids[name]
            for anc in ancestors:
                _funcs, signals, _ext = _file_index(anc)
                if name in signals:
                    return other_symbol_nid(anc, name)
            return None
        holder, _ = _declared(cls or self_cls, project, "signals", name)
        return member_nid(holder, name) if holder is not None else None

    # --- static types: what a receiver expression is known to be -----------------
    # A type is ("object", class), ("array", element class) or ("dict", (key
    # class, value class)), a class being (script, inner class name or "").
    # Anything the project has no script for (an engine class, an untyped value)
    # is None.
    env: dict[str, tuple | None] = {}       # locals of the function being walked

    def named_type(name: str | None, ctx: tuple[Path, str]) -> tuple | None:
        """The type a type name written in class ``ctx`` stands for."""
        if not name:
            return None
        container = _CONTAINER_RE.match(name)
        if container is None:
            cls = _named_class(name, ctx, project)
            return ("object", cls) if cls is not None else None
        args = [_named_class(a.strip(), ctx, project) for a in container.group(2).split(",")]
        if container.group(1) == "Array":
            return ("array", args[0]) if args[0] is not None else None
        if len(args) == 2 and (args[0] is not None or args[1] is not None):
            return ("dict", (args[0], args[1]))
        return None

    def node_at(anchor: Path, ref) -> tuple | None:
        """The node the script ``anchor`` reaches by ``ref`` — a path (``A/B``) or
        a tuple of steps — as ("node", (anchor, steps)); it is given a class only
        when a member of it is asked for (as_object). An absolute ``/root/Name``
        is the autoload of that name."""
        if not ref or project is None:
            return None
        if isinstance(ref, str):
            if ref.startswith("/"):
                script = project.autoloads.get(ref[len("/root/"):]) if ref.startswith("/root/") else None
                if script is not None and script.suffix == ".gd":
                    return ("object", (script, ""))
                return None
            ref = tuple(ref.split("/"))
        return ("node", (anchor, ref))

    def one_of(classes, certain: bool) -> tuple | None:
        """The type of a value that is one of ``classes``."""
        classes = tuple(dict.fromkeys(classes))
        if not classes or len(classes) > _MAX_CANDIDATES:
            return None
        if len(classes) == 1 and certain:
            return ("object", classes[0])
        return ("objects", classes)

    def as_object(of: tuple | None) -> tuple | None:
        """A node as the class of the script the project's scenes put on it — or,
        where the scenes the script runs in differ, as each of those classes."""
        if of is None or of[0] != "node":
            return of
        # imported here: godot_resource builds on this module
        from graphify.extractors.godot_resource import _node_scripts_for
        scripts, certain = _node_scripts_for(of[1][0], of[1][1], project)
        return one_of([(s, "") for s in sorted(scripts)], certain)

    def classes_of(of: tuple | None) -> tuple[tuple, bool]:
        """The classes a value may be, and whether it is certainly the one."""
        of = as_object(of)
        if of is not None and of[0] == "object":
            return (of[1],), True
        if of is not None and of[0] == "objects":
            return of[1], False
        return (), False

    def declared_in(of: tuple | None, table: str, name: str) -> tuple[list, str]:
        """The classes declaring function / signal ``name`` among those ``of``
        may be, and the confidence an edge to them carries: EXTRACTED only when
        there is no doubt which one it is."""
        classes, certain = classes_of(of)
        holders: list = []
        for cls in classes:
            holder, _ = _declared(cls, project, table, name)
            if holder is None:
                certain = False
            elif holder not in holders:
                holders.append(holder)
        return holders, "EXTRACTED" if certain and len(holders) == 1 else "INFERRED"

    def ref_steps(ref: str, ctx: tuple[Path, str]):
        """A node reference as the index records it, for node_at: a path, or
        ``@name`` for ``get_node(name)`` — a string constant of class ``ctx``,
        else a ``NodePath`` property the scene sets on the node."""
        if not ref.startswith("@"):
            return ref
        name = ref[1:]
        literal = _declared(ctx, project, "strings", name)[1]
        if literal is not None:
            return literal
        if _declared(ctx, project, "members", name)[0] is None:
            return None
        return (("prop", name, _declared(ctx, project, "defaults", name)[1], False),)

    def declared_type(cls: tuple[Path, str], name: str, table: str) -> tuple | None:
        """Type of member var (table "members") / return type of function (table
        "returns") ``name`` of ``cls``, declared by it or by a class it extends."""
        holder, entry = _declared(cls, project, table, name)
        if holder is None:
            return None
        if table == "returns":
            return named_type(entry, holder)
        # the annotation; else what the member is initialised with, then what a
        # function of the class assigns to it (`_hud = get_node(hud_path)`); an
        # engine annotation (`var _hud: Node`) says nothing either way
        assigned = _type_index(*holder).assigned.get(name, (None, None, True))
        for type_name, node, certain in ((*entry, True), assigned):
            typed = named_type(type_name, holder)
            if typed is None and node and not holder[1]:
                typed = node_at(holder[0], ref_steps(node, holder))
            if typed is not None:
                # assigned here, but something untyped is assigned elsewhere
                # (`provider = injected`): a candidate, not a fact
                return typed if certain else one_of(classes_of(typed)[0], False)
        if holder[1]:
            return None
        # ... nor does a bare `@export var target: Node`: a scene fills it in
        return node_at(holder[0], (("prop", name, None, True),))

    def member_of(of: tuple | None, name: str, table: str) -> tuple | None:
        """Type of member / return type ``name`` of whatever ``of`` is."""
        classes, certain = classes_of(of)
        if certain:
            return declared_type(classes[0], name, table)
        found: list = []
        for cls in classes:
            found += classes_of(declared_type(cls, name, table))[0]
        return one_of(found, False)

    def path_arg(args):
        """What a ``get_node(...)`` is given: a literal path, or a name — a string
        constant or a ``NodePath`` member of this class."""
        first = next((a for a in args.children if a.is_named), None) if args is not None else None
        if first is None:
            return None
        if first.type in ("string", "node_path"):
            return _read_text(first, source).lstrip("^&").strip("\"'")
        if first.type == "identifier" and _read_text(first, source) not in env:
            return ref_steps("@" + _read_text(first, source), self_cls)
        return None

    def node_step(name: str, args):
        """The way through the scene tree a navigation call takes: a path or steps."""
        if name in ("get_node", "get_node_or_null"):
            return path_arg(args)
        if name == "get_parent":
            return ("..",)
        if name == "find_child":
            wanted = string_arg(args, 0)
            return (("find", wanted),) if wanted and not set(wanted) & set("*?/") else None
        return None

    def node_ref(node):
        """The path a ``$A/B`` / ``%Name`` literal names, or the way a bare
        ``get_node("A/B")`` / ``get_parent()`` / ``find_child("A")`` takes."""
        if node.type == "get_node":
            return _read_text(node, source).lstrip("$").strip('"')
        if node.type != "call":
            return None
        fn = next((c for c in node.children if c.type == "identifier"), None)
        if fn is None:
            return None
        return node_step(_read_text(fn, source),
                         next((c for c in node.children if c.type == "arguments"), None))

    def from_node(of: tuple, step) -> tuple | None:
        """The node ``step`` away from a node, or from the node an object's script runs on."""
        if isinstance(step, str):
            if step.startswith("/"):
                return node_at(self_script, step)
            step = tuple(step.split("/"))
        if of[0] == "node":
            return node_at(of[1][0], of[1][1] + step)
        if of[0] == "object" and not of[1][1]:
            return node_at(of[1][0], step)
        return None

    def expr_type(node) -> tuple | None:
        t = node.type
        if t == "identifier":
            name = _read_text(node, source)
            if name in env:
                return env[name]
            if name == "self":
                return ("object", self_cls)
            if name == "super":
                chain = _class_chain(self_cls, project)
                return ("object", chain[1]) if len(chain) > 1 else None
            holder, _ = _declared(self_cls, project, "members", name)
            if holder is not None:
                return declared_type(self_cls, name, "members")
            if name == "owner" and not self_cls[1]:
                return node_at(self_script, (("owner",),))  # Node.owner: the scene's root
            script = resolve_receiver(name)
            return ("object", (script, "")) if script is not None else named_type(name, self_cls)
        if t == "attribute":
            return chain_type([c for c in node.children if c.is_named])
        if t == "get_node":
            return node_at(self_script, node_ref(node)) if not self_cls[1] else None
        if t == "call":
            fn = next((c for c in node.children if c.type == "identifier"), None)
            if fn is None:
                return None
            name = _read_text(fn, source)
            if _declared(self_cls, project, "funcs", name)[0] is not None or self_cls[1]:
                return declared_type(self_cls, name, "returns")
            if name == "get_tree":
                return ("tree", None)
            return node_at(self_script, node_ref(node))     # get_node("A") / get_parent() on self
        if t in ("parenthesized_expression", "await_expression"):
            inner = next((c for c in node.children if c.is_named), None)
            return expr_type(inner) if inner is not None else None
        if t == "binary_operator":
            if any(c.type == "as" for c in node.children):
                cast = named_type(_read_text(node.children[-1], source), self_cls)
                # an engine cast over a node (`$Hud as Control`) leaves the node's own type
                return cast or (expr_type(node.children[0]) if node_ref(node.children[0]) else None)
            return None
        if t == "subscript":
            base = next((c for c in node.children if c.is_named), None)
            of = expr_type(base) if base is not None else None
            if of is not None and of[0] == "array":
                return ("object", of[1])
            if of is not None and of[0] == "dict" and of[1][1] is not None:
                return ("object", of[1][1])
            return None
        return None

    def chain_type(parts: list) -> tuple | None:
        """Type of ``head.member.method()...``: each step is looked up in the
        class the step before it is typed as; a step through the scene tree
        (``.get_node("A")``, ``.get_parent()``, ``.owner``) moves to that node."""
        current = expr_type(parts[0]) if parts else None
        for part in parts[1:]:
            if current is None:
                return None
            args = None
            if part.type == "identifier":
                name, table = _read_text(part, source), "members"
                step = (("owner",),) if name == "owner" else None
            elif part.type == "attribute_call":
                method = next((c for c in part.children if c.type == "identifier"), None)
                if method is None:
                    return None
                args = next((c for c in part.children if c.type == "arguments"), None)
                name, table = _read_text(method, source), "funcs"
                step = node_step(name, args)
            else:
                return None
            if current[0] == "tree":
                # get_tree().root is where the autoloads live; .current_scene is
                # whichever scene the tree was changed to
                if name == "root" and table == "members":
                    current = ("root", None)
                elif name == "current_scene" and table == "members":
                    current = node_at(self_script, (("current",),))
                else:
                    return None
                continue
            if current[0] == "root":
                if name in ("get_node", "get_node_or_null") and isinstance(step, str):
                    current = node_at(self_script, "/root/" + step)
                    continue
                return None
            if step is not None and current[0] in ("node", "object"):
                # ... unless the class declares that very name itself
                if current[0] == "node" or _declared(current[1], project, table, name)[0] is None:
                    current = from_node(current, step)
                    continue
            if table == "members":
                current = member_of(current, name, "members")
            elif name != "new":                             # Class.new() is a Class
                current = member_of(current, name, "returns")
        return current

    def reference_method(node, caller_nid: str) -> None:
        """``obj.method`` handed over as a Callable, also ``obj.method.bind(...)``:
        a reference to that method of the class ``obj`` is typed as."""
        parts = [c for c in node.children if c.is_named]
        while parts and parts[-1].type == "attribute_call":
            method = next((c for c in parts[-1].children if c.type == "identifier"), None)
            if method is None or _read_text(method, source) not in ("bind", "unbind"):
                return
            parts.pop()
        if len(parts) < 2 or parts[-1].type != "identifier":
            return
        name = _read_text(parts[-1], source)
        holders, confidence = declared_in(chain_type(parts[:-1]), "funcs", name)
        for holder in holders:
            add_edge(caller_nid, member_nid(holder, name), "references", line_of(node),
                     confidence, context="callable")

    def declare(node) -> None:
        """Record the names one construct declares (a parameter list, a ``var``,
        a loop variable) with their type, or None when they have none — an
        untyped local still hides a typed member of the same name."""
        t = node.type
        if t == "parameters":
            for p in node.children:
                if p.type == "identifier":
                    env[_read_text(p, source)] = None
                elif p.type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
                    name = next((c for c in p.children if c.type == "identifier"), None)
                    kind = next((c for c in p.children if c.type == "type"), None)
                    if name is None:
                        continue
                    if kind is not None:
                        env[_read_text(name, source)] = named_type(_read_text(kind, source), self_cls)
                    elif p.type == "typed_default_parameter":       # `n := value`
                        env[_read_text(name, source)] = expr_type(p.children[-1])
                    else:
                        env[_read_text(name, source)] = None
        elif t == "variable_statement":
            name = node.child_by_field_name("name")
            kind = node.child_by_field_name("type")
            value = node.child_by_field_name("value")
            if name is not None:
                annotated = kind is not None and kind.type == "type"
                declared = named_type(_read_text(kind, source), self_cls) if annotated else None
                if declared is None and value is not None:
                    # an engine annotation over a node (`var hud: Control = $Hud`)
                    # says less than the scene does; with no annotation the
                    # initialiser is all there is
                    found = expr_type(value)
                    if not annotated or node_ref(value) or (found is not None and found[0] == "node"):
                        declared = found
                env[_read_text(name, source)] = declared
        elif t == "for_statement":
            named = [c for c in node.children if c.is_named and c.type != "body"]
            if named and named[0].type == "identifier":
                if len(named) > 1 and named[1].type == "type":
                    kind = named_type(_read_text(named[1], source), self_cls)
                else:
                    of = expr_type(named[-1]) if len(named) > 1 else None
                    kind = None
                    if of is not None and of[0] == "array":
                        kind = ("object", of[1])
                    elif of is not None and of[0] == "dict" and of[1][0] is not None:
                        kind = ("object", of[1][0])      # iterating a dictionary yields keys
                env[_read_text(named[0], source)] = kind

    def callable_name(node) -> str | None:
        """The function of this class an expression hands over as a Callable:
        ``cb``, ``self.cb``, or either with ``.bind(...)`` / ``.unbind(n)``."""
        if node.type == "identifier":
            names = [_read_text(node, source)]
        elif node.type == "attribute":
            names = [_read_text(c, source) for c in node.children if c.type == "identifier"]
            for c in node.children:
                if c.type == "attribute_call":
                    method = next((m for m in c.children if m.type == "identifier"), None)
                    if method is None or _read_text(method, source) not in ("bind", "unbind"):
                        return None
                elif c.is_named and c.type != "identifier":
                    return None
        else:
            return None
        via_self = names[:1] == ["self"]
        if via_self:
            names = names[1:]
        if len(names) != 1 or names[0] in ("self", "super"):
            return None
        if not via_self and names[0] in env:
            return None         # a local of that name, not the function
        return names[0]

    def reference_callable(node, caller_nid: str, context: str = "callable") -> None:
        """A function of this class used as a value is a reference to it."""
        name = callable_name(node)
        target = resolve_bare(name) if name else None
        if target is not None:
            add_edge(caller_nid, target, "references", line_of(node), context=context)

    def string_arg(args, at: int) -> str | None:
        """Argument ``at`` when it is a string literal: the method a
        ``call("name", ...)`` / ``Callable(obj, "name")`` names."""
        operands = [a for a in args.children if a.is_named] if args is not None else []
        return string_value(operands[at]) if len(operands) > at else None

    def handle_attribute(node, caller_nid: str) -> None:
        # attribute := <head> ('.' identifier | '.' attribute_call)*
        # Every call of the chain is resolved against the type of what precedes it.
        parts = [c for c in node.children if c.is_named]
        line = line_of(node)
        for at, call in enumerate(parts):
            if call.type != "attribute_call":
                continue
            method_node = next((c for c in call.children if c.type == "identifier"), None)
            if method_node is None:
                continue
            method = _read_text(method_node, source)
            args = next((c for c in call.children if c.type == "arguments"), None)
            receiver = parts[:at]
            names = [_read_text(c, source) for c in receiver if c.type == "identifier"]
            shown = f"{'.'.join(names)}.{method}" if names else method
            on_self = len(receiver) == 1 and names[:1] in (["self"], ["super"])

            # signal wiring: <signal>.connect(cb) / <signal>.emit(...)
            if method in ("connect", "emit") and receiver and receiver[-1].type == "identifier":
                # the callback is a function of this script whoever owns the signal
                # (`button.pressed`, `_timer.timeout`), so it is wired before the
                # signal's own resolution can bail out
                first = next((a for a in args.children if a.is_named), None) if args is not None else None
                if method == "connect" and first is not None:
                    reference_callable(first, caller_nid, "connect")
                owner = receiver[:-1]
                if owner and not (len(owner) == 1 and names[0] == "self"):
                    of = chain_type(owner)
                    if not classes_of(of)[0]:
                        # a signal of some other object (a child node, an untyped
                        # local) — nothing in this project index names it
                        unresolved_calls.append({"caller_nid": caller_nid, "callee": shown, "line": line})
                        continue
                    holders, confidence = declared_in(of, "signals", names[-1])
                    for holder in holders:
                        add_edge(caller_nid, member_nid(holder, names[-1]), "uses", line,
                                 confidence, context=method)
                    continue
                sig = signal_nid_for(None, names[-1])
                if sig is not None:
                    add_edge(caller_nid, sig, "uses", line, context=method)
                continue

            # obj.call("name") / obj.call_deferred("name") name the method in a string
            if method in _DYNAMIC_CALLS and string_arg(args, 0):
                method = string_arg(args, 0)

            if on_self:
                target = resolve_bare(method)
                if target is not None:
                    add_edge(caller_nid, target, "calls", line, context="call")
                else:
                    unresolved_calls.append({"caller_nid": caller_nid, "callee": method, "line": line})
                continue

            of = chain_type(receiver)
            classes, certain = classes_of(of)
            if classes and method == "new":
                for cls in classes:
                    add_edge(caller_nid, class_nid(cls), "references", line,
                             "EXTRACTED" if certain else "INFERRED", context="instantiates")
                continue
            holders, confidence = declared_in(of, "funcs", method)
            for holder in holders:
                add_edge(caller_nid, member_nid(holder, method), "calls", line, confidence, context="call")
            if not holders:
                unresolved_calls.append({"caller_nid": caller_nid, "callee": shown, "line": line})

    def handle_call(node, caller_nid: str) -> None:
        fn = None
        args = None
        for c in node.children:
            if c.type == "identifier" and fn is None:
                fn = _read_text(c, source)
            elif c.type == "arguments":
                args = c
        if fn is None:
            return
        line = line_of(node)
        if fn in ("preload", "load"):
            return
        if fn == "emit_signal" and args is not None:
            for a in args.children:
                name = string_value(a)
                if name is not None:
                    sig = signal_nid_for(None, name)
                    if sig is not None:
                        add_edge(caller_nid, sig, "uses", line, context="emit")
                    return
            return
        if fn in ("connect", "is_connected", "disconnect") and args is not None:
            # Godot 3 style connect("sig", target, "method") is not Godot 4; skip.
            return
        if fn == "Callable" and args is not None:
            # Callable(object, "method"): a reference to that object's method
            method = string_arg(args, 1)
            of = expr_type(next(a for a in args.children if a.is_named)) if method else None
            holders, confidence = declared_in(of, "funcs", method)
            for holder in holders:
                add_edge(caller_nid, member_nid(holder, method), "references", line,
                         confidence, context="callable")
            return
        if fn in _DYNAMIC_CALLS and string_arg(args, 0):
            fn = string_arg(args, 0)        # call("name") / call_deferred("name") on self
        target = resolve_bare(fn)
        if target is not None:
            add_edge(caller_nid, target, "calls", line, context="call")
        elif not _ENGINE_BUILTIN_RE.match(fn):
            unresolved_calls.append({"caller_nid": caller_nid, "callee": fn, "line": line})

    def walk_calls(node, caller_nid: str) -> None:
        t = node.type
        if t == "function_definition":
            return
        if t == "identifier":
            reference_callable(node, caller_nid)
            return
        if t == "attribute":
            handle_attribute(node, caller_nid)
            reference_callable(node, caller_nid)    # `self._on_done`, `_on_done.bind(1)`
            reference_method(node, caller_nid)      # `hud.show_x`, `hud.show_x.bind(1)`
            for c in node.children:
                if c.type == "attribute_call":
                    for a in c.children:
                        if a.type == "arguments":
                            walk_calls(a, caller_nid)
                elif c.type != "identifier":
                    # the receiver: a call, a nested attribute, or a subscript /
                    # parenthesised expression that holds one (`rows[pick()].run()`).
                    # An identifier of the chain is the receiver's own name or a
                    # member of it, never a function handed over as a value.
                    walk_calls(c, caller_nid)
            return
        if t == "call":
            handle_call(node, caller_nid)
            for c in node.children:
                if c.type == "arguments":
                    walk_calls(c, caller_nid)
            return
        if t in ("body", "for_statement", "lambda"):
            # a block: what it declares goes out of scope with it
            outer = dict(env)
            if t == "for_statement":
                declare(node)
            for c in node.children:
                if c.type == "parameters":      # of a lambda
                    declare(c)
                walk_calls(c, caller_nid)
            env.clear()
            env.update(outer)
            return
        if t == "pair" and any(c.type == "=" for c in node.children):
            # `{key = value}`: the key is a name, not an expression
            for c in [c for c in node.children if c.is_named][1:]:
                walk_calls(c, caller_nid)
            return
        if t in ("type", "parameters"):
            return          # names here are types and declarations, never values
        for c in node.children:
            walk_calls(c, caller_nid)
        if t == "variable_statement":
            declare(node)       # after its initialiser: `var hud = hud.child` reads the old hud

    for nid, body, definition, owner in function_bodies:
        scope = inner_funcs.get(owner, {})
        self_cls = (self_script, inner_names.get(owner, ""))
        env.clear()
        parameters = definition.child_by_field_name("parameters")
        if parameters is not None:
            declare(parameters)
        walk_calls(body, nid)

    # --- pass 3: documentation citations in comments -----------------------------
    spans = [(d.start_byte, d.end_byte, nid) for nid, _b, d, _owner in function_bodies]

    def enclosing(byte: int) -> str:
        for start, end, nid in spans:
            if start <= byte < end:
                return nid
        return file_nid

    def section_node(page: str, section: str, line: int) -> str:
        # The caller hands a basename, but the resolver enforces containment itself:
        # a page is looked up under the project root (or its docs/), and only a real
        # file that RESOLVES inside the root is allowed to rewrite the recorded path.
        page_rel = page.lstrip("./")
        if project is not None and ".." not in Path(page_rel).parts:
            try:
                root = project.root.resolve()
            except OSError:
                root = project.root
            for candidate in (root / page_rel, root / "docs" / page_rel):
                try:
                    resolved = candidate.resolve()
                    if resolved.is_file() and resolved.is_relative_to(root):
                        page_rel = resolved.relative_to(root).as_posix()
                        break
                except OSError:
                    continue
        nid = _make_id("doc", page_rel, "s" + section)
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": f"{Path(page_rel).name} §{section}",
                          "file_type": "doc", "type": "section",
                          "source_file": page_rel, "source_location": f"§{section}"})
        return nid

    last_page: str | None = None

    def walk_comments(node) -> None:
        nonlocal last_page
        if node.type == "comment":
            text = _read_text(node, source)
            line = line_of(node)
            src = enclosing(node.start_byte)
            explicit = list(_CITATION_RE.finditer(text))
            spans = [(m.start(), m.end()) for m in explicit]
            bare = [m for m in _BARE_CITATION_RE.finditer(text)
                    if not any(a <= m.start() < b for a, b in spans)]
            # walk every citation in the order it appears, so a bare section takes the
            # page named before it — never one named later on the same line
            for m in sorted(explicit + bare, key=lambda m: m.start()):
                if m.re is _CITATION_RE:
                    last_page = m.group(1).split("/")[-1]
                    add_edge(src, section_node(last_page, m.group(2), line), "references", line,
                             context="citation")
                elif last_page is not None:
                    add_edge(src, section_node(last_page, m.group(1), line), "references", line,
                             confidence="INFERRED", context="citation")
            return
        for c in node.children:
            walk_comments(c)
    walk_comments(root)

    return {"nodes": nodes, "edges": edges, "raw_calls": [],
            "unresolved_calls": unresolved_calls}
