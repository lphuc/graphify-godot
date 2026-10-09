"""Godot shaders (.gdshader / .gdshaderinc), and the edges scripts and scenes aim at them."""
import importlib.util
from pathlib import Path

import pytest

from graphify.extract import _file_stem, _make_id, extract
from graphify.extractors.gdscript import _resource_nid
from graphify.extractors.gdshader import extract_gdshader

_INCLUDE = (
    "// vec3 commented_out(float i) {\n"
    "const float STRENGTH = 0.5;\n\n"
    "vec3 tint(vec3 colour, float i) {\n"
    "\treturn mix(colour, vec3(0.8, 0.9, 1.2), STRENGTH * i);\n"
    "}\n"
)

_SHADER = (
    "shader_type canvas_item;\n"
    '#include "res://shaders/common.gdshaderinc"\n\n'
    "uniform float speed : hint_range(0.0, 4.0) = 1.0;\n\n"
    "/* float in_a_block_comment(float x) {\n"
    "   return x; } */\n"
    "float wave(vec2 at,\n"
    "\t\tfloat time) {\n"
    "\treturn sin(at.x * speed + time);\n"
    "}\n\n"
    "void fragment() {\n"
    "\tfloat height = wave(UV, TIME);\n"
    "\tif (height > 0.0) {\n"
    "\t\tCOLOR.rgb = tint(COLOR.rgb, height);\n"
    "\t} else if (height < -0.5) {\n"
    "\t\tdiscard;\n"
    "\t}\n"
    "}\n"
)


def _project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.godot").write_text(
        'config_version=5\n\n[application]\n\nconfig/name="Probe"\n', encoding="utf-8")
    (root / "shaders").mkdir()
    (root / "shaders" / "common.gdshaderinc").write_text(_INCLUDE, encoding="utf-8")
    (root / "shaders" / "water.gdshader").write_text(_SHADER, encoding="utf-8")
    return root


def _edges(result: dict, relation: str) -> list[dict]:
    return [e for e in result["edges"] if e["relation"] == relation]


def _shader_func(path: Path, name: str) -> str:
    return _make_id(_file_stem(path), path.suffix.lstrip("."), name)


def test_shader_declares_its_functions_and_nothing_else(tmp_path):
    root = _project(tmp_path / "proj")
    shader = root / "shaders" / "water.gdshader"
    r = extract_gdshader(shader)

    assert sorted(n["label"] for n in r["nodes"]) == ["fragment()", "water.gdshader", "wave()"]
    file_nid = _resource_nid(shader)
    assert {e["target"] for e in _edges(r, "contains")} == {
        _shader_func(shader, "wave"), _shader_func(shader, "fragment")}
    assert {e["source"] for e in _edges(r, "contains")} == {file_nid}
    # the wrapped parameter list does not hide where the function starts
    wave = next(n for n in r["nodes"] if n["label"] == "wave()")
    assert wave["source_location"] == "L8"


def test_shader_include_is_an_import_and_types_the_calls_into_it(tmp_path):
    root = _project(tmp_path / "proj")
    shader = root / "shaders" / "water.gdshader"
    include = (root / "shaders" / "common.gdshaderinc").resolve()
    r = extract_gdshader(shader)

    imports = _edges(r, "imports")
    assert [(e["target"], e["source_location"]) for e in imports] == [(_resource_nid(include), "L2")]
    assert imports[0]["target_file"] == str(include)

    fragment = _shader_func(shader, "fragment")
    calls = {(e["source"], e["target"]) for e in _edges(r, "calls")}
    # one in this file, one in the file it includes; `sin`, `mix` and `vec3` are the language's
    assert calls == {(fragment, _shader_func(shader, "wave")),
                     (fragment, _shader_func(include, "tint"))}


def test_shader_include_may_be_relative_and_may_nest(tmp_path):
    root = _project(tmp_path / "proj")
    (root / "shaders" / "lib").mkdir()
    deep = root / "shaders" / "lib" / "noise.gdshaderinc"
    deep.write_text("float noise(vec2 p) {\n\treturn fract(p.x);\n}\n", encoding="utf-8")
    middle = root / "shaders" / "common.gdshaderinc"
    middle.write_text('#include "lib/noise.gdshaderinc"\n' + _INCLUDE, encoding="utf-8")
    shader = root / "shaders" / "fog.gdshader"
    shader.write_text(
        'shader_type canvas_item;\n#include "common.gdshaderinc"\n\n'
        "void fragment() {\n\tCOLOR.a = noise(UV);\n}\n", encoding="utf-8")

    r = extract_gdshader(shader)
    assert [e["target"] for e in _edges(r, "imports")] == [_resource_nid(middle.resolve())]
    # `noise` comes through the include of the include
    assert [e["target"] for e in _edges(r, "calls")] == [_shader_func(deep.resolve(), "noise")]


def test_shader_missing_include_and_include_cycle_are_harmless(tmp_path):
    root = _project(tmp_path / "proj")
    a = root / "shaders" / "a.gdshaderinc"
    b = root / "shaders" / "b.gdshaderinc"
    a.write_text('#include "b.gdshaderinc"\n#include "res://shaders/gone.gdshaderinc"\n'
                 "float a_fn() {\n\treturn b_fn();\n}\n", encoding="utf-8")
    b.write_text('#include "a.gdshaderinc"\nfloat b_fn() {\n\treturn 1.0;\n}\n', encoding="utf-8")

    r = extract_gdshader(a)
    assert [e["target"] for e in _edges(r, "imports")] == [_resource_nid(b.resolve())]
    assert [e["target"] for e in _edges(r, "calls")] == [_shader_func(b.resolve(), "b_fn")]


def test_shaders_are_registered_everywhere_a_code_file_is():
    from graphify.detect import CODE_EXTENSIONS
    from graphify.extract import _CACHE_BYPASS_SUFFIXES, _DISPATCH, _lang_family
    from graphify.extractors import LANGUAGE_EXTRACTORS
    from graphify.watch import _WATCHED_EXTENSIONS
    import graphify.extract as facade

    for suffix in (".gdshader", ".gdshaderinc"):
        assert _DISPATCH[suffix] is extract_gdshader
        assert suffix in CODE_EXTENSIONS and suffix in _WATCHED_EXTENSIONS
        # a call into an included file is resolved from THAT file's text
        assert suffix in _CACHE_BYPASS_SUFFIXES
        assert _lang_family(f"shaders/x{suffix}") == "gdshader"
    assert facade.extract_gdshader is extract_gdshader
    assert LANGUAGE_EXTRACTORS["gdshader"] is extract_gdshader


@pytest.mark.skipif(importlib.util.find_spec("tree_sitter_gdscript") is None,
                    reason="tree-sitter-gdscript not installed")
def test_scripts_scenes_and_materials_reach_the_shader_through_the_pipeline(tmp_path):
    """Through extract(), which rewrites every id root-relative: a `fog.gd` beside
    a `fog.gdshader` must stay two files, each with its own `fragment`."""
    root = _project(tmp_path / "proj")
    (root / "shaders" / "water.gd").write_text(
        'extends Node\n\nconst WATER := preload("res://shaders/water.gdshader")\n\n\n'
        "func fragment() -> void:\n\tpass\n", encoding="utf-8")
    (root / "water.tscn").write_text(
        '[gd_scene load_steps=3 format=3]\n\n'
        '[ext_resource type="Shader" path="res://shaders/water.gdshader" id="1_w"]\n'
        '[ext_resource type="Texture2D" path="res://icon.png" id="2_t"]\n\n'
        '[sub_resource type="ShaderMaterial" id="mat"]\nshader = ExtResource("1_w")\n\n'
        '[node name="Water" type="Sprite2D"]\nmaterial = SubResource("mat")\n', encoding="utf-8")
    (root / "water_material.tres").write_text(
        '[gd_resource type="ShaderMaterial" load_steps=2 format=3]\n\n'
        '[ext_resource type="Shader" path="res://shaders/water.gdshader" id="1_w"]\n\n'
        '[resource]\nshader = ExtResource("1_w")\n', encoding="utf-8")

    paths = sorted(p for p in root.rglob("*")
                   if p.suffix in (".gd", ".tscn", ".tres", ".godot", ".gdshader", ".gdshaderinc"))
    r = extract(paths, cache_root=tmp_path / "cache", root=root, parallel=False)

    by_id = {n["id"]: n for n in r["nodes"]}
    assert len(by_id) == len(r["nodes"]), "one node per id"
    for e in r["edges"]:
        assert e["target"] in by_id, f"dangling edge: {e}"
        assert "target_file" not in e

    def targets(relation: str, source_file: str) -> list[tuple[str, str]]:
        return sorted((by_id[e["target"]]["label"], by_id[e["target"]]["source_file"])
                      for e in _edges(r, relation) if e["source_file"] == source_file)

    shader = ("water.gdshader", "shaders/water.gdshader")
    assert targets("imports", "shaders/water.gd") == [shader]
    assert targets("uses_resource", "water.tscn") == [shader]
    assert targets("uses_resource", "water_material.tres") == [shader]
    assert targets("imports", "shaders/water.gdshader") == [
        ("common.gdshaderinc", "shaders/common.gdshaderinc")]
    assert targets("calls", "shaders/water.gdshader") == [
        ("tint()", "shaders/common.gdshaderinc"), ("wave()", "shaders/water.gdshader")]
    # the script's `fragment` and the shader's are two nodes
    assert sorted(n["source_file"] for n in r["nodes"] if n["label"] == "fragment()") == [
        "shaders/water.gd", "shaders/water.gdshader"]
