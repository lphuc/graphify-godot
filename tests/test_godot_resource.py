import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from graphify.extract import extract, extract_godot_resource, _make_id, _file_stem
from graphify.extractors import godot_resource as gr

_FIXTURES = Path(__file__).parent / "fixtures" / "godot" / "resource"


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


def _norm(result):
    """Order-independent snapshot of a result's nodes and edges."""
    n = sorted((x["id"], x["label"], x.get("source_location") or "")
               for x in result["nodes"])
    e = sorted((x["source"], x["target"], x["relation"],
                x.get("context") or "", x.get("source_location") or "")
               for x in result["edges"])
    return n, e


class _FixtureProject(unittest.TestCase):
    """Copy the fixture Godot project into an isolated temp dir per test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        shutil.copytree(_FIXTURES, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p


class TestGodotResource(_FixtureProject):
    """The .tscn/.tres/project.godot extractor (grammar path when available)."""

    def test_scene_script_subscene_and_connection(self):
        p = self.root / "scenes" / "Main.tscn"
        r = extract_godot_resource(p)

        enemy_gd = _make_id(str((self.root / "scripts" / "enemy.gd").resolve()))
        bullet = gr._resource_nid((self.root / "scenes" / "Bullet.tscn").resolve())

        attaches = {e["target"] for e in _edges(r, "attaches_script")}
        self.assertIn(enemy_gd, attaches)

        instances = {e["target"] for e in _edges(r, "instances")}
        self.assertIn(bullet, instances)

        # the connection method resolves to the ROOT script's function node id,
        # i.e. the same id the gdscript extractor emits for take_damage()
        stem = _file_stem((self.root / "scripts" / "enemy.gd").resolve())
        take_damage_nid = _make_id(stem, "take_damage")
        conn_targets = {e["target"] for e in _edges(r, "connects")}
        self.assertIn(take_damage_nid, conn_targets)

    def test_script_edge_is_emitted_once_and_names_its_node(self):
        r = extract_godot_resource(self.root / "scenes" / "Main.tscn")
        attaches = _edges(r, "attaches_script")
        self.assertEqual(len(attaches), 1)
        self.assertEqual(attaches[0]["context"], ".")

    def test_referenced_files_get_no_stub_node_and_assets_no_edge(self):
        scene = self._write(
            "scenes/Hud.tscn",
            '[gd_scene format=3]\n\n'
            '[ext_resource type="Texture2D" path="res://icon.png" id="1_t"]\n'
            '[ext_resource type="Script" path="res://scripts/enemy.gd" id="2_s"]\n'
            '[ext_resource type="Resource" path="res://data/stats.tres" id="3_r"]\n\n'
            '[node name="Hud" type="Control"]\nscript = ExtResource("2_s")\n')
        r = extract_godot_resource(scene)
        # the scene's own file node only: a stub for enemy.gd would be owned by
        # this scene, and the pipeline would keep it apart from the real script
        self.assertEqual([n["label"] for n in r["nodes"]], ["Hud.tscn"])
        self.assertEqual(sorted(e["relation"] for e in r["edges"]),
                         ["attaches_script", "uses_resource"])
        stats = gr._resource_nid((self.root / "data" / "stats.tres").resolve())
        self.assertEqual([e["target"] for e in _edges(r, "uses_resource")], [stats])

    def test_connection_resolves_a_method_declared_by_an_ancestor(self):
        self._write("scripts/boss.gd", "extends Enemy\n")
        scene = self._write(
            "scenes/Boss.tscn",
            '[gd_scene format=3]\n\n'
            '[ext_resource type="Script" path="res://scripts/boss.gd" id="1_b"]\n\n'
            '[node name="Boss" type="CharacterBody2D"]\nscript = ExtResource("1_b")\n\n'
            '[node name="Hitbox" type="Area2D" parent="."]\n\n'
            '[connection signal="body_entered" from="Hitbox" to="." method="take_damage"]\n'
            # Hitbox carries no script in this scene: nothing to resolve against
            '[connection signal="ready" from="." to="Hitbox" method="take_damage"]\n')
        r = extract_godot_resource(scene)
        stem = _file_stem((self.root / "scripts" / "enemy.gd").resolve())
        self.assertEqual([e["target"] for e in _edges(r, "connects")],
                         [_make_id(stem, "take_damage")])

    def test_project_godot_autoloads_and_main_scene(self):
        p = self.root / "project.godot"
        r = extract_godot_resource(p)

        self.assertTrue(_edges(r, "autoload"), "no autoload edge emitted")
        self.assertTrue(_edges(r, "main_scene"), "no main_scene edge emitted")

        gstate = _make_id(str((self.root / "scripts" / "game_state.gd").resolve()))
        script_targets = {e["target"] for e in _edges(r, "script")}
        self.assertIn(gstate, script_targets)

    def test_uid_autoload_resolves_through_its_sidecar(self):
        bus = self._write("scripts/audio_bus.gd", "extends Node\n")
        self._write("scripts/audio_bus.gd.uid", "uid://bprobe1234\n")
        p = self.root / "project.godot"
        p.write_text(p.read_text(encoding="utf-8") + 'AudioBus="*uid://bprobe1234"\n',
                     encoding="utf-8")
        r = extract_godot_resource(p)
        self.assertIn(_make_id(str(bus.resolve())), {e["target"] for e in _edges(r, "script")})

    def test_line_parser_fallback_matches(self):
        # Force the dependency-free line parser (as if the grammar were absent)
        # and confirm it still emits the same edges the default path produces.
        scene = self.root / "scenes" / "Main.tscn"
        default = extract_godot_resource(scene)
        saved = gr._RESOURCE_PARSER
        try:
            gr._RESOURCE_PARSER = None          # disable grammar -> line parser
            forced_lines = extract_godot_resource(scene)
        finally:
            gr._RESOURCE_PARSER = saved
        self.assertEqual(_norm(default), _norm(forced_lines))

    def test_line_parser_keeps_a_bracket_inside_a_header_value(self):
        text = (self.root / "weird.tscn").read_text(encoding="utf-8")
        blocks = gr._blocks_from_lines(
            text + '[connection signal="s" from="." to="." method="m" binds= [1, 2]]\n')
        self.assertEqual([b.kind for b in blocks],
                         ["gd_scene", "ext_resource", "node", "connection"])
        self.assertEqual(blocks[2].attrs["name"], "Odd]Name")
        self.assertEqual([k for k, _v, _loc in blocks[2].props], ["script"])
        self.assertEqual(blocks[3].attrs["method"], "m")


class TestGodotResourcePipeline(_FixtureProject):
    """Through extract(), which rewrites every id root-relative: this is where a
    scene's edge either reaches the script's real node or is left dangling."""

    def _extract(self):
        paths = sorted(p for p in self.root.rglob("*")
                       if p.suffix in (".gd", ".tscn", ".tres", ".godot"))
        return extract(paths, cache_root=Path(self.tmp.name) / "cache",
                       root=self.root, parallel=False)

    def _targets(self, r, relation, source_file):
        by_id = {n["id"]: n for n in r["nodes"]}
        return [(by_id[e["target"]]["label"], by_id[e["target"]]["source_file"])
                for e in _edges(r, relation) if e["source_file"] == source_file]

    def test_scene_and_project_edges_reach_the_real_nodes(self):
        r = self._extract()
        ids = [n["id"] for n in r["nodes"]]
        self.assertEqual(len(ids), len(set(ids)), "one node per id")
        for e in r["edges"]:
            self.assertIn(e["target"], ids, f"dangling edge: {e}")
        self.assertEqual(self._targets(r, "attaches_script", "scenes/Main.tscn"),
                         [("Enemy", "scripts/enemy.gd")])
        self.assertEqual(self._targets(r, "instances", "scenes/Main.tscn"),
                         [("Bullet.tscn", "scenes/Bullet.tscn")])
        self.assertEqual(self._targets(r, "connects", "scenes/Main.tscn"),
                         [("take_damage()", "scripts/enemy.gd")])
        self.assertEqual(self._targets(r, "main_scene", "project.godot"),
                         [("Main.tscn", "scenes/Main.tscn")])
        self.assertEqual(self._targets(r, "script", "project.godot"),
                         [("GameState", "scripts/game_state.gd")])

    def test_scene_beside_a_same_named_script_leaves_the_script_reachable(self):
        # enemy.tscn next to enemy.gd is the usual Godot layout. Were both given
        # the extensionless file id, the pipeline would salt them apart and the
        # `extends Enemy` edge from a third file would point at neither.
        self._write(
            "scripts/enemy.tscn",
            '[gd_scene format=3]\n\n'
            '[ext_resource type="Script" path="res://scripts/enemy.gd" id="1_e"]\n\n'
            '[node name="Enemy" type="CharacterBody2D"]\nscript = ExtResource("1_e")\n')
        self._write("scripts/grunt.gd", "extends Enemy\n")
        r = self._extract()
        enemy = ("Enemy", "scripts/enemy.gd")
        self.assertEqual(self._targets(r, "inherits", "scripts/grunt.gd"), [enemy])
        self.assertEqual(self._targets(r, "attaches_script", "scripts/enemy.tscn"), [enemy])

    def test_one_file_batch_names_the_same_nodes_as_a_full_run(self):
        # an update re-extracts only the edited file; the files its edges point
        # at are then not in the batch, and must still be named by their real ids
        full_ids = {n["id"] for n in self._extract()["nodes"]}
        for rel in ("scenes/Main.tscn", "project.godot"):
            one = extract([self.root / rel], cache_root=Path(self.tmp.name) / "cache_one",
                          root=self.root, parallel=False)
            own = {n["id"] for n in one["nodes"]}
            cross = [e["target"] for e in one["edges"] if e["target"] not in own]
            self.assertTrue(cross, rel)
            self.assertEqual([t for t in cross if t not in full_ids], [], rel)

    def test_scene_is_not_reported_as_a_file_without_symbols(self):
        # a scene or resource is one file node by design; what it holds is
        # carried by its edges, so naming it in the "no symbols" warning is noise
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self._extract()
        self.assertNotIn(".tscn", err.getvalue())
        # a script that declares nothing is still worth the warning
        self.assertIn("scripts/game_state.gd", err.getvalue())


@unittest.skipUnless(gr._load_resource_parser() is not None,
                     "godot_resource grammar (tree-sitter-language-pack) not installed")
class TestGodotResourceGrammar(_FixtureProject):
    """Behaviour specific to the grammar front end."""

    def test_grammar_and_line_parsers_agree(self):
        scene = self.root / "scenes" / "Main.tscn"
        text = scene.read_text()
        via_grammar = gr._build_scene(scene, gr._blocks_from_grammar(text))
        via_lines = gr._build_scene(scene, gr._blocks_from_lines(text))
        self.assertEqual(_norm(via_grammar), _norm(via_lines))

    def test_grammar_survives_bracket_in_quoted_value(self):
        # A ']' inside a quoted attribute value (node name "Odd]Name") is parsed
        # by the grammar as part of the value.
        scene = self.root / "weird.tscn"
        r = extract_godot_resource(scene)
        enemy_gd = _make_id(str((self.root / "scripts" / "enemy.gd").resolve()))
        attaches = {e["target"] for e in _edges(r, "attaches_script")}
        self.assertIn(enemy_gd, attaches)


if __name__ == "__main__":
    unittest.main()
