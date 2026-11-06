"""Every required numerical oracle must be obtainable from THIS repository, and its manifest record must be true.

The `tests/data/*.npz` fixtures are mandatory (conftest fails collection when one is missing, since a skipped oracle
still reports green). `oracle.yml` builds them on a clean runner, which proves the generators need nothing outside the
repository; these tests pin the locations and the manifest, and do not run the generators.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
GEN = TESTS / "gen_fixtures.py"


def _gen_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_fixtures", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixture_generators_live_in_repo():
    gen = _gen_module()
    for key, spec in gen.GENERATORS.items():
        script = (ROOT / spec["argv"][1]).resolve()
        assert script.is_relative_to(ROOT), f"{key} is regenerated from {script}, outside this repository"
        assert script.is_file(), f"{key}: missing generator {script}"


def _fake_fixtures(gen, tmp_path, monkeypatch):
    """Point the module at placeholder fixtures, with generators that only rewrite their bytes."""
    data = tmp_path / "data"
    data.mkdir()

    def run_generator(key):
        for name in gen.GENERATORS[key]["writes"]:
            (data / name).write_bytes(b"built-by-" + key.encode() + name.encode())

    monkeypatch.setattr(gen, "FIXTURE_DIR", data)
    monkeypatch.setattr(gen, "MANIFEST", data / "manifest.json")
    monkeypatch.setattr(gen, "run_generator", run_generator)
    return data


def test_manifest_verifies_and_records_what_made_each_file(tmp_path, monkeypatch):
    gen = _gen_module()
    data = _fake_fixtures(gen, tmp_path, monkeypatch)
    assert gen.verify() == 1  # no manifest yet

    assert gen.main(["--all"]) == 0
    fixtures = json.loads(gen.MANIFEST.read_text())["fixtures"]
    assert set(fixtures) == set(gen.ALL_FIXTURES)
    entry = fixtures[gen.ALL_FIXTURES[0]]
    assert entry["vulcan3_commit"]
    assert entry["versions"]["jax"]
    assert entry["config_sha256_16"]
    assert entry["generator"]
    assert entry["note"]

    victim = data / gen.ALL_FIXTURES[0]
    victim.write_bytes(victim.read_bytes() + b"!")
    assert gen.verify() == 1
    assert gen.main(["--all"]) == 1  # a skipped fixture is still held to its record


def test_a_fixture_keeps_the_provenance_of_the_run_that_made_it(tmp_path, monkeypatch):
    """A later commit relabels only what it rebuilt: `--all` over present files and `--only` on one generator."""
    gen = _gen_module()
    _fake_fixtures(gen, tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "_git_head", lambda: "first")
    assert gen.main(["--all"]) == 0
    before = gen.MANIFEST.read_bytes()

    monkeypatch.setattr(gen, "_git_head", lambda: "second")
    assert gen.main(["--all"]) == 0
    assert gen.MANIFEST.read_bytes() == before

    assert gen.main(["--only", "photo", "--force"]) == 0
    fixtures = json.loads(gen.MANIFEST.read_text())["fixtures"]
    commits = {name: entry["vulcan3_commit"] for name, entry in fixtures.items()}
    rebuilt = set(gen.GENERATORS["photo"]["writes"])
    assert commits == {name: ("second" if name in rebuilt else "first") for name in gen.ALL_FIXTURES}


def test_a_record_without_provenance_is_refused_with_the_remedy(tmp_path, monkeypatch, capsys):
    """A manifest from before per-fixture records, or from `--only`, reads as a plain refusal, not a traceback."""
    gen = _gen_module()
    data = _fake_fixtures(gen, tmp_path, monkeypatch)
    assert gen.main(["--all"]) == 0
    assert gen.built_on(gen.ALL_FIXTURES[0]) == gen.platform.machine()
    assert gen.built_on("no_such_fixture.npz") is None

    manifest = json.loads(gen.MANIFEST.read_text())
    record = manifest["fixtures"][gen.ALL_FIXTURES[0]]
    manifest["fixtures"][gen.ALL_FIXTURES[0]] = {"sha256": record["sha256"]}
    gen.MANIFEST.write_text(json.dumps(manifest))
    assert gen.built_on(gen.ALL_FIXTURES[0]) is None
    assert gen.verify() == 1
    assert "--force" in capsys.readouterr().err
    (data / "manifest.json").unlink()
    assert gen.built_on(gen.ALL_FIXTURES[0]) is None
