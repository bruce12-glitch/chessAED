"""Update pipeline manifest tests for shipped-locked v2 (companion patch)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, text.count(old))
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)


DRAFT_RESET = (
    "            m2 = MF.load(p)\n"
    "            m2[\"lock\"] = {\"locked\": False, \"manifest_hash\": None}\n"
    "            m2[\"scenario\"][\"digest\"] = None\n"
    "            m2[\"code\"] = {\"git_commit\": None, \"file_sha256\": None}\n"
    "            m2[\"evaluation\"][\"probes\"][\"reference\"] = None\n"
    "            for a in m2[\"models\"]:\n"
    "                m2[\"models\"][a][\"param_count\"] = None\n"
    "            import yaml as _yaml\n"
    "            p.write_text(_yaml.safe_dump(m2, sort_keys=False), encoding=\"utf-8\")\n"
)

patch("tests/test_pipeline.py",
      "    def test_lock_verify_tamper_and_relock(self) -> None:\n"
      "        with TemporaryDirectory() as d:\n"
      "            p = self._copy(Path(d))\n"
      "            m = MF.lock(p, param_counts={\"cnn\": 111, \"gnn\": 222}, probe_reference={\"mixed_midgame\": 5})",
      "    def test_lock_verify_tamper_and_relock(self) -> None:\n"
      "        with TemporaryDirectory() as d:\n"
      "            p = self._copy(Path(d))\n"
      + DRAFT_RESET +
      "            m = MF.lock(p, param_counts={\"cnn\": 111, \"gnn\": 222}, probe_reference={\"mixed_midgame\": 5})")

patch("tests/test_pipeline.py",
      "    def test_unlocked_refused_when_required(self) -> None:\n"
      "        with self.assertRaises(MF.ManifestError):\n"
      "            MF.verify(MF.load(), require_locked=True)",
      "    def test_unlocked_refused_when_required(self) -> None:\n"
      "        m = MF.clone(MF.load())\n"
      "        m[\"lock\"] = {\"locked\": False, \"manifest_hash\": None}\n"
      "        with self.assertRaises(MF.ManifestError):\n"
      "            MF.verify(m, require_locked=True)")

patch("tests/test_pipeline.py",
      "    def test_copy_into_idempotent_and_refuses_mixing(self) -> None:",
      "    def test_copy_into_preserves_source_filename(self) -> None:\n"
      "        with TemporaryDirectory() as d:\n"
      "            p = self._copy(Path(d))\n"
      "            out = Path(d) / \"out\"\n"
      "            dst = MF.copy_into(out, p)\n"
      "            self.assertEqual(dst.name, p.name)\n"
      "\n"
      "    def test_copy_into_idempotent_and_refuses_mixing(self) -> None:")
