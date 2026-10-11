"""GhostSwitch was called OmniBrain and, before that, MiniBrain: setups made under either name keep working."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core_models
from core.request_contract import parse_request
from core.request_journal import RequestJournal

MESSAGES = [{"role": "user", "content": "Hello"}]


class LegacyNamesTest(unittest.TestCase):
    def test_old_variables_stand_in_for_unset_new_ones(self):
        environ = {"OMNIBRAIN_HOME": "/omni", "MINIBRAIN_HOME": "/mini", "MINIBRAIN_MAX_IMAGES": "2",
                   "OMNIBRAIN_GGUF_CONTEXT": "8192", "GHOSTSWITCH_GGUF_CONTEXT": "4096"}
        core_models.map_legacy_environment(environ)
        self.assertEqual(environ["GHOSTSWITCH_HOME"], "/omni")
        self.assertEqual(environ["GHOSTSWITCH_MAX_IMAGES"], "2")
        self.assertEqual(environ["GHOSTSWITCH_GGUF_CONTEXT"], "4096")

    def test_folders_from_before_the_renames_stay_in_use(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=False) as environ:
            environ.pop("GHOSTSWITCH_HOME", None)
            root = Path(directory)
            self.assertEqual(core_models.user_dir(root), root / "ghostswitch")
            (root / "minibrain").mkdir()
            self.assertEqual(core_models.user_dir(root), root / "minibrain")
            (root / "omnibrain").mkdir()
            self.assertEqual(core_models.user_dir(root), root / "omnibrain")
            (root / "ghostswitch").mkdir()
            self.assertEqual(core_models.user_dir(root), root / "ghostswitch")

    def test_adapters_added_from_hugging_face_before_the_rename_stay(self):
        from core.expert_packages import user_root
        with tempfile.TemporaryDirectory() as home, patch.dict("os.environ", {"HOME": home}) as environ:
            environ.pop("GHOSTSWITCH_USER_EXPERTS", None)
            config = Path(home) / ".config"
            self.assertEqual(user_root(), config / "ghostswitch" / "experts")
            (config / "omnibrain" / "experts").mkdir(parents=True)
            self.assertEqual(user_root(), config / "omnibrain" / "experts")
            (config / "ghostswitch" / "experts").mkdir(parents=True)
            self.assertEqual(user_root(), config / "ghostswitch" / "experts")

    def test_old_model_ids_are_accepted(self):
        for model in ("ghostswitch-v6", "omnibrain-v6", "minibrain-v6"):
            self.assertEqual(parse_request({"model": model, "messages": MESSAGES})["messages"][0]["content"], "Hello")
        with self.assertRaises(ValueError):
            parse_request({"model": "gpt-4", "messages": MESSAGES})

    def test_journal_records_move_to_the_new_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, date INTEGER, body TEXT)")
            for identifier, key in (("a", "omnibrain"), ("b", "minibrain")):
                record = {"id": identifier, "date": 1, "response": {"status": "done", key: {"route": "base"}}}
                db.execute("INSERT INTO requests VALUES (?, 1, ?)", (identifier, json.dumps(record)))
            db.commit()
            db.close()
            for record in RequestJournal(path).list():
                self.assertEqual(record["response"]["ghostswitch"], {"route": "base"})
                self.assertNotIn("omnibrain", record["response"])
                self.assertNotIn("minibrain", record["response"])


if __name__ == "__main__":
    unittest.main()
