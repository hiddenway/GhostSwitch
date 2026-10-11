from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import prepare_gguf_adapters as prep

UPSTREAM_LORA_AWARE = '''    def _reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
        if hasattr(tensor, 'get_lora_A_B'):
            pass

''' + prep.OUT_PROJ_REORDER


def checkout(folder: Path, qwen_source: str) -> Path:
    root = folder / "llama.cpp"
    (root / "conversion").mkdir(parents=True)
    (root / "gguf-py").mkdir()
    (root / "convert_lora_to_gguf.py").write_text("")
    (root / "conversion" / "qwen.py").write_text(qwen_source)
    return root


class ConverterPatchTests(unittest.TestCase):
    def converted(self, qwen_source: str) -> str:
        with tempfile.TemporaryDirectory() as folder:
            root = checkout(Path(folder), qwen_source)
            work = Path(folder) / "work"
            work.mkdir()
            prep._patched_converter(root, work)
            return (work / "llama.cpp" / "conversion" / "qwen.py").read_text()

    def test_llama_cpp_that_reorders_lora_factors_is_used_unchanged(self):
        self.assertEqual(self.converted(UPSTREAM_LORA_AWARE), UPSTREAM_LORA_AWARE)

    def test_older_llama_cpp_gets_a_column_permutation_without_a_device_move(self):
        patched = self.converted(prep.OUT_PROJ_REORDER)
        self.assertIn("_lora_A[:, col_perm]", patched)
        # Moving the index to the converter's lazy meta device wrote zeros into ssm_out.lora_a.
        self.assertNotIn(".to(device", patched)

    def test_files_from_an_older_converter_are_rebuilt(self):
        converted = {"repo_id": "a/b", "revision": "1", "gguf_file": "x.gguf"}
        ready = {**converted, "gguf_source_filename": "x.gguf"}
        self.assertEqual(prep.adapter_metadata(converted)["converter"], prep.CONVERTER_VERSION)
        # Ready-made GGUF downloads are not converted, so they keep their metadata.
        self.assertNotIn("converter", prep.adapter_metadata(ready))

    def test_unknown_converter_revision_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported llama.cpp converter revision"):
            self.converted("class Qwen: pass\n")


if __name__ == "__main__":
    unittest.main()
