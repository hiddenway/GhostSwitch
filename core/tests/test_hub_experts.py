from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import expert_packages, hub_experts

SHA = "a" * 40


def model(repo, files, **extra):
    return {"id": repo, "author": repo.split("/")[0], "sha": SHA, "downloads": extra.pop("downloads", 1), "likes": 0,
            "siblings": [{"rfilename": name} for name in files], **extra}


LISTING = [
    model("good/medical-lora", ["adapter_config.json", "adapter_model.safetensors", "README.md"], downloads=50),
    model("nested/lora", ["adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                          "checkpoint-50/adapter_config.json", "checkpoint-50/adapter_model.safetensors"]),
    model("full/model", ["model.safetensors", "config.json"]),
    model("secret/lora", ["adapter_config.json", "adapter_model.safetensors"], gated="auto"),
    model("MaliDDD/gpqa_marginal_9b-sft_1.0", ["adapter_config.json", "adapter_model.safetensors"]),
    model("blackshell69/Qwen3.5-9B-distilled-coding-agentic", ["adapter_config.json", "adapter_model.safetensors"]),
    model("Muneerali199/rakshak-cwe-v2", ["adapter_config.json", "adapter_model.safetensors"]),
]


class HubTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.user = Path(folder.name)
        environment = patch.dict(os.environ, {"GHOSTSWITCH_USER_EXPERTS": str(self.user)})
        environment.start()
        self.addCleanup(environment.stop)
        expert_packages.reload_packages()
        self.addCleanup(expert_packages.reload_packages)
        hub_experts._cache.clear()

    def fake_get(self, base="Qwen/Qwen3.5-9B"):
        def get(url, raw=False):
            if "/api/models?" in url:
                return LISTING if "Qwen%2FQwen3.5-9B" in url else []
            if "/api/models/" in url:
                repo = url.split("/api/models/")[1]
                return next(row for row in LISTING if row["id"] == repo)
            if url.endswith("adapter_config.json"):
                return {"peft_type": "LORA", "base_model_name_or_path": base}
            if url.endswith("README.md"):
                return "---\ntags: [lora]\n---\n# Medical LoRA\n\nAnswers medical exam questions in four options.\n"
            raise AssertionError(url)
        return get

    def test_discovery_offers_public_compatible_adapters_only(self):
        with patch.object(hub_experts, "_get", side_effect=self.fake_get()):
            rows = hub_experts.discover("qwen35-9b")
        self.assertEqual([row["repo_id"] for row in rows], ["good/medical-lora", "nested/lora"])
        self.assertEqual(rows[1]["adapter_dir"], "adapter")
        reasons = hub_experts.blocked_repos("qwen35-9b")
        self.assertIn("MaliDDD/gpqa_marginal_9b-sft_1.0", reasons)
        self.assertIn("Muneerali199/rakshak-cwe-v2", reasons)

    def test_added_adapter_is_a_manual_expert_outside_the_router(self):
        with patch.object(hub_experts, "_get", side_effect=self.fake_get()):
            expert_id = hub_experts.install("qwen35-9b", "good/medical-lora")
            rows = hub_experts.discover("qwen35-9b")
        package = expert_packages.load_packages("qwen35-9b")[expert_id]
        self.assertEqual(expert_id, "hub_good_medical_lora")
        self.assertTrue(package["profile"]["manual_only"])
        self.assertEqual(package["profile"]["description"], "Answers medical exam questions in four options.")
        self.assertEqual(package["adapter"]["revision"], SHA)
        self.assertEqual(rows[0]["installed"], expert_id)
        raw = json.loads((Path(expert_packages.__file__).parent / "config" / "router_tree.json").read_text())
        tree = expert_packages.package_router_config(raw, expert_packages.load_packages("qwen35-9b"))
        self.assertNotIn(expert_id, {node.get("expert_id") for node in tree["nodes"]})
        hub_experts.remove("qwen35-9b", expert_id)
        self.assertNotIn(expert_id, expert_packages.load_packages("qwen35-9b"))
        with self.assertRaises(KeyError):
            hub_experts.remove("qwen35-9b", "coding_expert")

    def test_adapters_for_other_models_and_blocked_repos_are_refused(self):
        with patch.object(hub_experts, "_get", side_effect=self.fake_get("Jackrong/Qwen3.5-9B-Opus-Distilled")):
            with self.assertRaisesRegex(ValueError, "not on this base model"):
                hub_experts.install("qwen35-9b", "good/medical-lora")
        with patch.object(hub_experts, "_get", side_effect=self.fake_get("unsloth/Qwen3.5-9B")):
            self.assertTrue(hub_experts.install("qwen35-9b", "good/medical-lora").startswith("hub_"))
        with self.assertRaisesRegex(ValueError, "blocked"):
            hub_experts.install("qwen35-9b", "MaliDDD/gpqa_marginal_9b-sft_1.0")


if __name__ == "__main__":
    unittest.main()
