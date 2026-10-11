from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core import lora_runtime
from core.lora_runtime import AdapterSpec, LoraManager, PreparedAdapter

BASE = "test-base"


class FakeModel:
    def update_modules(self, modules):
        pass


def manager(budget: int, sizes: dict[str, int]) -> LoraManager:
    """A manager whose adapters are placeholders of the given sizes, prepared without MLX."""
    result = LoraManager(memory_budget_bytes=budget)
    result._prepare = lambda name, model, base: (PreparedAdapter({}, {}, sizes[name], 1), 0.0, 0.0, 0.0)
    return result


@unittest.skipIf(lora_runtime.tree_unflatten is None, "the LoRA manager needs MLX")
class LoraMemoryBudgetTest(unittest.TestCase):
    def setUp(self):
        specs = {name: AdapterSpec("owner/repo", "0" * 40, BASE) for name in "abcde"}
        patcher = patch.dict(lora_runtime.ADAPTERS, specs, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.model = FakeModel()

    def use(self, adapters: LoraManager, name: str) -> dict:
        with adapters.activate(name, self.model, BASE) as timings:
            return dict(timings)

    def test_least_recently_used_adapter_goes_first(self):
        adapters = manager(300, dict.fromkeys("abcde", 100))
        for name in "abc":
            self.assertEqual(self.use(adapters, name)["evicted_adapters"], [])
        self.assertEqual(self.use(adapters, "d")["evicted_adapters"], ["a"])
        self.assertTrue(self.use(adapters, "b")["cache_hit"])
        self.assertEqual(self.use(adapters, "e")["evicted_adapters"], ["c"])
        self.assertEqual(adapters.cached_adapters, ("d", "b", "e"))

    def test_adapter_larger_than_budget_is_kept_while_used(self):
        adapters = manager(50, {"a": 100, "b": 30})
        self.assertEqual(self.use(adapters, "a")["evicted_adapters"], [])
        self.assertEqual(adapters.cached_adapters, ("a",))
        self.assertEqual(self.use(adapters, "b")["evicted_adapters"], ["a"])
        self.assertEqual(adapters.cached_adapters, ("b",))

    def test_zero_budget_keeps_only_the_latest_adapter(self):
        adapters = manager(0, dict.fromkeys("abc", 10))
        for name in "abc":
            self.use(adapters, name)
            self.assertEqual(adapters.cached_adapters, (name,))
        self.assertFalse(self.use(adapters, "a")["cache_hit"])

    def test_active_adapter_is_never_evicted(self):
        adapters = manager(0, dict.fromkeys("ab", 10))
        with adapters.activate("a", self.model, BASE):
            self.assertEqual(adapters._evict(keep="b"), [])
            self.assertIn("a", adapters.cached_adapters)
            with self.assertRaises(RuntimeError):
                adapters.unload("a")

    def test_budget_from_environment(self):
        for value, expected in (("0", 0), ("1024", 1024 * 1024 * 1024), ("", 512 * 1024 * 1024),
                                ("many", 512 * 1024 * 1024)):
            with patch.dict(os.environ, {"GHOSTSWITCH_LORA_MEMORY_MB": value}):
                self.assertEqual(LoraManager()._memory_budget, expected)


if __name__ == "__main__":
    unittest.main()
