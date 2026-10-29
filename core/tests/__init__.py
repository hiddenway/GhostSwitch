"""Unit tests. None of them may load or download a real model.

Router v7 calls the model directly instead of through llm.chat, so a test that forgets to
patch it would load a 7 GB checkpoint (or download it). Loading fails loudly instead; a
test that needs a model patches llm.get_model with a fake.
"""

from core import llm


def _no_model(role: str):
    raise RuntimeError(f"unit tests must not load a model (role {role!r}); "
                       "patch llm.chat, llm.get_model or router_v7._scorer")


llm.MODEL_MANAGER.get_model = _no_model
