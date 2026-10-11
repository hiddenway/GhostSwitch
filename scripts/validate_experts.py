"""Validate contributed expert packages and composed routes without model downloads."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.expert_packages import load_packages, package_router_config
from core.expert_catalog import ExpertCatalog
from core_models import backbone_ids


def main() -> int:
    try:
        ExpertCatalog.load()
        for backbone in backbone_ids():
            packages = load_packages(backbone)
            raw = json.loads((ROOT / "core/config/router_tree.json").read_text())
            tree = package_router_config(raw, packages)
            print(f"Valid {backbone}: {len(packages)} expert packages, {len(tree['nodes'])} routing nodes")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Invalid expert catalog: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
