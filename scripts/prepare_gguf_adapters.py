"""Download the pinned LoRAs of a Qwen3.5 backbone and prepare llama.cpp GGUF files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.expert_packages import load_packages
from core.file_lock import exclusive
from core.download_progress import download_file, ProgressCallback
from core_models import active_backbone, hf_base_repo, local_model_dir, user_dir
OUT_PROJ_REORDER = '''            elif ".out_proj." in name:
                # Out projection weight: reorder columns (input dimension)
                data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)
'''
PATCHED_OUT_PROJ_REORDER = '''            elif ".out_proj." in name:
                # The column permutation acts on LoRA A's input dimension.
                if hasattr(data_torch, "_lora_A") and hasattr(data_torch, "_lora_B"):
                    col_perm = self._reorder_v_heads(
                        torch.arange(num_v_heads * head_v_dim, dtype=torch.long).unsqueeze(0),
                        1, num_k_heads, num_v_per_k, head_v_dim,
                    ).squeeze(0)
                    # Index with the real tensor: moving it to the converter's lazy (meta)
                    # device drops its values and writes zeros or garbage into the GGUF.
                    data_torch = data_torch.__class__(data_torch._lora_A[:, col_perm], data_torch._lora_B)
                else:
                    data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)
'''


def _patched_converter(llama_cpp_dir: Path, temporary: Path) -> Path:
    source_script = llama_cpp_dir / "convert_lora_to_gguf.py"
    source_conversion = llama_cpp_dir / "conversion"
    gguf_python = llama_cpp_dir / "gguf-py"
    if not source_script.is_file() or not source_conversion.is_dir() or not gguf_python.is_dir():
        raise FileNotFoundError(
            "--llama-cpp-dir must point to a llama.cpp checkout containing "
            "convert_lora_to_gguf.py, conversion/, and gguf-py/."
        )
    target = temporary / "llama.cpp"
    target.mkdir()
    shutil.copy2(source_script, target / source_script.name)
    shutil.copytree(source_conversion, target / "conversion")
    (target / "gguf-py").symlink_to(gguf_python.resolve())
    qwen = target / "conversion" / "qwen.py"
    content = qwen.read_text(encoding="utf-8")
    # Newer llama.cpp reorders LoRA factors itself; patch only revisions that do not.
    if "get_lora_A_B" not in content and PATCHED_OUT_PROJ_REORDER not in content:
        if OUT_PROJ_REORDER not in content:
            raise RuntimeError(
                "Unsupported llama.cpp converter revision: Qwen linear-attention "
                "out_proj conversion block was not recognized."
            )
        qwen.write_text(content.replace(OUT_PROJ_REORDER, PATCHED_OUT_PROJ_REORDER), encoding="utf-8")
    return target / source_script.name


def _ensure_base_config(backbone: str, progress: ProgressCallback | None = None) -> None:
    base_dir = local_model_dir(backbone)
    base_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (base_dir / filename).is_file():
            download_file(hf_base_repo(backbone), filename, revision="main", local_dir=base_dir, progress=progress)


# Raised when converted files must be rebuilt. 2: linear-attention out_proj LoRA factors were
# zeroed (and layer 0 filled with garbage) by the earlier out_proj patch.
CONVERTER_VERSION = 2


def adapter_metadata(row: dict) -> dict:
    value = {
        "repo_id": row["repo_id"],
        "revision": row["revision"],
        "source_subdir": row.get("source_subdir", ""),
        "gguf_repo_id": row.get("gguf_repo_id"),
        "gguf_revision": row.get("gguf_revision"),
        "source_config": row.get("source_config_filename"),
        "source_file": row.get("gguf_source_filename") or row.get("source_filename"),
    }
    if not row.get("gguf_source_filename"):
        value["converter"] = CONVERTER_VERSION
    return value


def _prepare_one(
    expert_id: str,
    row: dict[str, object],
    output_dir: Path,
    download_dir: Path,
    converter: Path,
    converter_root: Path,
    progress: ProgressCallback | None = None,
    backbone: str | None = None,
) -> Path:
    output = output_dir / str(row["gguf_file"])
    metadata_path = output.with_suffix(output.suffix + ".json")
    expected_metadata = adapter_metadata(row)
    try:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing_metadata = None
    if output.is_file() and output.stat().st_size and existing_metadata == expected_metadata:
        print(f"Ready: {expert_id} -> {output}", flush=True)
        return output
    temporary_output = output.with_name(output.name + ".partial")

    repo_id = str(row["repo_id"])
    revision = str(row["revision"])
    if row.get("gguf_source_filename"):
        source_repo = str(row.get("gguf_repo_id") or repo_id)
        source_revision = str(row.get("gguf_revision") or revision)
        cached = download_file(
            source_repo, str(row["gguf_source_filename"]), revision=source_revision,
            local_dir=download_dir / expert_id, progress=progress,
        )
        if progress:
            progress({"phase": "saving"})
        shutil.copy2(cached, temporary_output)
        os.replace(temporary_output, output)
        _write_metadata(metadata_path, expected_metadata)
        print(f"Downloaded: {expert_id} -> {output}", flush=True)
        return output

    peft_dir = download_dir / expert_id
    peft_dir.mkdir(parents=True, exist_ok=True)
    for logical, filename in (("adapter_config.json", row.get("source_config_filename", "adapter_config.json")),
                              ("adapter_model.safetensors", row.get("source_filename", "adapter_model.safetensors"))):
        cached = download_file(repo_id, str(filename), revision=revision, progress=progress)
        shutil.copy2(cached, peft_dir / logical)
    command = [
        sys.executable, str(converter), str(peft_dir), "--base", str(local_model_dir(backbone or active_backbone())),
        "--outfile", str(temporary_output), "--outtype", "f16",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(converter_root), str(converter_root / "gguf-py"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    print(f"Converting: {expert_id}", flush=True)
    if progress:
        progress({"phase": "converting"})
    subprocess.run(command, check=True, env=environment)
    if progress:
        progress({"phase": "saving"})
    os.replace(temporary_output, output)
    _write_metadata(metadata_path, expected_metadata)
    return output


def _write_metadata(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_expert(expert_id: str, cache_dir: Path, llama_cpp_dir: Path | None = None,
                   progress: ProgressCallback | None = None, backbone: str | None = None) -> Path:
    backbone = backbone or active_backbone()
    row = load_packages(backbone)[expert_id]["adapter"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with exclusive(cache_dir / (expert_id + ".lock")):
        output = cache_dir / row["gguf_file"]
        if output.is_file() and output.stat().st_size:
            try:
                if json.loads(output.with_suffix(output.suffix + ".json").read_text()) == adapter_metadata(row):
                    return output
            except (OSError, ValueError):
                pass
        with tempfile.TemporaryDirectory(prefix="ghostswitch-lora-converter-") as temp:
            temporary = Path(temp)
            converter = temporary / "unused-converter"
            if not row.get("gguf_source_filename"):
                checkout = llama_cpp_dir or (Path(os.environ["LLAMA_CPP_DIR"]) if os.environ.get("LLAMA_CPP_DIR") else None)
                if checkout is None:
                    raise RuntimeError("This expert needs PEFT to GGUF conversion. Set LLAMA_CPP_DIR or choose an expert with a ready GGUF source.")
                _ensure_base_config(backbone, progress)
                converter = _patched_converter(checkout.resolve(), temporary)
            return _prepare_one(expert_id, row, cache_dir, temporary / "downloads", converter, converter.parent,
                                progress, backbone)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-cpp-dir", type=Path,
        default=Path(os.environ["LLAMA_CPP_DIR"]) if os.environ.get("LLAMA_CPP_DIR") else None,
        help="llama.cpp checkout containing convert_lora_to_gguf.py (or LLAMA_CPP_DIR).",
    )
    parser.add_argument("--backbone", default=active_backbone(), help="Backbone whose experts to prepare.")
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Output directory; defaults to GHOSTSWITCH_GGUF_LORA_DIR or ~/.cache/ghostswitch/gguf-lora/<backbone>.",
    )
    parser.add_argument(
        "--experts", nargs="+",
        help="Prepare only these expert ids; by default, prepare every manifest entry.",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir or Path(os.environ.get(
        "GHOSTSWITCH_GGUF_LORA_DIR", user_dir(Path.home() / ".cache") / "gguf-lora" / args.backbone,
    ))
    adapters = {eid: p["adapter"] for eid, p in load_packages(args.backbone).items()}
    selected = args.experts or list(adapters)
    unknown = set(selected) - set(adapters)
    if unknown:
        parser.error(f"Unknown expert id(s): {', '.join(sorted(unknown))}")
    for expert_id in selected:
        prepare_expert(expert_id, cache_dir, args.llama_cpp_dir, backbone=args.backbone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
