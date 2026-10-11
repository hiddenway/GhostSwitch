"""Make the CUDA runtime from NVIDIA's pip packages loadable before llama.cpp starts.

The llama-cpp-python CUDA wheels link the CUDA runtime and cuBLAS dynamically but ship neither.
The installer adds them as nvidia-cuda-runtime-cu12 and nvidia-cublas-cu12, whose libraries sit in
site-packages where no loader looks. Without those packages (a CPU build, or a CUDA Toolkit that
llama-cpp-python finds through CUDA_PATH) this does nothing.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# cuBLAS needs cuBLASLt, and both need the runtime, so they load in this order.
_LINUX_LIBRARIES = (("cuda_runtime", "libcudart.so.12"), ("cublas", "libcublasLt.so.12"),
                    ("cublas", "libcublas.so.12"))
_WINDOWS_FOLDERS = ("cuda_runtime", "cublas")
_prepared = False


def _nvidia_roots() -> list[Path]:
    try:
        import nvidia  # The namespace package the nvidia-* wheels install into.
    except ImportError:
        return []
    return [Path(path) for path in getattr(nvidia, "__path__", [])]


def prepare() -> None:
    """Call once before the first llama_cpp import; later calls do nothing."""
    global _prepared
    if _prepared:
        return
    _prepared = True
    roots = _nvidia_roots()
    if sys.platform == "win32":
        folders = [root / name / "bin" for root in roots for name in _WINDOWS_FOLDERS
                   if (root / name / "bin").is_dir()]
        for folder in folders:
            os.add_dll_directory(str(folder))
        if folders:
            # llama-cpp-python loads its DLLs with the standard search order, which reads PATH.
            os.environ["PATH"] = os.pathsep.join([*map(str, folders), os.environ.get("PATH", "")])
    elif sys.platform.startswith("linux"):
        for name, library in _LINUX_LIBRARIES:
            path = next((root / name / "lib" / library for root in roots
                         if (root / name / "lib" / library).is_file()), None)
            if path is None:
                continue
            try:
                # A library already loaded under its soname satisfies llama.cpp's dependency on it.
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # llama_cpp's own import then reports what is missing.
                continue
