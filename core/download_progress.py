"""Report Hugging Face HTTP/Xet downloads without changing global progress hooks."""
from pathlib import Path
import time
from typing import Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import tqdm

ProgressCallback = Callable[[dict], None]


def download_file(repo_id: str, filename: str, *, revision: str,
                  progress: ProgressCallback | None = None, **kwargs) -> str:
    if progress is None:
        return hf_hub_download(repo_id, filename, revision=revision, **kwargs)
    progress({"phase": "downloading", "file": filename, "downloaded_bytes": 0, "total_bytes": None, "received_bytes": 0, "bytes_per_second": 0})

    class DownloadBar(tqdm):
        def __init__(self, *args, **options):
            self._completed = int(options.get("initial", 0) or 0)
            self._received = 0
            self._started = time.monotonic()
            self._xet = False
            options["disable"] = True
            super().__init__(*args, **options)
            self._report()

        def _report(self):
            total = self.total if self.total and self.total > 0 else None
            progress({"phase": "downloading", "file": filename,
                      "downloaded_bytes": min(self._completed, total) if total else self._completed,
                      "total_bytes": total, "received_bytes": self._received,
                      "bytes_per_second": self._received / max(time.monotonic() - self._started, 0.1)})

        def update(self, n=1):
            self._completed += n
            if not self._xet:
                self._received += n
            result = super().update(n)
            self.n = self._completed
            self._report()
            return result

        def set_transfer_postfix_str(self, *args, **kwargs):
            # huggingface_hub 2.x describes Xet network activity here; the app shows its own text.
            pass

        def update_transfer(self, n):
            # Xet can receive many compressed/deduplicated bytes before it
            # reconstructs a file chunk. Report network activity separately:
            # it has no reliable total and must not be a fake file percentage.
            if not self._xet:
                self._received = 0
            self._xet = True
            self._received += n
            self._report()

    path = hf_hub_download(repo_id, filename, revision=revision, tqdm_class=DownloadBar, **kwargs)
    size = Path(path).stat().st_size
    progress({"phase": "downloading", "file": filename, "downloaded_bytes": size, "total_bytes": size})
    return path
