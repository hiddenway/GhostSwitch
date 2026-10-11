"""Images and files in chat messages, given as OpenAI content parts.

A user message may carry ``text``, ``image_url`` (a base64 ``data:`` URL) and
``file`` parts. Files become text: plain text and code are decoded, PDF and DOCX
text is extracted. Images stay pixels for the model; the message text names
each one with a label such as ``[Image 1: photo.png]`` so routing, search
planning and review, which read text, know an image is there.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import math
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() and int(value) > 0 else default


# The model sees at most this many images per request: the most recent ones.
MAX_IMAGES = _env_int("GHOSTSWITCH_MAX_IMAGES", 4)
# Larger images are scaled down first; about one model token per 32x32 pixels.
MAX_IMAGE_PIXELS = _env_int("GHOSTSWITCH_IMAGE_MAX_PIXELS", 1024 * 1024)
MAX_SOURCE_PIXELS = 64 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
# Text taken from one file; the rest is cut with a visible note.
MAX_FILE_CHARS = _env_int("GHOSTSWITCH_MAX_FILE_CHARS", 100_000)
_DATA_URL = re.compile(r"data:([\w.+-]+/[\w.+-]+)?((?:;[\w.+-]+=[^;,]*)*)(;base64)?,(.*)", re.S)
_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(eq=False)
class ImageInput:
    """One decoded RGB image, scaled to the pixel budget."""

    name: str
    image: Any  # PIL.Image.Image in RGB mode
    digest: str
    label: str = ""
    _png: bytes | None = field(default=None, repr=False)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def png_bytes(self) -> bytes:
        if self._png is None:
            buffer = io.BytesIO()
            self.image.save(buffer, format="PNG")
            self._png = buffer.getvalue()
        return self._png


def _decode_data_url(url: Any, what: str) -> tuple[str, bytes]:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError(f"{what} must be a base64 data: URL; the server does not download remote files")
    match = _DATA_URL.fullmatch(url)
    if not match or not match.group(3):
        raise ValueError(f"{what} must be a base64 data: URL")
    try:
        data = base64.b64decode(match.group(4), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} contains invalid base64 data") from exc
    if not data:
        raise ValueError(f"{what} is empty")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"{what} exceeds {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB")
    return (match.group(1) or "application/octet-stream").lower(), data


def _clean_name(value: Any, default: str) -> str:
    name = value.strip() if isinstance(value, str) else ""
    name = re.sub(r"[\x00-\x1f\[\]<>]", "", name.replace("\\", "/").rsplit("/", 1)[-1])[:120]
    return name or default


def load_image(data: bytes, name: str) -> ImageInput:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError("Image input needs Pillow on the server; install requirements-common.txt") from exc
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
                raise ValueError(f"Image {name} is too large ({width}x{height})")
            if max(width, height) / min(width, height) > 100:
                raise ValueError(f"Image {name} is too narrow; its sides differ more than 100 times")
            opened.seek(0)  # The first frame of an animation.
            image = ImageOps.exif_transpose(opened)
            image.load()
    except ValueError:
        raise
    except Exception as exc:  # Pillow raises several types for undecodable data.
        raise ValueError(f"Could not read image {name}; send PNG, JPEG, WebP, GIF or BMP") from exc
    if image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        image = canvas
    else:
        image = image.convert("RGB")
    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / (width * height))
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                             Image.Resampling.BICUBIC)
    digest = hashlib.sha256(image.tobytes() + repr(image.size).encode()).hexdigest()
    return ImageInput(name=name, image=image, digest=digest)


def _decode_text(data: bytes, name: str) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unsupported file {name}: send text, code, PDF or DOCX files") from exc
    if "\x00" in text:
        raise ValueError(f"Unsupported file {name}: send text, code, PDF or DOCX files")
    return text.replace("\r\n", "\n")


def _pdf_text(data: bytes, name: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF files need pypdf on the server; install requirements-common.txt") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"PDF {name} is password protected")
        pages = []
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[Page {number}]\n{text}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF {name}") from exc
    if not pages:
        raise ValueError(f"PDF {name} has no text layer; send its pages as images instead")
    return "\n\n".join(pages)


def _docx_text(data: bytes, name: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"Could not read DOCX {name}") from exc
    paragraphs = []
    for paragraph in root.iter(f"{_DOCX_NS}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{_DOCX_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_DOCX_NS}tab":
                parts.append("\t")
            elif node.tag in {f"{_DOCX_NS}br", f"{_DOCX_NS}cr"}:
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs).strip()


def file_text(data: bytes, mime: str, name: str) -> str:
    """The text a file contributes to the prompt, cut to MAX_FILE_CHARS."""
    lower = name.lower()
    if mime == "application/pdf" or data.startswith(b"%PDF-"):
        text = _pdf_text(data, name)
    elif lower.endswith(".docx") or "wordprocessingml" in mime:
        text = _docx_text(data, name)
    elif mime.startswith("image/"):
        raise ValueError(f"Send image {name} as an image_url part")
    else:
        text = _decode_text(data, name)
    if len(text) > MAX_FILE_CHARS:
        text = (text[:MAX_FILE_CHARS]
                + f"\n[File truncated: showing the first {MAX_FILE_CHARS:,} of {len(text):,} characters]")
    return text


def parse_parts(content: list, role: str) -> list[str | ImageInput]:
    """Content parts as text segments and images, in their original order."""
    if not content:
        raise ValueError("Content part arrays must not be empty")
    segments: list[str | ImageInput] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            if not isinstance(part.get("text"), str) or set(part) - {"type", "text"}:
                raise ValueError("A text part needs a string text field")
            segments.append(part["text"])
            continue
        if role != "user":
            raise ValueError("Images and files are supported only in user messages")
        if kind == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                if set(image_url) - {"url", "detail"}:
                    raise ValueError("Invalid image_url part")
                url = image_url.get("url")
            else:
                url = image_url
            name = _clean_name(part.get("name"), "image")
            _, data = _decode_data_url(url, f"Image {name}")
            segments.append(load_image(data, name))
        elif kind == "file":
            file = part.get("file")
            if not isinstance(file, dict) or set(file) - {"filename", "file_data"}:
                raise ValueError("A file part needs file.filename and file.file_data; file_id is not supported")
            name = _clean_name(file.get("filename"), "file")
            mime, data = _decode_data_url(file.get("file_data"), f"File {name}")
            segments.append(f'<file name="{name}">\n{file_text(data, mime, name)}\n</file>')
        else:
            raise ValueError("Content parts must be text, image_url or file")
    return segments


def finalize(messages: list[dict[str, Any]]) -> None:
    """Number images, keep the most recent MAX_IMAGES, and write each message's text."""
    images = [segment for message in messages if isinstance(message.get("content"), list)
              for segment in message["content"] if isinstance(segment, ImageInput)]
    shown = set(map(id, images[-MAX_IMAGES:])) if MAX_IMAGES else set()
    for number, image in enumerate(images, 1):
        image.label = (f"[Image {number}: {image.name}]" if id(image) in shown
                       else f"[Image {number} omitted: {image.name}]")
    for message in messages:
        segments = message.get("content")
        if not isinstance(segments, list):
            continue
        texts = [segment.label if isinstance(segment, ImageInput) else segment for segment in segments]
        message["content"] = "\n\n".join(text for text in texts if text.strip())
        attached = [segment for segment in segments if isinstance(segment, ImageInput) and id(segment) in shown]
        if attached:
            message["images"] = attached


def message_images(messages: list[dict[str, Any]]) -> list[ImageInput]:
    return [image for message in messages for image in message.get("images", ())]


def without_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages for a text-only JSON view; image labels remain in the text."""
    return [{key: value for key, value in message.items() if key != "images"} for message in messages]


def redact(payload: Any) -> Any:
    """A copy of a request for logs, with base64 data: URLs replaced by their size."""
    def visit(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("data:") and len(value) > 256:
            head = value.split(",", 1)[0]
            return f"{head},<{len(value) - len(head) - 1} characters omitted>"
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value
    return visit(copy.deepcopy(payload) if isinstance(payload, (dict, list)) else payload)
