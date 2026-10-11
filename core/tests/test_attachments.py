from __future__ import annotations

import base64
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from core import attachments, llm, main, verifier_v6, vision
from core.request_contract import parse_request
from core.tests.test_v6 import PROFILE, Registry


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def png(width: int = 64, height: int = 48, color=(200, 30, 30), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_part(name: str = "photo.png", **kwargs) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(png(**kwargs), "image/png")}, "name": name}


def file_part(name: str, data: bytes, mime: str) -> dict:
    return {"type": "file", "file": {"filename": name, "file_data": data_url(data, mime)}}


def pdf(lines: list[str]) -> bytes:
    """A one-page PDF with a Helvetica text layer."""
    content = "BT /F1 18 Tf 72 720 Td " + " ".join(f"({line}) Tj 0 -26 Td" for line in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
               f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()


def docx(paragraphs: list[str]) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{namespace}"><w:body>{body}</w:body></w:document>')
    return buffer.getvalue()


class ContentPartTests(unittest.TestCase):
    def test_image_file_and_text_parts_become_labelled_text_and_images(self):
        request = parse_request({"messages": [{"role": "user", "content": [
            image_part("cat.png"), file_part("notes.md", "# Plan\nShip it".encode(), "text/markdown"),
            {"type": "text", "text": "What is shown?"},
        ]}]})
        message = request["messages"][0]
        self.assertEqual(message["content"],
                         '[Image 1: cat.png]\n\n<file name="notes.md">\n# Plan\nShip it\n</file>\n\nWhat is shown?')
        self.assertEqual([image.label for image in message["images"]], ["[Image 1: cat.png]"])
        self.assertEqual(message["images"][0].image.mode, "RGB")

    def test_an_image_alone_is_a_valid_user_message(self):
        request = parse_request({"messages": [{"role": "user", "content": [image_part()]}]})
        self.assertEqual(request["messages"][0]["content"], "[Image 1: photo.png]")

    def test_only_the_most_recent_images_reach_the_model(self):
        messages = [{"role": "user", "content": [image_part(f"{index}.png")]} for index in range(3)]
        for index in range(2):
            messages.insert(2 * index + 1, {"role": "assistant", "content": "Seen"})
        with patch.object(attachments, "MAX_IMAGES", 2):
            request = parse_request({"messages": messages})
        self.assertEqual(request["messages"][0]["content"], "[Image 1 omitted: 0.png]")
        self.assertNotIn("images", request["messages"][0])
        self.assertEqual([image.name for image in attachments.message_images(request["messages"])],
                         ["1.png", "2.png"])

    def test_text_parts_are_accepted_for_every_role(self):
        request = parse_request({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]})
        self.assertEqual([message["content"] for message in request["messages"]], ["Be brief.", "Hi"])

    def test_images_outside_user_messages_and_remote_urls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "only in user messages"):
            parse_request({"messages": [{"role": "user", "content": "Hi"},
                                        {"role": "assistant", "content": [image_part()]},
                                        {"role": "user", "content": "And?"}]})
        with self.assertRaisesRegex(ValueError, "does not download"):
            parse_request({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]})
        with self.assertRaisesRegex(ValueError, "Could not read image"):
            parse_request({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(b"not an image", "image/png")}}]}]})

    def test_transparent_images_are_flattened_on_white_and_large_ones_scaled(self):
        image = attachments.load_image(png(10, 10, (0, 0, 0, 0), "RGBA"), "clear.png")
        self.assertEqual(image.image.getpixel((5, 5)), (255, 255, 255))
        with patch.object(attachments, "MAX_IMAGE_PIXELS", 100 * 100):
            image = attachments.load_image(png(400, 200), "wide.png")
        self.assertLessEqual(image.size[0] * image.size[1], 100 * 100)
        self.assertAlmostEqual(image.size[0] / image.size[1], 2, delta=0.05)

    def test_pdf_and_docx_text_is_extracted(self):
        text = attachments.file_text(pdf(["Total due: 4,250 EUR"]), "application/pdf", "invoice.pdf")
        self.assertIn("[Page 1]", text)
        self.assertIn("Total due: 4,250 EUR", text)
        self.assertEqual(attachments.file_text(docx(["Первый", "Второй"]), "", "contract.docx"), "Первый\nВторой")

    def test_unreadable_files_explain_what_to_send(self):
        scan = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(scan, format="PDF")
        with self.assertRaisesRegex(ValueError, "no text layer"):
            attachments.file_text(scan.getvalue(), "application/pdf", "scan.pdf")
        with self.assertRaisesRegex(ValueError, "send text, code, PDF or DOCX"):
            attachments.file_text(bytes(range(256)), "application/octet-stream", "blob.bin")

    def test_long_files_are_cut_with_a_visible_note(self):
        with patch.object(attachments, "MAX_FILE_CHARS", 10):
            text = attachments.file_text(b"x" * 25, "text/plain", "long.txt")
        self.assertTrue(text.startswith("x" * 10))
        self.assertIn("first 10 of 25 characters", text)

    def test_redaction_keeps_small_values_and_replaces_data(self):
        url = data_url(bytes(range(256)) * 4, "image/png")
        payload = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
                   "note": "data:short"}
        redacted = attachments.redact(payload)
        self.assertEqual(redacted["messages"][0]["content"][0]["image_url"]["url"],
                         f"data:image/png;base64,<{len(url) - len('data:image/png;base64,')} characters omitted>")
        self.assertEqual(redacted["note"], "data:short")
        self.assertEqual(payload["messages"][0]["content"][0]["image_url"]["url"], url)


class GenerationViewTests(unittest.TestCase):
    def test_each_image_slot_follows_its_label(self):
        first, second = (attachments.load_image(png(), name) for name in ("a.png", "b.png"))
        first.label, second.label = "[Image 1: a.png]", "[Image 2: b.png]"
        rendered = llm._template_messages([{"role": "user", "content": "[Image 1: a.png]\n\nCompare with [Image 2: b.png]",
                                            "images": [first, second]}])
        self.assertEqual(rendered, [{"role": "user", "content": [
            {"type": "text", "text": "[Image 1: a.png]"}, {"type": "image"},
            {"type": "text", "text": "\n\nCompare with [Image 2: b.png]"}, {"type": "image"},
        ]}])

    def test_an_image_without_its_label_goes_first(self):
        image = attachments.load_image(png(), "a.png")
        image.label = "[Image 1: a.png]"
        rendered = llm._template_messages([{"role": "user", "content": "Describe", "images": [image]}])
        self.assertEqual(rendered[0]["content"], [{"type": "image"}, {"type": "text", "text": "Describe"}])

    def test_verifier_sees_text_json_and_the_images(self):
        request = parse_request({"messages": [{"role": "user", "content": [image_part("cat.png"), {"type": "text", "text": "What animal?"}]}]})
        with patch.object(verifier_v6.llm, "chat", return_value='{"status":"ACCEPT","feedback":null}') as chat:
            result = verifier_v6.verify(request["messages"], "A cat.", finish_reason="stop", review=True)
        self.assertEqual(result["status"], "ACCEPT")
        system, review = chat.call_args.args[0]
        self.assertIn("The user attached images", system["content"])
        self.assertTrue(review["content"].startswith("[Image 1: cat.png]\n\n{"))
        body = json.loads(review["content"].split("\n\n", 1)[1])
        self.assertEqual(body["messages"], [{"role": "user", "content": "[Image 1: cat.png]\n\nWhat animal?"}])
        self.assertEqual(review["images"], request["messages"][0]["images"])

    def test_run_request_prepares_vision_and_keeps_images_for_generation(self):
        payload = {"messages": [{"role": "user", "content": [image_part(), {"type": "text", "text": "Describe"}]}],
                   "mode": "base"}
        with patch.object(main, "_profile", return_value=PROFILE), \
             patch.object(main.llm, "vision_available", return_value=True), \
             patch.object(main.llm, "prepare_vision") as prepare, \
             patch.object(main.llm, "chat", return_value="A red square.") as chat, \
             patch.object(main.verifier_v6, "verify", return_value={"status": "ACCEPT", "feedback": None}):
            result = main.run_request(payload, registry=Registry())
        self.assertEqual(result["status"], "success")
        prepare.assert_called_once()
        vision_stage = next(stage for stage in result["ghostswitch"]["stages"] if stage["phase"] == "vision")
        self.assertEqual(vision_stage["images"], 1)
        self.assertEqual(len(chat.call_args.args[0][-1]["images"]), 1)

    def test_gguf_projector_is_downloaded_once_when_missing(self):
        spec = llm.ModelSpec("/models/Qwen3.5-9B-Q4_K_M.gguf", 9, backend="gguf")
        with patch.dict(llm.MODEL_SPECS, {"thinker": spec}), \
             patch.object(llm, "find_mmproj", side_effect=[None, "/cache/mmproj.gguf"]), \
             patch.object(llm, "mmproj_source", return_value=("org/repo", "mmproj.gguf", "abc")), \
             patch("core.download_progress.download_file") as download:
            llm.prepare_vision()
            llm.prepare_vision()
        download.assert_called_once_with("org/repo", "mmproj.gguf", revision="abc", progress=None)

    def test_images_are_rejected_when_the_model_has_no_vision(self):
        payload = {"messages": [{"role": "user", "content": [image_part()]}], "mode": "base"}
        with patch.object(main.llm, "vision_available", return_value=False), \
             self.assertRaisesRegex(ValueError, "does not accept images"):
            main.run_request(payload, registry=Registry())


SETTINGS = vision.VisionSettings(
    hidden_size=32, num_heads=2, depth=1, intermediate_size=64, out_hidden_size=48, patch_size=16,
    temporal_patch_size=2, merge_size=2, in_channels=3, num_position_embeddings=16, image_token_id=9,
    image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), min_pixels=32 * 32, max_pixels=4096 * 4096,
    mrope_section=(11, 11, 10),
)


class VisionTests(unittest.TestCase):
    def test_smart_resize_rounds_to_the_patch_grid_within_limits(self):
        self.assertEqual(vision.smart_resize(448, 640, 32, 65536, 1 << 24), (448, 640))
        height, width = vision.smart_resize(3000, 4000, 32, 65536, 1024 * 1024)
        self.assertTrue(height % 32 == 0 and width % 32 == 0 and height * width <= 1024 * 1024)
        self.assertEqual(vision.smart_resize(20, 20, 32, 65536, 1 << 24), (256, 256))

    def test_preprocess_orders_patches_by_merge_block(self):
        pixels = np.zeros((64, 96, 3), dtype=np.uint8)
        pixels[:, 32:64] = 255  # a white middle block column
        patches, grid = vision.preprocess(Image.fromarray(pixels), SETTINGS, 1 << 24)
        self.assertEqual(grid, (1, 4, 6))
        self.assertEqual(patches.shape, (24, 2 * 16 * 16 * 3))
        # Block (0, 1) holds patches 4..7: rows 0-1, columns 2-3, all white.
        np.testing.assert_allclose(patches[4:8], 1.0)
        np.testing.assert_allclose(patches[:4], -1.0)

    def test_multimodal_positions_match_qwen_rope_index(self):
        tokens = [1, 2, 9, 9, 9, 9, 9, 9, 3, 4]
        positions, delta = vision.mrope_positions(tokens, 9, [(1, 2, 3)])
        np.testing.assert_array_equal(positions, [
            [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
            [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
            [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
        ])
        self.assertEqual(delta, 7 - len(tokens))
        with self.assertRaisesRegex(ValueError, "does not match"):
            vision.mrope_positions([9, 9, 1], 9, [(1, 2, 2)])

    @unittest.skipIf(vision.mx is None, "MLX is not installed")
    def test_multimodal_rope_equals_ordinary_rope_for_text(self):
        import mlx.core as mx
        import mlx.nn as nn
        base = nn.RoPE(64, traditional=False, base=1e7)
        x = mx.random.normal((1, 2, 5, 256))
        rope = vision.MultimodalRoPE(base, np.tile(np.arange(12), (3, 1)), 0, (11, 11, 10))
        for offset in (0, 4, 10, 20):
            np.testing.assert_allclose(np.array(rope(x, offset=offset)), np.array(base(x, offset=offset)),
                                       atol=1e-5)

    @unittest.skipIf(vision.mx is None, "MLX is not installed")
    def test_vision_tower_output_has_one_row_per_merged_patch(self):
        tower = vision.VisionTower(SETTINGS)
        patches, grid = vision.preprocess(Image.new("RGB", (96, 64)), SETTINGS, 1 << 24)
        import mlx.core as mx
        features = tower(mx.array(patches), grid)
        self.assertEqual(features.shape, (24 // 4, 48))


if __name__ == "__main__":
    unittest.main()
