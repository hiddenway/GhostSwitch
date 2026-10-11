# Receipt and invoice extraction (Qwen3.5-4B)

Extracts structured fields such as merchant, address, date, line items, tax, and total from receipt or invoice text transcribed by OCR.

Source: https://huggingface.co/singhabhishekkk/apprentice-qwen35-4b-lora-receipts

Revision: `404cf0e062504457967c7906f5fe345c7b522ebf`. Check the source model card for license, training data, limitations, and usage terms before redistribution or use. This package contains metadata only; weights are fetched on demand. It was migrated from the archived Qwen3.5-4B registry in `versions/legacy-4b`. No independent quality claim is made.

## Examples

Use for: Extract merchant, date, and total from: NORTH PIER / 2026-04-17 / TOTAL USD 28.40

Not for: Write a poem about shopping.

Output language follows the user; the source training language does not override it.
