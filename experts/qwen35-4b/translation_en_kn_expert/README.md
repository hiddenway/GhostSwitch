# Kannada translation and writing (Qwen3.5-4B)

Translates text from English or other languages into Kannada and writes short Kannada texts; trained on English to Kannada translation. Requests in other languages rely on the base model's understanding; quality is best for English sources.

Source: https://huggingface.co/kshitijdesai99/Qwen-3.5-4B-finetuned_mt-nllb-en-kn

Revision: `cca74f70f981de6af55113f865372a174619fc95`. Check the source model card for license, training data, limitations, and usage terms before redistribution or use. This package contains metadata only; weights are fetched on demand. It was migrated from the archived Qwen3.5-4B registry in `versions/legacy-4b`. No independent quality claim is made.

## Examples

Use for: Translate 'Good morning, how are you?' into Kannada.

Not for: Translate Russian into German.

Also suitable: Write a short poem about an elephant in Kannada (requested in Russian).

Output language follows the user; the source training language does not override it.
