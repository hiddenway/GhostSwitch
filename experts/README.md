# Community expert packages

Each base model has its own catalog, discovered from `experts/<backbone>/*/` at process startup: `experts/qwen35-9b/` for Qwen3.5-9B and `experts/qwen35-4b/` for Qwen3.5-4B. The running server reads only the catalog of its current base model. Each contribution is a folder; changing Python routing code or a global adapter list is unnecessary. Users obtain contributions by updating the Git checkout, restarting GhostSwitch, and selecting the new expert in **Эксперты**.

## Verified experts and adapters from Hugging Face

The library has two kinds of experts, shown separately in the app:

- **GhostSwitch experts** are the packages in this folder. Each has a description, a prompt, a README and a place in the routing tree, so automatic selection can choose it. Its card says **Verified** when its `evaluation.json` shows it beats the base model, **Tested · no clear gain yet** when a test found no significant gain, and **Not tested yet** otherwise.
- **Adapters from Hugging Face** are any public LoRA trained for the base model (`base_model:adapter:Qwen/Qwen3.5-9B` and the other names in `accepted_base_names`). The app lists them under **Browse Hugging Face**; adding one writes a package to `~/.config/ghostswitch/experts/<backbone>/hub_<repo>/` (`GHOSTSWITCH_USER_EXPERTS` overrides the folder) pinned to the repository's current commit. They are marked **manual only**: the router never picks them, and the user chooses one for a conversation. An adapter is offered only when the repository holds `adapter_config.json` next to `adapter_model.safetensors`, is public and not gated, and its `base_model_name_or_path` is this base model (not a base, distilled or other variant). `GET /v1/experts/hub?q=` lists them; `POST /v1/experts/hub` with `{"action":"add","repo_id":"…"}` or `{"action":"remove","expert_id":"hub_…"}` adds or removes one from the same origin.

### Blocklist

An expert whose `evaluation.json` sets `"blocked": true` is hidden from the app and the router, and its repository is not offered from Hugging Face. `experts/<backbone>/blocklist.json` lists further repositories that must never be offered, each with a reason, such as adapters trained on benchmark questions. [docs/expert-blocklist.ru.md](../docs/expert-blocklist.ru.md) explains each blocked expert.

### Making an adapter a verified expert

Any adapter from Hugging Face can become a GhostSwitch expert:

1. Write a package in `experts/<backbone>/<expert_id>/` (structure below): a description of what it does and does not do, a task prompt, a README with its training data, license and limitations, and a routing parent with a router description.
2. Measure it against the base model on a public test of its field that it was not trained on: add a task to `scripts/expert_eval.py` and run `python scripts/expert_eval.py run --experts <expert_id>` on a GPU (see [docs/benchmarking.md](../docs/benchmarking.md)).
3. Run `python scripts/expert_eval.py report --write` and commit the resulting `evaluation.json`. Only a `"verdict": "helps"` result (better than the base model with p < 0.05) makes it **Verified**; a worse or equal result blocks it.

## Required structure

```text
experts/qwen35-9b/my_expert/
├── expert.json             # semantic metadata, domain, routing, initial default
├── lora.json               # pinned Hugging Face sources; contains no weights
├── prompt.txt              # specialist instructions, inherited by generation
├── README.md               # source/license, scope, examples, limitations
└── response_policy.json    # optional final_guidance / review_guidance
```

Copy the template and replace every placeholder:

```bash
cp -R experts/_template experts/qwen35-9b/my_expert
python scripts/validate_experts.py
```

The template is intentionally incomplete and is outside the discovered catalog. Change the folder and `expert.json.id` together. IDs use lowercase letters, digits and underscores, starting with a letter. `schema_version` is currently `1`. `capabilities` must be nonempty; `aliases` and `keywords` are string arrays. `backbone` names the folder's base model (`qwen35-9b` or `qwen35-4b`); compatible source lineages are defined in `core_models/backbones.json`. A LoRA trained against a different Qwen size or architecture is incompatible, even if its filename mentions Qwen.

Keep `default_enabled: false` for community contributions. First-start defaults are coding, reasoning and finance. Changing package defaults does not overwrite an existing user's selection.

## Hugging Face reference file

`lora.json` must name an HTTPS repository URL and an immutable, lowercase **40-character commit SHA**. Branch names such as `main` are rejected. `source.config` names a PEFT `adapter_config.json`; `source.weights` names a single `.safetensors` file. Paths are relative to the repository root, including any nested directory:

```json
{
  "schema_version": 1,
  "display_id": "my-adapter",
  "accepted_base_names": ["Qwen/Qwen3.5-9B"],
  "source": {
    "url": "https://huggingface.co/OWNER/REPOSITORY",
    "revision": "REPLACE_WITH_40_CHARACTER_COMMIT_SHA",
    "config": "adapter/adapter_config.json",
    "weights": "adapter/adapter_model.safetensors"
  },
  "gguf": {"filename": "my_expert.gguf"}
}
```

For a ready GGUF **LoRA adapter**, add `gguf.source` with `url`, `revision`, and `weights` (a repository-relative `.gguf` path). The source may be a different Hugging Face repository, with its own pinned revision. `gguf.filename` is a unique local cache basename. A full merged GGUF model cannot be attached as a LoRA. If no ready source is given, GhostSwitch converts the PEFT weights using the installed llama.cpp converter. Omit `gguf` entirely for an MLX-only entry; such an entry is hidden in a GGUF session.

Weights (`.safetensors`, `.gguf`, `.bin`, `.pt`, `.pth`) must never be committed inside a package. The validator rejects them and package symlinks; `.gitignore` also excludes weight files. Files in Hugging Face are downloaded as data. GhostSwitch does not execute code from contributed model repositories.

Current MLX conversion supports PEFT linear LoRA A/B pairs with a uniform rank and compatible target modules. DoRA, merged full checkpoints, pickle weights and arbitrary native MLX adapter layouts are not supported by this reference format. Local conversion and module checks reject incompatible adapters; listing a package does not certify its quality or compatibility with every runtime version.

## Routing and instructions

`routing.parent` must refer to a branch in `core/config/router_tree.json`. Existing branches: `coding`, `reasoning`, `finance`, `legal_contract`, `cybersecurity`, `engineering_design`, `data_tasks`, `science_publishing`, `roleplay`, and `root`. Choose a domain branch; `root` has little remaining capacity. `routing.node_id` must be unique (defaults to the expert ID). `routing.description` describes a task boundary, including jurisdiction, input-language specialization and output requirements. The composed tree permits at most 15 children per branch and six levels. If a branch is full, add an intermediate branch to the scaffold before contributing more leaves.

`prompt.txt` supplements the shared base prompt. Describe the expertise and format; preserve the shared response-language policy. Training in Spanish or Korean must not force every answer into that language. Source-language and jurisdiction limits still apply.

An optional `response_policy.json` can contain:

```json
{
  "final_guidance": "Specialist generation rules for this domain.",
  "review_guidance": "Specific checks the text verifier should perform.",
  "thinking": false
}
```

v6 appends `final_guidance` to the expert prompt and passes `review_guidance` to the text verifier. `"thinking": true` makes requests routed to this expert generate with the backbone's reasoning block; the reasoning is returned as `reasoning_content`, separately from the answer. Enable it only for adapters trained on reasoning traces (for example, `reasoning_expert`): it adds latency, and it does not repair an adapter that is weak in its domain. Archived v5 policy switches in the migrated finance package do not change v6 orchestration. Tool-call verification continues to use the shared agent contract.

## Contribution checklist

1. Add one folder using this structure and update all placeholders.
2. Pin each source commit and verify that its files and base architecture match.
3. Document the model card, license or uncertainty, scope, suitable/unsuitable examples, and known limitations in the package README.
4. Run `python scripts/validate_experts.py`. It checks metadata, source paths, backbone lineage, filename collisions and the composed routing tree without downloading a model.
5. Commit the folder and submit it through the repository's contribution workflow. After it is pulled, restart GhostSwitch. The user can then download and enable it from the library.

The metadata validator cannot check remote file availability, tensor contents or model quality. Those are checked during preparation, activation or a separately requested benchmark.
