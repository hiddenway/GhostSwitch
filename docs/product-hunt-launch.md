# GhostSwitch — Product Hunt launch kit

Prepared on 2026-11-12. This is a local draft, not a submitted or scheduled launch.
Product claims below follow the repository README; installation and performance
have not been independently retested for this launch kit.

## Positioning

For developers and local-AI enthusiasts who want to try specialist adapters
without managing a separate full model for every task.

Lead with automatic expert selection on a local model. Keep the experimental
status visible. Avoid promising expert-level accuracy across the catalog or
describing a potential library of adapters as a trillion-parameter model.

## Submission fields

**Name**

GhostSwitch

**Product URL**

https://github.com/hiddenway/GhostSwitch

A public GitHub repository is an accepted product link. A separate website is
not required to submit.

**Tagline**

Local AI that picks the right expert for each task

**Description**

Run AI locally with one model and a library of specialist adapters. GhostSwitch automatically picks an expert for each message, with a browser chat and an OpenAI-compatible API. Free, open source, and experimental. Works offline after model downloads.

**Pricing:** Free.

**Open source:** Yes; link the repository. The application is Apache-2.0;
downloaded models and adapters retain their own licenses.

**Suggested launch tags:** Artificial Intelligence, Developer Tools, Open Source.
Choose the closest available tags in the live editor, up to three.

**Role:** I made this. Add the actual makers using their personal accounts.

## First maker comment

Hey Product Hunt! I'm the maker of GhostSwitch.

GhostSwitch explores a question: can one small local model become more useful
by switching specialist adapters for different tasks?

It keeps one Qwen3.5 model loaded, selects an expert for each message, and
attaches that expert's LoRA adapter. You get a browser chat and an
OpenAI-compatible API for your own tools. You can also choose an expert manually
or use the base model alone.

It's free and open source. After downloading the model and experts, you can
chat offline. Web search is optional. The 9B setup targets machines with 16 GB
of memory; a smaller 4B option is available for 8 GB machines.

This is an experiment, and I want the results to be inspectable. In the
evaluation documented in the repo, the coding expert scored 88.3% on 60
HumanEval+ tasks versus 76.7% for the base model. That is a small, specific test,
not a claim that every expert improves every answer. Most other measured
experts have not demonstrated an improvement, and adapters that fail the
project's quality gate are excluded from automatic routing.

The adapters come from their respective Hugging Face authors; GhostSwitch
packages and routes their work.

I'd especially like feedback from people already running local AI:
Which task would make automatic expert switching useful to you? If you try it,
please share your hardware, model, and any installation or routing problems.

Code, setup, measurements, and limitations:
https://github.com/hiddenway/GhostSwitch

## Gallery plan

Existing source images are in `.github/assets/`. These are source assets,
not a finished Product Hunt gallery. Review them for current UI and readable
text, then export gallery slides at the recommended 1270 × 760 px. Preserve
the screenshot proportions and use padding rather than stretching.

| Order | Message | Existing source |
| --- | --- | --- |
| Logo | Product identity | `logo-icon.png` (512 × 512; export 240 × 240) |
| 1 | Local AI. Automatic expert switching. | `screenshot-chat.png` |
| 2 | One base model. Different specialist adapters. | `how-it-works.png` |
| 3 | Choose the experts you want to run. | `screenshot-experts.png` |
| 4, optional | Start with a model that fits your machine. | `screenshot-first-run.png` |

Optional 45–60 second screen recording: show a real coding request, the
selected expert in the response details, the expert library, and the local API
address. Show the actual wait or clearly label any cut. Mention the hardware
and initial model download. Do not imply instant responses from an edited demo.

## Before scheduling

- [ ] Check personal-account posting access and complete the maker profile.
- [ ] Pick the release commit; the working tree contains ongoing runtime changes.
- [ ] Test the public installer from that release on a clean supported machine:
  download, first answer, stop, restart, and `ghostswitch doctor`.
- [ ] Check each advertised platform in CI or on hardware; qualify any untested
  platform rather than treating this document as verification.
- [ ] Verify the chat and expert details against the selected release. Local
  README measurements differ from the currently published GitHub README.
- [ ] Review the maker comment's benchmark values against the release evidence.
- [ ] Export the logo and at least three readable gallery slides; at least one
  gallery image is required by the current submission guide.
- [ ] Paste the text into the Product Hunt editor and save a draft.
- [ ] Review the resulting page and pick a day when a maker can answer comments.
- [ ] Schedule only after the product and launch page are ready. The guide allows
  dates up to 30 days ahead and says the launch starts at midnight Pacific Time.
  Check the displayed local time when scheduling.

## Launch-day post draft

GhostSwitch is on Product Hunt: local AI that automatically switches specialist
adapters on one base model. Free, open source, and experimental, with browser
chat and an OpenAI-compatible API.

If you run local AI, I'd love your feedback on the setup and expert routing.

[Insert the actual Product Hunt launch URL before sharing.]

Ask for feedback, not upvotes. Record installation issues, hardware, useful
tasks, and reproducible routing failures. Those are more useful follow-up
signals for this project than leaderboard position alone.

## Official submission references

Requirements checked on 2026-11-12:

- [How to post a product](https://help.producthunt.com/en/articles/479557-how-to-post-a-product):
  submit from a personal account; a separate hunter is not needed.
- [Build your product page](https://help.producthunt.com/en/articles/17834537-4-build-your-product-page):
  name up to 40 characters, tagline up to 60, description up to 500, maker
  comment up to 5,000; square logo recommended at 240 × 240 px, up to 2 MB;
  gallery recommended at 1270 × 760 px; one image required, three or more
  encouraged. Also covers draft saving, scheduling, and promotion rules.
