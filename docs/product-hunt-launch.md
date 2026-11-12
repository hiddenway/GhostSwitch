# GhostSwitch — Product Hunt launch kit

Updated on 2026-11-12. English copy for the existing Product Hunt draft.
The launch is not scheduled. [Review the saved draft](https://www.producthunt.com/products/ghostswitch?launch=ghostswitch).
Product claims below follow the repository README; installation and performance
have not been independently retested for this launch kit.

## Positioning

**Big AI power. Small hardware.**

Specialist AI on your everyday laptop. For curious people, makers and
developers who want to run AI locally without buying a high-end GPU.

Lead with accessible hardware and a growing range of skills. Explain the
mechanism in one sentence: one compact model switches small experts for
different tasks. Keep the experimental status visible. The slogan describes
the product's ambition and range of uses, not performance parity with larger
models. Do not promise that every expert improves quality.

## Submission fields

**Name**

GhostSwitch

**Product URL**

https://github.com/hiddenway/GhostSwitch

A public GitHub repository is an accepted product link. A separate website is
not required to submit.

**Tagline**

Big AI power. Small hardware.

**Description**

Specialist AI on your everyday laptop. GhostSwitch uses one compact model and automatically switches experts for coding, math, design and more. No high-end GPU required. Choose 4B for 8 GB RAM or 9B for 16 GB. Runs locally, with browser chat and an OpenAI-compatible API. Free, open source, and experimental. Expert quality varies. Works offline after model downloads.

**Pricing:** Free.

**Open source:** Yes; link the repository. The application is Apache-2.0;
downloaded models and adapters retain their own licenses.

**Suggested launch tags:** Artificial Intelligence, Developer Tools, Open Source.
Choose the closest available tags in the live editor, up to three.

**Role:** I made this. Add the actual makers using their personal accounts.

## First maker comment

Hey Product Hunt! I'm Danylo, the maker of GhostSwitch.

Big AI power. Small hardware.

You shouldn't need a high-end GPU to explore specialist AI on your own computer.
GhostSwitch brings a growing library of skills to an everyday laptop.

The idea is simple: keep one compact model loaded and switch small expert
add-ons for different tasks. Ask a question, and GhostSwitch picks an enabled
expert or uses the base model when none fits. You can choose an expert yourself,
too.

What you get:

- Local AI with a browser chat.
- A 4B option for 8 GB RAM, or 9B for 16 GB and up.
- Automatic expert selection, without loading a full model for every task.
- An OpenAI-compatible API for your tools.
- Free, open-source software. Offline chat after the initial downloads.

No high-end GPU is required, but hardware affects speed: CPU-only answers are
slower. Web search is optional.

GhostSwitch is experimental. More experts mean more tasks to explore, not a
guarantee of better answers. Only the coding expert has shown a measured
improvement over the base model in the project's evaluation so far. The results
and limitations are in the repo. The expert adapters are the work of their
respective Hugging Face authors; GhostSwitch packages and routes them.

What would you want AI to help with on the laptop you already own? If you try
GhostSwitch, I'd love to hear your hardware, your task, and what worked or broke.

Code, setup, measurements, and limitations:
https://github.com/hiddenway/GhostSwitch

## Gallery plan

Assets are in `.github/assets/`. The hero and overview follow the original
GitHub artwork: dark backgrounds, flat diagrams and restrained violet/teal
accents. The screenshots show the actual application; the hero shows one base
model connected to an active specialist. The recommended gallery
size is 1270 × 760 px; keep source proportions if exporting another size.

| Order | Message | Existing source |
| --- | --- | --- |
| Logo | Product identity | `logo-icon.png` (512 × 512; export 240 × 240) |
| 1 | Big AI power. Small hardware. | `product-hunt-cover.png` |
| 2 | See GhostSwitch in action. | `screenshot-chat.png` |
| 3 | More skills. One small model. | `how-it-works.png` |
| 4 | Choose the experts you want to run. | `screenshot-experts.png` |

`banner.png` is the wide README version. `product-hunt-cover.png` and
`social-preview.png` use a taller composition so the copy fits the launch gallery.
The 8–16 GB hero badge refers to the 4B/9B hardware options, not a guaranteed
memory ceiling for every context length or backend.

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

Big AI power. Small hardware.

GhostSwitch brings specialist AI to your everyday laptop. One compact model,
automatic expert switching, and no high-end GPU required. Runs locally. Free,
open source, and experimental.

What would you try on the laptop you already own? I'd love your feedback.

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
