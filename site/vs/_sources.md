# Comparison sources (not served; see site/.assetsignore)

Every competitor claim on /vs/ and the guide pages comes from this file. All pages
were read on **2026-10-10** unless a line says otherwise. Anything not found on an
official page is marked "not stated" and left off the site. Re-check before editing
a claim; products change fast.

voiceio facts come from the repo itself: README.md, CONTRIBUTING.md, pyproject.toml
(version 1.5.0), config.example.toml.

---

## Voxtype

Read from a clone of github.com/peteonrails/voxtype at commit 6d2fc6d (2026-09-26),
version 1.1.0 in Cargo.toml, plus https://voxtype.io and the releases page.

- Platforms: Linux and macOS. macOS support "is in beta", binaries unsigned, Homebrew cask, macOS 13+. [README](https://github.com/peteonrails/voxtype), [docs/INSTALL_MACOS.md](https://github.com/peteonrails/voxtype/blob/main/docs/INSTALL_MACOS.md), [voxtype.io](https://voxtype.io)
- License: MIT (Copyright 2025 Peter Jackson). Free. "No cloud, no subscription, no telemetry." [LICENSE](https://github.com/peteonrails/voxtype/blob/main/LICENSE), [voxtype.io](https://voxtype.io)
- Latest release: v1.1.0 (late Sept 2026; GitHub shows Sept 24, voxtype.io says Sept 23). [releases](https://github.com/peteonrails/voxtype/releases)
- Written in Rust. Packages: AUR, .deb, .rpm, Homebrew; signed release binaries. [README](https://github.com/peteonrails/voxtype)
- Engines: Whisper (whisper.cpp, default), Parakeet, Moonshine, SenseVoice, Paraformer, Dolphin, Omnilingual, Cohere Transcribe, OpenVINO Whisper. README calls Cohere Transcribe "9-11× realtime" on a Zen 4 CPU. GPU: Vulkan, CUDA 12/13, MIGraphX (AMD), OpenVINO/Intel NPU. [README](https://github.com/peteonrails/voxtype)
- Local by default; optional remote Whisper servers; optional Soniox cloud streaming backend (paid, audio leaves the machine). [README](https://github.com/peteonrails/voxtype), [docs/SONIOX.md](https://github.com/peteonrails/voxtype/blob/main/docs/SONIOX.md)
- Activation: push-to-talk (hold) by default, toggle mode available; compositor keybindings or evdev. [README](https://github.com/peteonrails/voxtype)
- Output: wtype, then dotool, ydotool, clipboard; "paste" mode. [README](https://github.com/peteonrails/voxtype)
- Live typing: optional `streaming = true` sliding-window engine, "text lands at the cursor incrementally as you speak"; requires toggle activation. Default gate types a word once two re-transcriptions agree; experimental `revision_mode` types immediately and corrects with backspace. voxtype.io: "Parakeet streaming types text at the cursor as you speak". [docs/CONFIGURATION.md#streaming](https://github.com/peteonrails/voxtype/blob/main/docs/CONFIGURATION.md), [voxtype.io](https://voxtype.io)
- Eager processing (experimental): chunks transcribed while recording. [docs/CONFIGURATION.md](https://github.com/peteonrails/voxtype/blob/main/docs/CONFIGURATION.md)
- Vocabulary: Whisper `initial_prompt` for domain terms; word replacement tables; spoken punctuation; post-processing command (any LLM or script). [docs/CONFIGURATION.md#initial_prompt](https://github.com/peteonrails/voxtype/blob/main/docs/CONFIGURATION.md), [README](https://github.com/peteonrails/voxtype)
- Learns from user / fine-tuning: not stated (no mention in README, docs or voxtype.io).
- Other: meeting mode (chunked continuous transcription, speaker attribution, export to Markdown/JSON/SRT/VTT); MPRIS media pause; waveform OSD; Waybar integration; TUI `voxtype configure`; many languages via multilingual engines. [README](https://github.com/peteonrails/voxtype)

## Aqua Voice

- Platforms: Mac (Apple Silicon and Intel), Windows 10/11, iOS 17+, Android 10+. [pricing](https://aquavoice.com/pricing), [llms.txt](https://aquavoice.com/llms.txt)
- Linux: "Does Aqua Voice run on Linux? No. Aqua Voice runs on Mac, Windows, iPhone, and Android." A separate Aqua CLI installs on Linux but only manages dictionary/settings and transcribes files (needs API key); not live dictation. [vs/openwhispr](https://aquavoice.com/vs/openwhispr), [docs/cli](https://aquavoice.com/docs/cli)
- Pricing (pricing page, 2026-10-10): Free: 1,000 free words; Pro $8/mo billed annually or $10 monthly; Max $24/mo annually or $30 monthly (adds Realtime Mode); Business $19/seat/mo annually or $24 monthly, min 2 seats. Students 70% off Pro and Max. [pricing](https://aquavoice.com/pricing)
- Processing: cloud. "Audio is processed in the cloud and is never sold..." FAQ: "Cloud dictation needs an internet connection. Offline availability varies by platform, app version, and mode." No offline mode documented. [llms.txt](https://aquavoice.com/llms.txt), [FAQ](https://aquavoice.com/info/faq)
- Models: own proprietary ASR "Avalon" (Aug 2025, replaced an earlier Whisper-based pipeline; Avalon 1.5 April 2026) plus language-model post-processing; Realtime mode streams audio to an unnamed third-party realtime provider. [llms.txt](https://aquavoice.com/llms.txt)
- Open source: no (commercial, Aqua Voice, Inc.). [vs/openwhispr](https://aquavoice.com/vs/openwhispr)
- Live typing: default Instant Mode inserts text after you release; Realtime Mode (Max plan, desktop) shows live words in the Aqua toolbar, and "your message only lands in the app once you finish". [llms.txt](https://aquavoice.com/llms.txt), [realtime](https://aquavoice.com/realtime)
- Vocabulary: Dictionary, Replacements, Custom Instructions (Pro+), synced across devices. [guide/dictionary](https://aquavoice.com/guide/dictionary), [llms.txt](https://aquavoice.com/llms.txt)
- Learns from user automatically: not stated. Deep Context reads on-screen content (optional, off by default). [llms.txt](https://aquavoice.com/llms.txt)
- Privacy: Privacy Mode off by default ("transcripts may be stored on our servers to improve the product"); on = not stored. SOC 2 Type II, ISO 27001; HIPAA (contact for BAA). [llms.txt](https://aquavoice.com/llms.txt), [privacy](https://aquavoice.com/info/privacy), [pricing](https://aquavoice.com/pricing)
- Strengths: own ASR model with published benchmarks, Deep Context, Edit Mode (rewrite selected text by voice), 49 languages, cross-device sync, API/MCP/CLI. [llms.txt](https://aquavoice.com/llms.txt)

## Wispr Flow

- Platforms: Mac (macOS 12+), Windows 10/11 x64, iOS 18.3+ (iPhone), Android 13+. [pricing](https://wisprflow.ai/pricing), [system requirements](https://docs.wisprflow.ai/articles/1036674442-supported-devices-and-system-requirements)
- Linux: "iPad, Linux, Chromebooks, virtual machines, and remote desktop environments are not supported." "There is no native Linux application." [requirements](https://docs.wisprflow.ai/articles/1036674442-supported-devices-and-system-requirements), [Linux/WSL article](https://docs.wisprflow.ai/articles/6478598909-using-flow-with-linux-wsl-and-terminal-applications)
- Pricing (pricing page, 2026-10-10): Free: 2,000 words/week desktop, 1,000/week mobile; Pro $15/user/mo monthly or $12 annual; Growth $23 monthly or $18 annual (page also shows higher prices with unlimited Notetaker; ambiguous, not used on site); Enterprise custom. [pricing](https://wisprflow.ai/pricing)
- Processing: "Transcription always occurs on the cloud." "An internet connection is required for transcription." [data-controls](https://wisprflow.ai/data-controls), [what is Flow](https://docs.wisprflow.ai/articles/2772472373-what-is-flow)
- Models: own speech model "Canto" since 2026-09-17; plus unnamed third-party AI providers under zero-data-retention agreements. [whats-new](https://wisprflow.ai/whats-new), [data-controls](https://wisprflow.ai/data-controls)
- Open source: no (not stated anywhere; proprietary).
- Live typing: "Desktop dictation inserts the completed text after you stop, rather than displaying a live word-by-word transcript as you speak." [what is Flow](https://docs.wisprflow.ai/articles/2772472373-what-is-flow)
- Vocabulary: personal Dictionary, team dictionaries (Pro+), Snippets, synced. [dictionary](https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary)
- Learns from user: yes, auto-adds corrected words to your dictionary ("Auto-add to Dictionary", can be turned off); per-app styles. Personalized speech models are stated as a goal, not shipped. [dictionary](https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary), [data-controls](https://wisprflow.ai/data-controls), [whats-new](https://wisprflow.ai/whats-new)
- Privacy: "Improve the model for everyone" setting controls training use; locked off on Growth/Enterprise. SOC 2 Type II, ISO 27001; HIPAA BAA available. [data-controls](https://wisprflow.ai/data-controls), [privacy](https://wisprflow.ai/privacy)
- Strengths: AI auto-edits (filler removal, backtrack), Command Mode (beta), 100+ languages, Notetaker, cross-device sync, generous free tier, enterprise admin. [pricing](https://wisprflow.ai/pricing), [command mode](https://docs.wisprflow.ai/articles/4816967992-how-to-use-command-mode)

## Superwhisper

- Platforms: macOS 14+, Windows 10/11, iOS, Android. Docs list Linux as "Coming soon". [introduction](https://superwhisper.com/docs/get-started/introduction.md), [quickstart](https://superwhisper.com/docs/get-started/quickstart.md)
- Linux: public `superwhisper-linux-releases` repo; README: "The current preview targets x86-64 systems running Omarchy" and "Superwhisper for Linux is prerelease software". v1.0.0 published 2026-10-09. [repo](https://github.com/superultrainc/superwhisper-linux-releases)
- Pricing (docs Plans page; superwhisper.com/pricing returns 404): Free $0 (local Whisper models, unlimited dictation, custom vocabulary); Pro $8.49/month, $84.99/year or $249.99 lifetime; Enterprise custom. [plans](https://superwhisper.com/docs/billing/plans.md), [homepage](https://superwhisper.com)
- Processing: local or cloud, chosen per mode. Fully on-device setups keep audio and text on the machine. [voice models](https://superwhisper.com/docs/models/voice.md), [sensitive data](https://superwhisper.com/docs/security/sensitive-data.md)
- Models: local Whisper (whisper.cpp) tiny to large-v3-turbo (free); local Parakeet and Cohere Transcribe (Pro); cloud S1-Voice, Deepgram Nova, ElevenLabs Scribe (Pro); language models for formatting (local S1-mini, cloud GPT/Claude/Gemini, BYO keys). [voice models](https://superwhisper.com/docs/models/voice.md), [language models](https://superwhisper.com/docs/models/language.md)
- Open source: no ("Application source and model weights are maintained separately"; repo hosts binaries only). [repo](https://github.com/superultrainc/superwhisper-linux-releases)
- Live typing: "Realtime" (Pro) shows words in the recording window as you speak, only with streaming-capable models (Nova, Scribe, Parakeet), not Whisper; result pasted at the cursor after dictation. [realtime](https://superwhisper.com/docs/common-issues/realtime.md), [advanced settings](https://superwhisper.com/docs/get-started/settings-advanced.md)
- Vocabulary: vocabulary words as recognition hints ("use vocabulary sparingly") and replacements, Free and Pro. [vocabulary](https://superwhisper.com/docs/get-started/interface-vocabulary.md)
- Learns from user automatically: not stated. Personalization via vocabulary, replacements, Modes, context awareness (Pro). [modes](https://superwhisper.com/docs/modes/modes.md)
- Strengths: many local and cloud models, AI formatting Modes, 100+ languages, meeting recording, file transcription, lifetime license, one license across devices. [homepage](https://superwhisper.com)

## Dragon (Nuance / Microsoft)

- Lineup: Dragon Professional v16, Dragon Legal v16, Dragon Law Enforcement v16 (locally installed, Windows); Dragon Professional Anywhere and Legal Anywhere (cloud, Windows client); Dragon Anywhere Mobile (iOS/Android, US & Canada); Dragon Medical One (cloud, via Microsoft). [dragon.nuance.com](https://dragon.nuance.com/en-us/dragon.html), [Medical One](https://www.microsoft.com/en-us/health-solutions/clinical-workflow/dragon-medical-one)
- Platforms: Windows 10/11 (and Windows Server) for desktop; iOS/Android mobile. No Linux client listed. [Professional v16 data sheet](https://dragon.nuance.com/shared/data-sheets/ds-dragon-professional-v16-en-us.pdf), [Anywhere Mobile](https://dragon.nuance.com/en-us/dragon-anywhere-mobile)
- Mac: "Dragon Professional Individual for Mac is being discontinued effective 10/22/2018". [Nuance support](https://nuance.custhelp.com/app/answers/detail/a_id/27843)
- Price: no public price on official pages ("Contact us"; Anywhere: "subscription-based pricing"). [Dragon Professional](https://dragon.nuance.com/en-us/dragon-professional), [Professional Anywhere](https://dragon.nuance.com/en-us/dragon-professional-anywhere)
- Processing: v16 desktop products locally installed (internet for download/activation); Anywhere products cloud-hosted on Azure; Mobile needs a connection. [data sheet](https://dragon.nuance.com/shared/data-sheets/ds-dragon-professional-v16-en-us.pdf), [Professional Anywhere](https://dragon.nuance.com/en-us/dragon-professional-anywhere)
- Models: proprietary "Nuance Deep Learning" engine. Open source: no.
- Live typing: "Just speak and the words appear on the computer screen, instantly." [Dragon Professional](https://dragon.nuance.com/en-us/dragon-professional)
- Vocabulary: custom words, import/export, shared word lists; Legal edition vocabulary. [data sheet](https://dragon.nuance.com/shared/data-sheets/ds-dragon-professional-v16-en-us.pdf), [comparison matrix](https://dragon.nuance.com/shared/data-sheets/fm-dragon-professional-v16-and-dragon-professional-anywhere-comparison-en-us.pdf)
- Learns from user: "Dragon learns words and phrases used most to minimize corrections and adapts to the user's voice while dictating." [comparison matrix](https://dragon.nuance.com/shared/data-sheets/fm-dragon-professional-v16-and-dragon-professional-anywhere-comparison-en-us.pdf)
- Hands-free / accessibility: "Full hands-free Dragon operation to open applications, navigate menus and utilize mouse-grid", macros, scripting; "Section 508 certified". [comparison matrix](https://dragon.nuance.com/shared/data-sheets/fm-dragon-professional-v16-and-dragon-professional-anywhere-comparison-en-us.pdf), [accessibility](https://dragon.nuance.com/en-us/accessibility-solutions-for-business)
- Not used: old v15 prices from search snippets (official store pages now 404).

## nerd-dictation

- "Offline Speech to Text for Desktop Linux", uses VOSK-API. Desktop Linux only. [README](https://github.com/ideasman42/nerd-dictation)
- License: GPL-3.0. [license.rst](https://github.com/ideasman42/nerd-dictation/blob/main/license.rst)
- Output: xdotool (default, X11), dotool/dotoolc, ydotool, wtype (Wayland), or stdout. [README](https://github.com/ideasman42/nerd-dictation)
- Live typing: types while you speak by default; `--defer-output` disables. [README](https://github.com/ideasman42/nerd-dictation)
- Customization: `~/.config/nerd-dictation/nerd-dictation.py` Python hook for text manipulation; `--vosk-grammar-file` restricts recognized phrases. Numbers to digits. [README](https://github.com/ideasman42/nerd-dictation)
- Learns from user: not stated.
- Stated limitations: "Text from VOSK is all lower-case."; capitalizing proper nouns listed under further work. [README](https://github.com/ideasman42/nerd-dictation)
- "single file Python script with minimal dependencies"; no background processes. Last commit 2025-10-10; no tagged releases. [README](https://github.com/ideasman42/nerd-dictation), [commits](https://github.com/ideasman42/nerd-dictation/commits/main)

## Talon Voice

- Platforms: macOS, Windows, "Linux (X11)". [talonvoice.com](https://talonvoice.com/)
- Wayland: "Talon, like many tools for automation or accessibility, does not support Wayland." [talon.wiki install guide](https://talon.wiki/Resource%20Hub/Talon%20Installation/installation_guide/)
- Talon 1.0 (2026-09-20) changelog: "Talon 1.0 (plus any minor fixes) is planned to be the last public release of Talon for Linux/X11"; "Ongoing Linux/X11 support" listed as a Talon+ feature. [changelog](https://talonvoice.com/dl/latest/changelog.html)
- Price: "Talon is free to use and ongoing development is made possible by donations on Patreon." Talon+ (formerly "beta") via Patreon; tier price not stated on pages read. [talon.wiki](https://talon.wiki/), [changelog](https://talonvoice.com/dl/latest/changelog.html)
- License: "Talon is closed source." Community command set (talonhub/community) is MIT. [FAQ](https://talon.wiki/Help/FAQ), [community](https://github.com/talonhub/community)
- Engines: built-in Conformer (wav2letter) ships with Talon; Dragon as alternative; Webspeech (needs internet) and Vosk listed as dictation-only. Talon+ adds a Conformer + Whisper hybrid engine. [speech engines](https://talon.wiki/Resource%20Hub/Speech%20Recognition/speech%20engines/), [changelog](https://talonvoice.com/dl/latest/changelog.html)
- Local vs cloud for Conformer: not stated explicitly on official pages read (not claimed on site).
- Control: hands-free commands ("Write. Code. Browse. Game"), eye tracking ("Mouse by just looking"), noise input ("Hiss to scroll"), scripting in .talon/Python. "Talon does not come with voice commands out of the box"; users install the community set. [talonvoice.com](https://talonvoice.com/), [install guide](https://talon.wiki/Resource%20Hub/Talon%20Installation/installation_guide/)
- Vocabulary: community set supports adding words and replacements. [community](https://github.com/talonhub/community)

## Whisper, whisper.cpp, faster-whisper

- OpenAI Whisper: speech recognition model plus Python package/CLI; transcribes audio, does not type into apps. "Whisper's code and model weights are released under the MIT License." Sizes tiny/base/small/medium/large/turbo (large-v3, large-v3-turbo). `initial_prompt` supported. [openai/whisper](https://github.com/openai/whisper)
- whisper.cpp: "High-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model"; MIT; many platforms and GPU backends; `whisper-stream` real-time example (samples every half second); `whisper-server` HTTP server; `--prompt`. Latest release v1.9.5 (2026-10-06). [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), [server](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server)
- faster-whisper: "a reimplementation of OpenAI's Whisper model using CTranslate2"; "up to 4 times faster than openai/whisper for the same accuracy while using less memory"; MIT; `hotwords` and `initial_prompt` in `transcribe()`; hotword tokens truncated to `max_length // 2 - 1`. [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper), [transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
- None of the three READMEs describes typing into other applications.
