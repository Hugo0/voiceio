from common import page, write, cta, faq_html, esc, CHECKED, COMPARE

ROWS = ["platforms", "price", "processing", "models", "open", "live", "vocab", "learns", "control"]
ROW_LABEL = {
    "platforms": "Platforms", "price": "Price", "processing": "Where speech is decoded", "models": "Speech models",
    "open": "Open source", "live": "Live typing into apps", "vocab": "Custom vocabulary",
    "learns": "Learns from you", "control": "Voice commands",
}
V = {  # voiceio, from README / CONTRIBUTING, v1.5.0
    "platforms": "Linux: Wayland and X11 (GNOME, KDE, Hyprland, sway, i3). Untested best-effort path on Windows/macOS",
    "price": "Free, MIT",
    "processing": "On your computer. Optional text-only cloud cleanup, off by default",
    "models": "faster-whisper small (default), medium, large-v3-turbo, distil-large-v3; experimental Parakeet and whisper.cpp server",
    "open": "Yes (MIT)",
    "live": "Yes: underlined preview in the focused app through IBus, corrected in place, committed when you stop",
    "vocab": "Hotwords ranked by use (about 35 per decode) plus find-and-replace corrections",
    "learns": "Yes: optional LoRA fine-tune on your own recordings, on CPU, kept only if it wins on held-out clips",
    "control": "Dictation commands only (\"new line\", \"scratch that\", \"correct that\")",
}

C = {}

C["voxtype"] = dict(
    name="Voxtype", url="https://voxtype.io",
    title="voiceio vs Voxtype: local Linux dictation compared",
    desc="Voxtype vs voiceio: two free, local, open-source Linux dictation tools. Engines, live typing, vocabulary and fine-tuning compared (checked 2026-10-10).",
    lede="Voxtype and voiceio are both free, MIT-licensed, local dictation tools for Linux with no cloud and no telemetry by default. Voxtype is a Rust binary with nine speech engines, GPU builds for AMD, NVIDIA and Intel, a meeting mode and a macOS beta. voiceio is a Python tool built on faster-whisper that streams a live, corrected-in-place preview through IBus and can fine-tune its model on your own voice. If you want the widest choice of engines and hardware, Voxtype is the stronger pick; if your errors are names and jargon you say every day, voiceio's fine-tuning is the difference.",
    chips=["both MIT", "both local", "both Linux"],
    them_if=[
        "You want many engines: Whisper, Parakeet, Moonshine, SenseVoice, Cohere Transcribe and more, including CJK and 1600+ languages through its multilingual engines.",
        "You have a GPU to use: Vulkan, CUDA 12/13, MIGraphX for Radeon, OpenVINO for Intel NPUs.",
        "You want a single native binary from AUR, .deb, .rpm or Homebrew, with no Python.",
        "You need meeting transcription with speaker attribution and SRT/VTT export, or a macOS build (beta).",
        "You prefer push-to-talk (hold the key, release to type).",
    ],
    us_if=[
        "The same names and terms keep coming out wrong: <code>voiceio learn</code> fine-tunes the model on your own recordings and only switches when it measurably wins.",
        "You want to see words as you speak, as an underlined preview corrected in place rather than typed and backspaced.",
        "You want vocabulary hotwords ranked by how often you use them, plus correction rules and a \"correct that\" voice command.",
        "You want to dictate from your phone through a server you run, with the same vocabulary and model.",
    ],
    them={
        "platforms": "Linux (Hyprland, Niri, sway, River, GNOME, KDE, X11); macOS beta",
        "price": "Free, MIT",
        "processing": "On your computer by default; optional remote Whisper server or Soniox cloud backend",
        "models": "Nine engines: Whisper (whisper.cpp), Parakeet, Moonshine, SenseVoice, Paraformer, Dolphin, Omnilingual, Cohere Transcribe, OpenVINO Whisper",
        "open": "Yes (MIT)",
        "live": "Optional streaming mode types at the cursor as you speak (toggle activation; experimental revision mode corrects with backspace). Default: type after release",
        "vocab": "Whisper initial prompt, replacement table, spoken punctuation, post-processing command",
        "learns": "Not stated",
        "control": "Spoken punctuation; no computer control stated",
    },
    faq=[
        ("Is voiceio a Voxtype alternative?",
         "Yes. Both are local, open-source dictation tools for Linux under the MIT license. voiceio adds a live IBus preview and per-user fine-tuning; Voxtype offers more engines, GPU builds, meeting mode and a macOS beta."),
        ("Which is faster?",
         "It depends on engine and hardware. Voxtype's README reports Cohere Transcribe at 9-11x realtime on a Zen 4 CPU and supports GPU backends. voiceio's default whisper-small runs at about 5x realtime on a laptop CPU. Measure on your own machine."),
        ("Do both work on Wayland?",
         "Yes. Voxtype types with wtype, dotool or ydotool and uses compositor keybindings. voiceio uses an IBus input method first, with wtype and ydotool as fallbacks."),
        ("Does Voxtype learn my voice?",
         "Its README and docs (checked 2026-10-10) do not describe fine-tuning or voice adaptation. It offers an initial prompt and replacement rules for domain terms."),
    ],
    sources=[("Voxtype README", "https://github.com/peteonrails/voxtype"), ("Voxtype configuration docs", "https://github.com/peteonrails/voxtype/blob/main/docs/CONFIGURATION.md"),
             ("Voxtype macOS install", "https://github.com/peteonrails/voxtype/blob/main/docs/INSTALL_MACOS.md"), ("voxtype.io", "https://voxtype.io"),
             ("Voxtype releases (v1.1.0)", "https://github.com/peteonrails/voxtype/releases")],
)

C["aqua-voice"] = dict(
    name="Aqua Voice", url="https://aquavoice.com",
    title="voiceio vs Aqua Voice: offline Linux alternative",
    desc="Aqua Voice vs voiceio: a cloud dictation app for Mac, Windows and phones vs local, open-source dictation for Linux. Checked 2026-10-10.",
    lede="Aqua Voice is a polished commercial dictation app for Mac, Windows, iPhone and Android. It runs its own speech model, Avalon, in the cloud, adds language-model formatting, and can read on-screen context to get names and code right. voiceio is free, open-source dictation for Linux that decodes speech on your own computer and fine-tunes itself on your voice. Aqua does not run on Linux and needs a connection for dictation; voiceio runs only on Linux, and does not try to match Aqua's AI editing features.",
    chips=["Aqua: cloud", "voiceio: local", "different platforms"],
    them_if=[
        "You use macOS, Windows, iPhone or Android, and want one subscription across them.",
        "You want AI formatting and editing: custom instructions, Edit Mode (rewrite selected text by voice), Deep Context from what is on screen.",
        "You want a managed product with team billing, SSO, SOC 2 Type II and ISO 27001.",
        "You dictate in many languages (Aqua lists 49).",
    ],
    us_if=[
        "You are on Linux. Aqua's own comparison page says it does not run there.",
        "You want audio to stay on your machine and dictation to work offline.",
        "You want software you can read and change (MIT), with no account and no word limits.",
        "You want the model itself to learn your names from your own voice, not only a word list.",
    ],
    them={
        "platforms": "Mac, Windows 10/11, iOS 17+, Android 10+. Not Linux (a CLI for settings and file transcription installs on Linux)",
        "price": "Free (1,000 words); Pro $8/mo yearly or $10 monthly; Max $24/mo yearly or $30 monthly; Business from $19/seat/mo (pricing page, 2026-10-10)",
        "processing": "Cloud; internet needed for cloud dictation",
        "models": "Own model, Avalon, plus language-model post-processing; Realtime mode uses a third-party provider",
        "open": "No",
        "live": "Default inserts after you stop. Realtime Mode (Max plan) shows words live in Aqua's toolbar; text lands in the app when you finish",
        "vocab": "Dictionary, replacements, custom instructions, synced across devices",
        "learns": "Not stated (manual dictionary; optional screen context)",
        "control": "Edit Mode, \"Send it\" (Max), Computer Control beta on macOS",
    },
    faq=[
        ("Does Aqua Voice work on Linux?",
         "No. Aqua's own comparison page says it runs on Mac, Windows, iPhone and Android (checked 2026-10-10). A separate Aqua CLI installs on Linux, but it manages settings and transcribes files rather than providing live dictation."),
        ("Is there an offline Aqua Voice alternative?",
         "voiceio runs its speech model on your computer and works offline after a one-time model download. On Linux, <a href=\"/vs/voxtype/\">Voxtype</a> and <a href=\"/vs/nerd-dictation/\">nerd-dictation</a> are also local."),
        ("Is Aqua Voice more accurate than Whisper?",
         "Aqua publishes benchmarks for its Avalon model, including Open ASR Leaderboard results. voiceio's approach is different: fine-tune a local Whisper model on your own recordings and measure it on your held-out clips with <code>voiceio learn eval</code>."),
        ("How much does Aqua Voice cost?",
         "As listed on aquavoice.com/pricing on 2026-10-10: a free tier with 1,000 words, Pro at $8/month billed yearly ($10 monthly), Max at $24/month billed yearly ($30 monthly), and Business from $19 per seat per month billed yearly. voiceio is free."),
    ],
    sources=[("Aqua Voice pricing", "https://aquavoice.com/pricing"), ("Aqua Voice llms.txt", "https://aquavoice.com/llms.txt"),
             ("Aqua vs OpenWhispr (Linux answer)", "https://aquavoice.com/vs/openwhispr"), ("Aqua Realtime", "https://aquavoice.com/realtime"),
             ("Aqua FAQ", "https://aquavoice.com/info/faq"), ("Aqua privacy policy", "https://aquavoice.com/info/privacy"), ("Aqua CLI docs", "https://aquavoice.com/docs/cli")],
)

C["wispr-flow"] = dict(
    name="Wispr Flow", url="https://wisprflow.ai",
    title="voiceio vs Wispr Flow: a Wispr Flow for Linux?",
    desc="Wispr Flow vs voiceio: Wispr Flow has no Linux app and transcribes in the cloud; voiceio is local, open-source Linux dictation. Checked 2026-10-10.",
    lede="Wispr Flow is a commercial dictation app for Mac, Windows, iPhone and Android. It transcribes in the cloud with its own speech model, Canto, cleans up filler words and self-corrections, and adds words you correct to your dictionary automatically. Its documentation says Linux is not supported. voiceio is free, open-source dictation for Linux that decodes on your own computer and can fine-tune its speech model on your voice. If you are on Linux, Wispr Flow is not an option today; elsewhere, it offers features voiceio does not.",
    chips=["Wispr: cloud", "voiceio: local", "no Wispr app for Linux"],
    them_if=[
        "You use macOS, Windows, iPhone or Android.",
        "You want AI cleanup out of the box: filler removal, \"backtrack\" self-corrections, per-app writing styles, Command Mode (beta).",
        "You dictate in many languages (100+ with auto-detection).",
        "You want a team product: shared dictionaries, SSO, SOC 2 Type II, a HIPAA BAA.",
    ],
    us_if=[
        "You are on Linux, which Wispr Flow's help center lists as unsupported.",
        "You want speech decoded on your own machine and dictation that works offline.",
        "You want to see your words as you speak; Wispr's docs say desktop dictation inserts the text after you stop.",
        "You want open source, no account and no word limits.",
    ],
    them={
        "platforms": "macOS 12+, Windows 10/11, iOS 18.3+, Android 13+. Linux not supported",
        "price": "Free (2,000 words/week desktop); Pro $15/user/mo monthly or $12 yearly; Growth $23 or $18; Enterprise custom (pricing page, 2026-10-10)",
        "processing": "Cloud; \"Transcription always occurs on the cloud\"",
        "models": "Own model, Canto (since Sept 2026), plus third-party AI providers",
        "open": "No",
        "live": "No: completed text is inserted after you stop",
        "vocab": "Dictionary, snippets, team dictionaries, synced",
        "learns": "Adds words you correct to your dictionary (can be turned off); per-app styles",
        "control": "Command Mode (beta): edit selected text, draft replies, search",
    },
    faq=[
        ("Is there a Wispr Flow for Linux?",
         "No. Wispr Flow's help center says Linux is not supported and there is no native Linux app (checked 2026-10-10). voiceio is a free, open-source dictation tool built for Linux, on Wayland and X11."),
        ("Does Wispr Flow work offline?",
         "No. Its documentation says transcription always occurs in the cloud and an internet connection is required. voiceio decodes on your computer and works offline after downloading the model once."),
        ("Does voiceio clean up filler words like Wispr Flow?",
         "voiceio has rule-based cleanup and filler removal, spoken numbers to digits, and dictation commands. An optional LLM pass (text only, off by default) can polish the final transcript. It does not offer Wispr's style rewriting or Command Mode."),
        ("How much does Wispr Flow cost?",
         "As listed on wisprflow.ai/pricing on 2026-10-10: free with 2,000 words per week on desktop, Pro at $15 per user per month ($12 billed yearly), Growth at $23 ($18 yearly), and custom Enterprise pricing. voiceio is free."),
    ],
    sources=[("Wispr Flow pricing", "https://wisprflow.ai/pricing"), ("Supported devices", "https://docs.wisprflow.ai/articles/1036674442-supported-devices-and-system-requirements"),
             ("Using Flow with Linux, WSL and terminals", "https://docs.wisprflow.ai/articles/6478598909-using-flow-with-linux-wsl-and-terminal-applications"),
             ("What is Flow", "https://docs.wisprflow.ai/articles/2772472373-what-is-flow"), ("Data controls", "https://wisprflow.ai/data-controls"),
             ("Dictionary", "https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary"), ("What's new (Canto)", "https://wisprflow.ai/whats-new")],
)

C["superwhisper"] = dict(
    name="Superwhisper", url="https://superwhisper.com",
    title="voiceio vs Superwhisper: dictation on Linux compared",
    desc="Superwhisper vs voiceio: Whisper dictation for Mac, Windows and phones (Linux preview for Omarchy) vs open-source Linux dictation. Checked 2026-10-10.",
    lede="Superwhisper is a commercial dictation app for macOS, Windows, iOS and Android that can run Whisper and other models locally or use cloud models, with AI \"modes\" that reformat what you say. On 2026-10-09 it published a prerelease Linux build that targets Omarchy on x86-64; its docs still list Linux as coming soon. voiceio is free, open-source dictation built for Linux, with a live in-app preview through IBus and fine-tuning on your own voice. Superwhisper has far more model choice and polish across platforms; voiceio is open source and Linux-first.",
    chips=["both can run locally", "Superwhisper: Linux preview", "voiceio: MIT"],
    them_if=[
        "You use macOS, Windows, iPhone or Android, and want one license across them (a lifetime option exists).",
        "You want a choice of local models (Whisper, Parakeet, Cohere) and cloud models (Deepgram, ElevenLabs, its own S1).",
        "You want AI modes that turn speech into emails, messages or notes with custom prompts.",
        "You need meeting recording, file transcription or 100+ languages.",
    ],
    us_if=[
        "You run a Linux desktop other than Omarchy, or want a stable rather than prerelease Linux build.",
        "You want open source: code you can read, audit and change.",
        "You want words to appear in the app as you speak; Superwhisper's realtime preview shows in its own window and pastes the result at the end.",
        "You want a model that learns your voice through fine-tuning.",
    ],
    them={
        "platforms": "macOS 14+, Windows 10/11, iOS, Android. Linux: prerelease build for Omarchy (x86-64), 2026-10-09",
        "price": "Free (local Whisper models); Pro $8.49/month, $84.99/year or $249.99 lifetime (docs plans page, 2026-10-10)",
        "processing": "Local or cloud, chosen per mode",
        "models": "Local Whisper (whisper.cpp), Parakeet, Cohere Transcribe; cloud S1-Voice, Deepgram Nova, ElevenLabs Scribe; LLMs for formatting",
        "open": "No",
        "live": "Realtime preview (Pro, streaming models only) in the recording window; text pasted at the cursor after you stop",
        "vocab": "Vocabulary hints and replacements (free and Pro)",
        "learns": "Not stated (manual vocabulary, modes, context awareness)",
        "control": "Not stated beyond shortcuts and modes",
    },
    faq=[
        ("Does Superwhisper work on Linux?",
         "Partly. On 2026-10-09 Superwhisper published a prerelease Linux build that, per its README, \"targets x86-64 systems running Omarchy\". Its docs still list Linux as coming soon (checked 2026-10-10). voiceio supports GNOME, KDE, Hyprland, sway and i3 on Wayland and X11."),
        ("Is there an open-source Superwhisper alternative?",
         "voiceio is MIT licensed and, like Superwhisper's free tier, runs Whisper locally. <a href=\"/vs/voxtype/\">Voxtype</a> is another open-source option on Linux."),
        ("Do both use Whisper?",
         "Yes. Superwhisper's free local models are Whisper via whisper.cpp. voiceio uses Whisper via faster-whisper by default, with experimental Parakeet and whisper.cpp backends."),
        ("How much does Superwhisper cost?",
         "Its docs list a free plan with local Whisper models and Pro at $8.49 per month, $84.99 per year or $249.99 once (checked 2026-10-10). voiceio is free."),
    ],
    sources=[("Superwhisper docs: introduction", "https://superwhisper.com/docs/get-started/introduction.md"), ("Plans", "https://superwhisper.com/docs/billing/plans.md"),
             ("Voice models", "https://superwhisper.com/docs/models/voice.md"), ("Realtime", "https://superwhisper.com/docs/common-issues/realtime.md"),
             ("Vocabulary", "https://superwhisper.com/docs/get-started/interface-vocabulary.md"), ("Linux releases repo", "https://github.com/superultrainc/superwhisper-linux-releases"),
             ("superwhisper.com", "https://superwhisper.com")],
)

C["dragon"] = dict(
    name="Dragon", url="https://dragon.nuance.com/en-us/dragon.html",
    title="voiceio vs Dragon: a Dragon alternative for Linux",
    desc="Dragon vs voiceio: Nuance's speech recognition for Windows (no Linux, Mac discontinued) vs free, local, open-source Linux dictation. Checked 2026-10-10.",
    lede="Dragon, from Nuance (part of Microsoft), is long-established speech recognition for Windows, with locally installed editions (Professional, Legal, Law Enforcement v16), cloud editions (Professional Anywhere, Dragon Medical One) and a mobile app. It adapts to a single speaker's voice and offers full hands-free control of the computer. Dragon for Mac was discontinued in 2018 and there is no Linux version. voiceio is free, open-source dictation for Linux that also learns your voice, by fine-tuning a Whisper model on your recordings, but it is a dictation tool, not a hands-free control system.",
    chips=["Dragon: Windows", "voiceio: Linux", "both adapt to your voice"],
    them_if=[
        "You are on Windows and need hands-free control: open apps, navigate menus, mouse grid, macros and scripting.",
        "You need an accessibility-certified tool (Section 508) that works with screen readers such as JAWS.",
        "You work in legal, law enforcement or medicine and want a domain edition and vocabulary.",
        "Your organization needs managed deployment, shared vocabularies and enterprise support.",
    ],
    us_if=[
        "You are on Linux, where Dragon has no version.",
        "You want something free and open source, with no license to buy.",
        "You want voice adaptation that runs on your own hardware and that you can measure: <code>voiceio learn eval</code> scores every model on clips it never trained on.",
    ],
    them={
        "platforms": "Windows 10/11 (desktop and cloud editions); iOS and Android (Dragon Anywhere Mobile, US and Canada). No Linux; Mac discontinued 2018",
        "price": "No public price on the official pages (contact sales; subscriptions for cloud editions)",
        "processing": "Local for v16 desktop editions; cloud (Azure) for Anywhere and Medical One",
        "models": "Proprietary Nuance deep-learning engine",
        "open": "No",
        "live": "Yes: \"words appear on the computer screen, instantly\"",
        "vocab": "Custom words, import/export, shared word lists, domain vocabularies",
        "learns": "Yes: adapts to the user's voice and most-used words",
        "control": "Full hands-free operation, mouse grid, macros, scripting",
    },
    faq=[
        ("Is there a Dragon NaturallySpeaking for Linux?",
         "No. Dragon's current desktop editions require Windows 10 or 11, and no Linux client is listed (checked 2026-10-10). On Linux, voiceio, <a href=\"/vs/voxtype/\">Voxtype</a> and <a href=\"/vs/nerd-dictation/\">nerd-dictation</a> provide dictation; <a href=\"/vs/talon/\">Talon</a> provides voice control on X11."),
        ("Does voiceio replace Dragon's voice commands?",
         "No. voiceio handles dictation and a few editing commands (\"new line\", \"scratch that\", \"correct that\"). For hands-free control of the whole computer, look at <a href=\"/vs/talon/\">Talon</a>."),
        ("Does voiceio learn my voice like Dragon?",
         "In a different way. Dragon adapts its profile as you dictate. voiceio keeps your recordings (only if you opt in), trains a LoRA fine-tune of Whisper on them on your CPU, and switches to it only when it beats the current model on held-out clips."),
        ("Is Dragon still available for Mac?",
         "No. Nuance's support site says Dragon Professional Individual for Mac was discontinued effective 22 October 2018."),
    ],
    sources=[("Dragon product lineup", "https://dragon.nuance.com/en-us/dragon.html"), ("Dragon Professional", "https://dragon.nuance.com/en-us/dragon-professional"),
             ("Dragon Professional v16 data sheet", "https://dragon.nuance.com/shared/data-sheets/ds-dragon-professional-v16-en-us.pdf"),
             ("Professional v16 vs Anywhere comparison", "https://dragon.nuance.com/shared/data-sheets/fm-dragon-professional-v16-and-dragon-professional-anywhere-comparison-en-us.pdf"),
             ("Dragon Professional Anywhere", "https://dragon.nuance.com/en-us/dragon-professional-anywhere"),
             ("Dragon accessibility", "https://dragon.nuance.com/en-us/accessibility-solutions-for-business"),
             ("Dragon for Mac discontinued", "https://nuance.custhelp.com/app/answers/detail/a_id/27843")],
)

C["nerd-dictation"] = dict(
    name="nerd-dictation", url="https://github.com/ideasman42/nerd-dictation",
    title="voiceio vs nerd-dictation: offline Linux dictation",
    desc="nerd-dictation vs voiceio: two offline, open-source Linux dictation tools. VOSK vs Whisper, typing, customization and learning compared (2026-10-10).",
    lede="nerd-dictation is a single-file Python script that does offline speech to text on Linux with the VOSK engine. It has no background process, types as you speak with xdotool, ydotool, dotool or wtype, and can be customized with a Python config script. voiceio is a larger tool: a background service built on Whisper, with punctuation and casing from the model, a live IBus preview, vocabulary hotwords and fine-tuning on your voice. nerd-dictation is the smaller, more hackable choice; voiceio does more out of the box and costs more to install.",
    chips=["both offline", "both open source", "VOSK vs Whisper"],
    them_if=[
        "You want something tiny: one Python file, minimal dependencies, nothing running in the background.",
        "You like to script it yourself: a Python hook lets you rewrite text any way you want.",
        "You want a restricted grammar: VOSK can be limited to a fixed set of phrases for higher accuracy on commands.",
        "Your hardware is modest; VOSK's small models are light.",
    ],
    us_if=[
        "You want punctuation and capitalization; nerd-dictation's README notes that VOSK text is all lower-case.",
        "You want Whisper-level accuracy on long, natural sentences.",
        "You want proper nouns right, through vocabulary hotwords, corrections and fine-tuning on your voice.",
        "You want an in-place, corrected live preview rather than raw keystrokes.",
    ],
    them={
        "platforms": "Desktop Linux (X11 and Wayland)",
        "price": "Free, GPL-3.0",
        "processing": "On your computer (VOSK)",
        "models": "VOSK models",
        "open": "Yes (GPL-3.0)",
        "live": "Yes: types as you speak by default, via xdotool, ydotool, dotool or wtype",
        "vocab": "Python config hook; optional VOSK grammar file",
        "learns": "Not stated",
        "control": "None built in; scriptable through the config hook",
    },
    faq=[
        ("Is voiceio a good nerd-dictation alternative?",
         "If you want Whisper accuracy, punctuation and casing, a live preview and fine-tuning, yes. If you want the smallest possible tool with no background process, nerd-dictation remains a good fit."),
        ("Does nerd-dictation work on Wayland?",
         "Yes, using ydotool, dotool or wtype; its default, xdotool, is X11 only (per its README, checked 2026-10-10)."),
        ("Does nerd-dictation use Whisper?",
         "No. It uses the VOSK API and VOSK models. voiceio uses Whisper through faster-whisper."),
        ("Which uses fewer resources?",
         "nerd-dictation, very likely: it runs only while you dictate and VOSK's small models are light. voiceio keeps a Whisper worker loaded so it can respond immediately."),
    ],
    sources=[("nerd-dictation README", "https://github.com/ideasman42/nerd-dictation"), ("nerd-dictation license", "https://github.com/ideasman42/nerd-dictation/blob/main/license.rst"),
             ("nerd-dictation commits", "https://github.com/ideasman42/nerd-dictation/commits/main")],
)

C["talon"] = dict(
    name="Talon", url="https://talonvoice.com",
    title="voiceio vs Talon: voice control vs dictation on Linux",
    desc="Talon vs voiceio: hands-free computer control (macOS, Windows, Linux X11) vs local dictation for Linux that learns your voice. Checked 2026-10-10.",
    lede="Talon is a hands-free input system: you drive the whole computer by voice, with eye tracking and noise input, and many people use it to code without a keyboard. It runs on macOS, Windows and Linux under X11, is free to use with a paid Patreon tier, and is closed source. voiceio does one narrower job: dictation into any app on Linux, on Wayland as well as X11, with a model that fine-tunes on your voice. If you need to control your computer by voice, Talon is the tool; if you want to write by voice on a modern Linux desktop, voiceio is.",
    chips=["Talon: voice control", "voiceio: dictation", "Talon: X11 only"],
    them_if=[
        "You need to operate the computer without your hands: commands, mouse by eye tracking, noise input such as a pop or hiss.",
        "You want to write code by voice with a large, scriptable command set (the community set is MIT).",
        "You use macOS or Windows, or Linux under X11.",
        "You have RSI or another reason to avoid the keyboard and mouse entirely.",
    ],
    us_if=[
        "You are on Wayland, which Talon does not support.",
        "You mostly write prose, messages and notes and want a live preview of long sentences.",
        "You want open source and a model that learns your names from your own voice.",
        "You want to keep using your keyboard and add dictation on a hotkey.",
    ],
    them={
        "platforms": "macOS, Windows, Linux (X11 only; no Wayland). Talon 1.0 is planned as the last public Linux release",
        "price": "Free to use; Talon+ via Patreon (tier price not stated on official pages)",
        "processing": "Built-in Conformer engine ships with Talon; offline operation not stated explicitly in the docs we read",
        "models": "Conformer (wav2letter); Dragon optional; Talon+ adds a Conformer + Whisper hybrid",
        "open": "No (community command set is MIT)",
        "live": "Has a dictation mode; Talon 1.0 adds a streaming dictation model (Hum)",
        "vocab": "Community set: custom vocabulary and replacements",
        "learns": "Not stated",
        "control": "Full computer control, eye tracking, noise input, scripting",
    },
    faq=[
        ("Does Talon work on Wayland?",
         "No. The Talon wiki says Talon does not support Wayland, so Linux users need an X11 session. Talon 1.0's changelog (September 2026) also says it is planned to be the last public release for Linux/X11, with ongoing Linux support listed as a Talon+ feature."),
        ("Can I use Talon and voiceio together?",
         "On X11, likely yes, on different hotkeys: Talon for commands, voiceio for long-form dictation. We have not tested the combination."),
        ("Is Talon open source?",
         "No. The Talon wiki says Talon is closed source. The community command set it relies on is MIT licensed."),
        ("Is voiceio good for coding by voice?",
         "For prose, comments and commit messages, yes. For navigating and editing code by voice, Talon's command system is far more capable."),
    ],
    sources=[("talonvoice.com", "https://talonvoice.com/"), ("Talon 1.0 changelog", "https://talonvoice.com/dl/latest/changelog.html"),
             ("Talon wiki: installation (Wayland)", "https://talon.wiki/Resource%20Hub/Talon%20Installation/installation_guide/"),
             ("Talon wiki: speech engines", "https://talon.wiki/Resource%20Hub/Speech%20Recognition/speech%20engines/"),
             ("Talon wiki: FAQ", "https://talon.wiki/Help/FAQ"), ("talonhub/community", "https://github.com/talonhub/community")],
)

C["whisper"] = dict(
    name="Whisper", url="https://github.com/openai/whisper",
    title="voiceio and Whisper: Whisper dictation on Linux",
    desc="How voiceio relates to OpenAI Whisper, whisper.cpp and faster-whisper: it builds on Whisper and adds typing, live preview, vocabulary and fine-tuning.",
    h1="voiceio and Whisper",
    lede="voiceio does not compete with Whisper; it is built on it. OpenAI's Whisper is a speech recognition model (MIT licensed), and whisper.cpp and faster-whisper are fast ways to run it. None of them types into other apps: they turn audio into text. voiceio is the dictation layer on top: a hotkey, a microphone, streaming decode with a live preview in the focused app, vocabulary and corrections, and fine-tuning Whisper on your own voice. It runs faster-whisper by default and can use a whisper.cpp server.",
    chips=["built on Whisper", "faster-whisper default", "whisper.cpp optional"],
    them_label="Whisper / libraries",
    them_if=[
        "You want to transcribe audio files or build your own pipeline: use Whisper, faster-whisper or whisper.cpp directly.",
        "You need a specific runtime or platform: whisper.cpp runs on macOS, iOS, Android, Windows, WebAssembly and more, with Metal, CUDA and Vulkan backends.",
        "You want a quick microphone test: whisper.cpp's stream example transcribes the mic continuously.",
    ],
    us_if=[
        "You want to dictate into any app on Linux with a hotkey, without writing glue code.",
        "You want a live preview while you speak and fast commits for long dictations.",
        "You want vocabulary hotwords, correction rules and spoken numbers handled for you.",
        "You want Whisper fine-tuned on your voice, evaluated on your own held-out clips.",
    ],
    them={
        "platforms": "Libraries and CLIs: Python (Whisper, faster-whisper); C/C++ for many platforms (whisper.cpp)",
        "price": "Free, MIT (code and weights for Whisper)",
        "processing": "Wherever you run it",
        "models": "tiny, base, small, medium, large-v3, large-v3-turbo (and .en variants)",
        "open": "Yes (MIT)",
        "live": "No typing into apps; whisper.cpp has a mic stream example",
        "vocab": "initial_prompt (all); hotwords (faster-whisper)",
        "learns": "Not built in; fine-tuning is up to you",
        "control": "None",
    },
    faq=[
        ("Is voiceio just Whisper?",
         "voiceio uses Whisper models for recognition. What it adds is the dictation app: audio capture, hotkeys, streaming decode, typing into the focused app through IBus, text cleanup, vocabulary ranking, corrections and a fine-tuning pipeline."),
        ("Which Whisper model does voiceio use?",
         "Whisper small through faster-whisper by default. You can switch to medium, large-v3-turbo or distil-large-v3 with <code>voiceio models use</code>, or point it at a whisper.cpp server."),
        ("Why does voiceio still use Whisper rather than a faster model?",
         "Most dictation errors are names and jargon, so the ability to bias the model matters more than raw speed. voiceio's contributors measured Parakeet as faster, but its biasing mode lost speech on real clips; the reasoning is in <a href=\"https://github.com/Hugo0/voiceio/blob/main/CONTRIBUTING.md#model-choice-why-whisper-and-what-would-change-it\">CONTRIBUTING.md</a>."),
        ("Can I use Whisper for dictation on Wayland?",
         "Yes. See <a href=\"/whisper-dictation-wayland/\">Whisper dictation on Wayland</a>."),
    ],
    sources=[("openai/whisper", "https://github.com/openai/whisper"), ("ggml-org/whisper.cpp", "https://github.com/ggml-org/whisper.cpp"),
             ("whisper.cpp server", "https://github.com/ggml-org/whisper.cpp/tree/master/examples/server"),
             ("SYSTRAN/faster-whisper", "https://github.com/SYSTRAN/faster-whisper"),
             ("faster-whisper transcribe.py (hotwords)", "https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py")],
)


def table(name, them, label=None):
    rows = "\n".join(
        f'      <tr><th scope="row">{ROW_LABEL[k]}</th><td class="us">{esc(V[k])}</td><td>{esc(them[k])}</td></tr>' for k in ROWS)
    return f"""<div class="table-scroll"><table class="cmp">
    <caption class="label" style="text-align:left;padding:12px 14px 0">voiceio vs {esc(label or name)} · checked {CHECKED}</caption>
    <thead><tr><th scope="col">&nbsp;</th><th scope="col">voiceio 1.5</th><th scope="col">{esc(label or name)}</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table></div>"""


def related(slug):
    items = [(s, l) for s, l in COMPARE if s != slug]
    return "\n".join(f'    <a href="/vs/{s}/"><span class="label">compare</span><b>{esc(C[s].get("h1", "voiceio vs " + l))}</b></a>' for s, l in items)


def build(slug):
    c = C[slug]
    name = c["name"]
    h1 = c.get("h1", f"voiceio vs {name}")
    lis = lambda xs: "\n".join(f"      <li>{x}</li>" for x in xs)
    srcs = "\n".join(f'    <li><a href="{u}" rel="nofollow">{esc(t)}</a></li>' for t, u in c["sources"])
    chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in c["chips"]) + f'<span class="chip on">checked {CHECKED}</span>'
    body = f"""<header class="head">
  <span class="label">// compare</span>
  <h1>{esc(h1)}</h1>
  <p class="lede">{c["lede"]}</p>
  <div class="meta">{chips}</div>
</header>

<section class="part" aria-labelledby="choose">
  <span class="label">// which one</span>
  <h2 id="choose">Which should you use?</h2>
  <div class="choose">
    <div>
      <span class="label">{esc(c.get("them_label", name))}</span>
      <h3>Choose {esc(c.get("them_label", name))} if…</h3>
      <ul>
{lis(c["them_if"])}
      </ul>
    </div>
    <div class="us">
      <span class="label">voiceio</span>
      <h3>Choose voiceio if…</h3>
      <ul>
{lis(c["us_if"])}
      </ul>
    </div>
  </div>
</section>

<section class="part" aria-labelledby="features">
  <span class="label">// side by side</span>
  <h2 id="features">Feature comparison</h2>
  {table(name, c["them"], c.get("them_label"))}
  <p class="note">{esc(name)} facts are from its official site, docs or repository, read on {CHECKED} (sources below). Prices are as listed on that date and can change. voiceio facts are from its <a href="https://github.com/Hugo0/voiceio#readme">README</a>. Spotted something out of date? <a href="https://github.com/Hugo0/voiceio/issues/new">Open an issue</a>.</p>
</section>

<section class="part" aria-labelledby="faq">
  <span class="label">// faq</span>
  <h2 id="faq">Questions</h2>
  {faq_html(c["faq"])}
</section>

{cta()}

<section class="part sources" aria-labelledby="src">
  <span class="label">// sources · last checked {CHECKED}</span>
  <h2 id="src" style="font-size:20px">Sources</h2>
  <ul>
{srcs}
  </ul>
</section>

<section class="part" aria-labelledby="more">
  <span class="label">// more comparisons</span>
  <h2 id="more" style="font-size:24px">Other comparisons</h2>
  <div class="cards">
{related(slug)}
  </div>
</section>"""
    write(f"/vs/{slug}/", page(
        path=f"/vs/{slug}/", og=slug, title=c["title"], desc=c["desc"],
        crumbs=[("voiceio", "/"), ("Compare", "/vs/"), (h1, f"/vs/{slug}/")],
        body=body, faq=c["faq"], current="compare"))


HUB_COLS = ["platforms", "price", "processing", "open", "live", "learns"]
HUB_SHORT = {
    "voiceio": ["Linux (Wayland, X11)", "Free", "Local", "Yes, MIT", "Yes, in-app preview", "Yes, fine-tunes on your voice"],
    "voxtype": ["Linux; macOS beta", "Free", "Local by default", "Yes, MIT", "Optional streaming mode", "Not stated"],
    "aqua-voice": ["Mac, Windows, iOS, Android", "Free tier; Pro $8/mo billed yearly", "Cloud", "No", "Max plan, in Aqua's toolbar", "Not stated"],
    "wispr-flow": ["Mac, Windows, iOS, Android", "Free tier; Pro $12/mo billed yearly", "Cloud", "No", "No, inserts after you stop", "Auto-adds corrected words"],
    "superwhisper": ["Mac, Windows, iOS, Android; Linux preview (Omarchy)", "Free tier; Pro $8.49/mo", "Local or cloud", "No", "Preview in its window (Pro)", "Not stated"],
    "dragon": ["Windows; mobile apps", "No public price", "Local or cloud, by edition", "No", "Yes", "Yes, adapts to your voice"],
    "nerd-dictation": ["Linux", "Free", "Local (VOSK)", "Yes, GPL-3.0", "Yes, types as you speak", "Not stated"],
    "talon": ["macOS, Windows, Linux X11", "Free; Patreon tier", "Engine ships with app", "No", "Dictation mode", "Not stated"],
    "whisper": ["Libraries, many platforms", "Free", "Wherever you run it", "Yes, MIT", "Does not type into apps", "Not built in"],
}


BLURB = {
    "voxtype": "Local, MIT-licensed Linux dictation with nine engines and GPU builds.",
    "aqua-voice": "Cloud dictation app for Mac, Windows and phones.",
    "wispr-flow": "Cloud dictation app with AI cleanup; no Linux app.",
    "superwhisper": "Local and cloud models on Apple platforms and Windows; Linux preview.",
    "dragon": "Speech recognition and voice control for Windows.",
    "nerd-dictation": "A single-file, offline VOSK dictation script for Linux.",
    "talon": "Hands-free computer control by voice; X11 only on Linux.",
    "whisper": "The model voiceio is built on, and the libraries that run it.",
}


def hub():
    head = "".join(f'<th scope="col">{ROW_LABEL[k]}</th>' for k in HUB_COLS)
    rows = []
    for slug in ["voiceio"] + [s for s, _ in COMPARE]:
        cells = HUB_SHORT[slug]
        if slug == "voiceio":
            name = "<b>voiceio</b>"
            tds = "".join(f'<td class="us">{esc(x)}</td>' for x in cells)
        else:
            name = f'<a href="/vs/{slug}/">{esc(C[slug]["name"])}</a>'
            tds = "".join(f"<td>{esc(x)}</td>" for x in cells)
        rows.append(f'      <tr><th scope="row">{name}</th>{tds}</tr>')
    rows = "\n".join(rows)
    cards = "\n".join(
        f'    <a href="/vs/{s}/"><span class="label">compare</span><b>{esc(C[s].get("h1", "voiceio vs " + C[s]["name"]))}</b><span>{esc(BLURB[s])}</span></a>'
        for s, _ in COMPARE)
    faq = [
        ("What is the best voice dictation for Linux?",
         "It depends on what you need. For local dictation into any app on Wayland or X11, voiceio and <a href=\"/vs/voxtype/\">Voxtype</a> are the most complete open-source options; <a href=\"/vs/nerd-dictation/\">nerd-dictation</a> is the smallest; <a href=\"/vs/talon/\">Talon</a> is for hands-free control on X11. Wispr Flow, Aqua Voice and Dragon do not run on Linux."),
        ("Which dictation tools work offline?",
         "voiceio, Voxtype, nerd-dictation, Superwhisper with local models, and Dragon's locally installed Windows editions. Wispr Flow and Aqua Voice transcribe in the cloud."),
        ("How were these comparisons made?",
         "From each product's official website, documentation, pricing page or repository, read on 2026-10-10, with links on every page. Where a fact was not stated, we say so rather than guess."),
    ]
    body = f"""<header class="head">
  <span class="label">// compare</span>
  <h1>voiceio vs other dictation tools</h1>
  <p class="lede">voiceio is free, open-source voice dictation for Linux. It decodes speech on your own computer, streams a live preview into the focused app, and can fine-tune its model on your own voice. Here is how it compares with other dictation tools, with sources for every claim. Several are excellent on platforms voiceio does not support; we say so.</p>
  <div class="meta"><span class="chip on">checked {CHECKED}</span><span class="chip">sources on every page</span></div>
</header>

<section class="part hub" aria-labelledby="table">
  <span class="label">// at a glance</span>
  <h2 id="table">At a glance</h2>
  <div class="table-scroll"><table class="cmp">
    <thead><tr><th scope="col">Tool</th>{head}</tr></thead>
    <tbody>
{rows}
    </tbody>
  </table></div>
  <p class="note">Summaries of each product's official pages as read on {CHECKED}. Prices are the lowest paid tier on the official pricing page that day. Full details and links are on each comparison page.</p>
</section>

<section class="part" aria-labelledby="pages">
  <span class="label">// one by one</span>
  <h2 id="pages">Detailed comparisons</h2>
  <div class="cards">
{cards}
  </div>
</section>

<section class="part" aria-labelledby="guides">
  <span class="label">// guides</span>
  <h2 id="guides">Guides</h2>
  <div class="cards">
    <a href="/linux-voice-dictation/"><span class="label">guide</span><b>Voice dictation on Linux</b><span>Set up dictation on Wayland or X11 in four steps.</span></a>
    <a href="/offline-speech-to-text/"><span class="label">guide</span><b>Offline speech to text</b><span>Dictation that never sends audio anywhere.</span></a>
    <a href="/whisper-dictation-wayland/"><span class="label">guide</span><b>Whisper dictation on Wayland</b><span>How Whisper gets text into Wayland apps.</span></a>
  </div>
</section>

<section class="part" aria-labelledby="faq">
  <span class="label">// faq</span>
  <h2 id="faq">Questions</h2>
  {faq_html(faq)}
</section>

{cta()}"""
    write("/vs/", page(
        path="/vs/", og="vs", title="voiceio vs other dictation tools: comparisons",
        desc="How voiceio compares with Voxtype, Wispr Flow, Aqua Voice, Superwhisper, Dragon, nerd-dictation, Talon and Whisper. Sourced, checked 2026-10-10.",
        crumbs=[("voiceio", "/"), ("Compare", "/vs/")], body=body, faq=faq, current="compare"))


if __name__ == "__main__":
    for s, _ in COMPARE:
        build(s)
    hub()
