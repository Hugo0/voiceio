from common import page, write, cta, faq_html, esc

INSTALL_STEPS = """<ol class="steps">
  <li><b>Install the system packages</b> (a C toolchain, PortAudio, IBus). On Debian or Ubuntu:
<pre><code><span class="p">$</span> sudo apt install pipx build-essential python3-dev \\
    portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi</code></pre>
  Fedora, Arch and NixOS commands are in the <a href="https://github.com/Hugo0/voiceio#install">README</a>.</li>
  <li><b>Install voiceio</b> with the desktop extra (microphone, hotkeys, tray):
<pre><code><span class="p">$</span> pipx install 'python-voiceio[desktop]'</code></pre></li>
  <li><b>Run the guided setup.</b> It downloads the speech model, sets your hotkey and installs a systemd user service so voiceio starts on login:
<pre><code><span class="p">$</span> voiceio setup</code></pre></li>
  <li><b>Dictate.</b> Press the hotkey you picked during setup in any app, speak, and press it again. <code>voiceio doctor</code> shows what works; <code>voiceio doctor --fix</code> repairs what it can.</li>
</ol>"""

OTHER_TOOLS = """<ul class="ticks">
  <li><a href="/vs/voxtype/">Voxtype</a>: a local, MIT-licensed Rust dictation tool for Linux with many speech engines and GPU builds.</li>
  <li><a href="/vs/nerd-dictation/">nerd-dictation</a>: a single-file Python script using the offline VOSK engine.</li>
  <li><a href="/vs/talon/">Talon</a>: hands-free computer control by voice; on Linux it runs on X11 only.</li>
  <li><a href="/vs/whisper/">Whisper, whisper.cpp, faster-whisper</a>: speech models and libraries that transcribe audio but do not type into apps by themselves.</li>
</ul>
<p class="muted">Aqua Voice, Wispr Flow and Dragon do not offer Linux apps, and Superwhisper has a prerelease Linux build for Omarchy only (all checked 2026-10-10). See the <a href="/vs/">comparison hub</a>.</p>"""


def linux():
    faq = [
        ("Does voice dictation work on Wayland?",
         "Yes. voiceio types through IBus, which works on Wayland and X11, including GNOME, KDE Plasma, Hyprland and sway. Where IBus is not available it falls back to wtype, ydotool or xdotool. See <a href=\"/whisper-dictation-wayland/\">Whisper dictation on Wayland</a>."),
        ("Is there a free dictation app for Linux?",
         "voiceio is free and MIT licensed. Other free options include <a href=\"/vs/voxtype/\">Voxtype</a> (MIT) and <a href=\"/vs/nerd-dictation/\">nerd-dictation</a> (GPL-3.0)."),
        ("Does it need an internet connection?",
         "Only to download the speech model the first time. After that, speech is decoded on your own computer and nothing is uploaded. See <a href=\"/offline-speech-to-text/\">offline speech to text</a>."),
        ("Which Linux distributions are supported?",
         "voiceio is a Python package (3.11+) installed with pipx. The README has package commands for Debian/Ubuntu, Fedora and Arch, and a flake for NixOS (not yet built, per the README)."),
        ("Can I dictate in languages other than English?",
         "The default Whisper models are multilingual; set <code>model.language</code> to a language code or <code>auto</code>. Fine-tuning and the vocabulary tools have been tested mostly on English."),
    ]
    body = f"""<header class="head">
  <span class="label">// guide · linux</span>
  <h1>Voice dictation on Linux</h1>
  <p class="lede">To dictate on Linux, install a speech-to-text tool that types into the focused window. voiceio does this on Wayland and X11: press a hotkey, speak, and the text streams into whatever app has focus, with a live underlined preview through IBus. Speech is decoded on your own computer with Whisper, so it works offline and nothing is uploaded. It is free and MIT licensed.</p>
  <div class="meta"><span class="chip on">Wayland + X11</span><span class="chip on">runs locally</span><span class="chip on">MIT</span><span class="chip">Python 3.11+</span></div>
</header>

<div class="prose">
<section class="part" aria-labelledby="how">
  <span class="label">// how to</span>
  <h2 id="how">Set up dictation in four steps</h2>
  {INSTALL_STEPS}
  <p class="muted">Prefer to hand it off? The <a href="/#install">install section</a> has a prompt for Claude Code, Codex or any coding agent, which follows the runbook in <a href="/llms.txt">llms.txt</a>.</p>
</section>

<section class="part" aria-labelledby="desktops">
  <span class="label">// desktops</span>
  <h2 id="desktops">What works where</h2>
  <div class="table-scroll"><table class="cmp">
    <thead><tr><th scope="col">Desktop</th><th scope="col">Text injection</th><th scope="col">Live preview</th><th scope="col">Status (README)</th></tr></thead>
    <tbody>
      <tr><th scope="row">GNOME (Wayland)</th><td>IBus, ydotool fallback</td><td>Yes</td><td>Tested daily</td></tr>
      <tr><th scope="row">GNOME (X11)</th><td>IBus, xdotool fallback</td><td>Yes</td><td>Supported</td></tr>
      <tr><th scope="row">KDE Plasma (Wayland)</th><td>IBus, ydotool fallback</td><td>Yes, with IBus</td><td>Should work</td></tr>
      <tr><th scope="row">Hyprland, sway</th><td>IBus, else wtype, then ydotool</td><td>Yes, with IBus</td><td>Should work</td></tr>
      <tr><th scope="row">i3, other X11</th><td>IBus, xdotool</td><td>Yes, with IBus</td><td>Should work</td></tr>
    </tbody>
  </table></div>
  <p class="note">From the <a href="https://github.com/Hugo0/voiceio/blob/main/docs/linux.md">voiceio README</a>. If fcitx5 is your input method, voiceio leaves it alone and types the final text with wtype or ydotool.</p>
</section>

<section class="part" aria-labelledby="names">
  <span class="label">// accuracy</span>
  <h2 id="names">Getting names and jargon right</h2>
  <p>Most dictation errors are proper nouns: people, products, your stack. voiceio has three tools for them, from cheapest to strongest:</p>
  <ul class="ticks">
    <li><b>Vocabulary.</b> <code>voiceio vocab add Kalshi</code> passes your terms to Whisper as hotwords, ranked by how often you use them. Whisper's hotword channel fits about 35 terms per decode.</li>
    <li><b>Corrections.</b> Find-and-replace rules for words that still come out wrong, or say "correct that" right after a mistake.</li>
    <li><b>Fine-tuning.</b> <code>voiceio learn</code> trains a LoRA fine-tune on your own kept recordings, on CPU, and switches to it only when it beats the current model on clips it never trained on. <code>voiceio learn schedule on</code> runs it when the machine is idle.</li>
  </ul>
</section>

<section class="part" aria-labelledby="others">
  <span class="label">// alternatives</span>
  <h2 id="others">Other ways to dictate on Linux</h2>
  {OTHER_TOOLS}
</section>

<section class="part" aria-labelledby="faq">
  <span class="label">// faq</span>
  <h2 id="faq">Questions</h2>
  {faq_html(faq)}
</section>
</div>
{cta()}"""
    write("/linux-voice-dictation/", page(
        path="/linux-voice-dictation/", og="linux-voice-dictation",
        title="Voice dictation on Linux (Wayland and X11) | voiceio",
        desc="How to dictate into any Linux app on Wayland or X11 with voiceio: local Whisper speech to text, live preview through IBus, free and open source.",
        crumbs=[("voiceio", "/"), ("Voice dictation on Linux", "/linux-voice-dictation/")],
        body=body, faq=faq, app_ld=True))


def offline():
    faq = [
        ("Can speech to text work without internet?",
         "Yes. Models such as Whisper run on an ordinary CPU. voiceio downloads the model once, from its own source, and then decodes everything on your computer."),
        ("Is offline dictation as accurate as cloud dictation?",
         "It depends on the model and your speech. Larger local models (medium, large-v3-turbo) are more accurate but slower without a GPU. voiceio's answer for names and jargon is to fine-tune the local model on your own voice; <code>voiceio learn eval</code> measures the result on your clips."),
        ("What does voiceio send over the network?",
         "Nothing by default: no account, no telemetry, and audio never leaves the machine. Two optional, off-by-default features send text (never audio) to an LLM endpoint you configure."),
        ("Do I need a GPU?",
         "No. The default <code>small</code> model runs at about 5x realtime on a laptop CPU, per the README, and fine-tuning runs on CPU too. A GPU makes <code>large-v3-turbo</code> practical."),
    ]
    body = f"""<header class="head">
  <span class="label">// guide · offline</span>
  <h1>Offline speech to text</h1>
  <p class="lede">Offline speech to text means the speech model runs on your own computer, so audio never goes to a server and dictation works without a connection. On Linux, voiceio does this with Whisper: after a one-time model download it decodes your speech locally, types the text into the focused app, and has no account and no telemetry.</p>
  <div class="meta"><span class="chip on">no upload</span><span class="chip on">no account</span><span class="chip on">no telemetry</span><span class="chip">CPU is enough</span></div>
</header>

<div class="prose">
<section class="part" aria-labelledby="how">
  <span class="label">// how to</span>
  <h2 id="how">Set up offline dictation</h2>
  {INSTALL_STEPS}
  <p>Once the model is downloaded, unplug the network and dictate: nothing changes. To pick a different local model:</p>
<pre><code><span class="p">$</span> voiceio models list          <span class="c"># known models and which is active</span>
<span class="p">$</span> voiceio models use medium    <span class="c"># better on names, about 2.5x slower</span></code></pre>
</section>

<section class="part" aria-labelledby="models">
  <span class="label">// models</span>
  <h2 id="models">Local models voiceio can use</h2>
  <div class="table-scroll"><table class="cmp">
    <thead><tr><th scope="col">Model</th><th scope="col">Runtime</th><th scope="col">Notes (README)</th></tr></thead>
    <tbody>
      <tr><th scope="row">small</th><td>faster-whisper</td><td>Default. About 5x realtime on a laptop CPU</td></tr>
      <tr><th scope="row">medium</th><td>faster-whisper</td><td>Better on names, about 2.5x slower</td></tr>
      <tr><th scope="row">large-v3-turbo</th><td>faster-whisper</td><td>Most accurate Whisper option for dictation; wants a GPU</td></tr>
      <tr><th scope="row">distil-large-v3</th><td>faster-whisper</td><td>English only</td></tr>
      <tr><th scope="row">Parakeet TDT 0.6b v3</th><td>sherpa-onnx</td><td>Experimental. Fast, no vocabulary biasing yet</td></tr>
      <tr><th scope="row">whisper.cpp server</th><td>whisper.cpp</td><td>Experimental. E.g. a Vulkan build on an AMD iGPU</td></tr>
    </tbody>
  </table></div>
  <p class="note">voiceio ships no model weights; each model downloads from its own source under its own license.</p>
</section>

<section class="part" aria-labelledby="data">
  <span class="label">// your data</span>
  <h2 id="data">What stays on your disk</h2>
  <p>Everything voiceio keeps is plain files in <code>~/.config/voiceio</code>, <code>~/.local/state/voiceio</code> and <code>~/.local/share/voiceio</code>, created readable only by you. Recordings are off unless you turn them on; you need them only for fine-tuning. Logs record sizes and timings, never your words.</p>
  <p>For a phone or another device, <code>python -m voiceio.server</code> serves the same engine over HTTP and WebSocket on a machine you control. It binds to localhost by default.</p>
</section>

<section class="part" aria-labelledby="others">
  <span class="label">// alternatives</span>
  <h2 id="others">Other offline options</h2>
  {OTHER_TOOLS}
</section>

<section class="part" aria-labelledby="faq">
  <span class="label">// faq</span>
  <h2 id="faq">Questions</h2>
  {faq_html(faq)}
</section>
</div>
{cta()}"""
    write("/offline-speech-to-text/", page(
        path="/offline-speech-to-text/", og="offline-speech-to-text",
        title="Offline speech to text on Linux, no cloud | voiceio",
        desc="Dictate offline on Linux: voiceio runs Whisper on your own CPU, types into any app, uploads nothing and has no telemetry. Setup steps and model choices.",
        crumbs=[("voiceio", "/"), ("Offline speech to text", "/offline-speech-to-text/")],
        body=body, faq=faq, app_ld=True))


def wayland():
    faq = [
        ("Why is dictation hard on Wayland?",
         "Wayland does not let one app type into another or read global hotkeys the way X11 did. Tools work around it with input methods (IBus, fcitx5), virtual keyboards (wtype, ydotool, dotool) or the compositor's own key bindings."),
        ("Why IBus instead of ydotool or wtype?",
         "An input method can show an underlined preview that is corrected in place and then committed. Keystroke tools can only type and backspace, which is slower and can drop characters on rapid corrections. voiceio uses IBus first and falls back to wtype or ydotool."),
        ("Does it work with fcitx5 (Omarchy)?",
         "Yes. If fcitx5 is your input method, voiceio leaves it alone and types the words with wtype or ydotool as you speak, without the in-place preview."),
        ("How do I bind the hotkey on Hyprland or sway?",
         "Use the built-in evdev hotkey, or bind <code>voiceio toggle</code> in your compositor, e.g. <code>bind = CTRL ALT, V, exec, voiceio toggle</code> on Hyprland."),
    ]
    body = f"""<header class="head">
  <span class="label">// guide · wayland</span>
  <h1>Whisper dictation on Wayland</h1>
  <p class="lede">To dictate with Whisper on Wayland, you need two pieces: a local Whisper decoder, and a way to put text into other apps that Wayland allows. voiceio uses faster-whisper for the first and an IBus input method for the second, so words appear in the focused app as an underlined live preview while you speak and are committed when you stop. It works on GNOME, KDE Plasma, Hyprland and sway, and on X11.</p>
  <div class="meta"><span class="chip on">IBus preedit</span><span class="chip on">faster-whisper</span><span class="chip">wtype · ydotool fallback</span></div>
</header>

<div class="prose">
<section class="part" aria-labelledby="how">
  <span class="label">// how to</span>
  <h2 id="how">Set it up</h2>
  {INSTALL_STEPS}
  <h3>Hotkeys on Wayland</h3>
  <p>The evdev backend reads the keyboard directly. <code>voiceio doctor --fix</code> installs a udev rule that gives the logged-in user access, with no logout needed. On GNOME, setup can add a shortcut instead. On Hyprland or sway, bind <code>voiceio toggle</code>:</p>
<pre><code><span class="c"># ~/.config/hypr/hyprland.conf</span>
bind = CTRL ALT, V, exec, voiceio toggle
<span class="c"># ~/.config/sway/config</span>
bindsym Ctrl+Alt+v exec voiceio toggle</code></pre>
</section>

<section class="part" aria-labelledby="inject">
  <span class="label">// typing</span>
  <h2 id="inject">How the text gets in</h2>
  <ul class="ticks">
    <li><b>IBus (preferred).</b> voiceio installs its own IBus engine and activates it only while you dictate. The preview is shown as preedit text and committed at the end.</li>
    <li><b>fcitx5.</b> Left alone; final text is typed with wtype or ydotool.</li>
    <li><b>wtype, ydotool, xdotool.</b> Keystroke fallbacks, picked by probing what works. <code>doctor --fix</code> installs a uaccess rule for <code>/dev/uinput</code> so ydotool works.</li>
    <li><b>Clipboard.</b> For terminals that ignore input methods.</li>
  </ul>
</section>

<section class="part" aria-labelledby="whisper">
  <span class="label">// the model</span>
  <h2 id="whisper">Which Whisper</h2>
  <p>The default is Whisper <code>small</code> through faster-whisper, about 5x realtime on a laptop CPU. <code>medium</code> is better on names and slower; <code>large-v3-turbo</code> wants a GPU. Long dictations are finalized in pieces while you talk, so stopping a multi-minute note does not mean waiting for the whole recording to decode again. Read how voiceio relates to the model itself in <a href="/vs/whisper/">voiceio and Whisper</a>.</p>
</section>

<section class="part" aria-labelledby="others">
  <span class="label">// alternatives</span>
  <h2 id="others">Other Whisper tools on Wayland</h2>
  <p><a href="/vs/voxtype/">Voxtype</a> also runs Whisper (via whisper.cpp) and other engines on Wayland, typing with wtype, dotool or ydotool. <a href="/vs/nerd-dictation/">nerd-dictation</a> uses VOSK rather than Whisper. <a href="/vs/talon/">Talon</a> does not support Wayland.</p>
</section>

<section class="part" aria-labelledby="faq">
  <span class="label">// faq</span>
  <h2 id="faq">Questions</h2>
  {faq_html(faq)}
</section>
</div>
{cta()}"""
    write("/whisper-dictation-wayland/", page(
        path="/whisper-dictation-wayland/", og="whisper-dictation-wayland",
        title="Whisper dictation on Wayland with live preview | voiceio",
        desc="Dictate with Whisper on Wayland: voiceio types into any app through IBus with a live preview, on GNOME, KDE, Hyprland and sway. Local and open source.",
        crumbs=[("voiceio", "/"), ("Whisper dictation on Wayland", "/whisper-dictation-wayland/")],
        body=body, faq=faq, app_ld=True))


if __name__ == "__main__":
    linux(); offline(); wayland()
