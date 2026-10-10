"""Shared page shell for the voiceio.dev SEO pages (one-off generator; output is plain static HTML)."""
import html
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[2] / "site"
ORIGIN = "https://voiceio.dev"
CHECKED = "2026-10-10"

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='2' "
           "fill='%2384cc3a'/%3E%3Cpath d='M3 7h2v2H3zM6 4h2v8H6zM9 6h2v4H9zM12 7h1v2h-1z' fill='%23151820'/%3E%3C/svg%3E")

COMPARE = [  # (slug, label)
    ("voxtype", "Voxtype"), ("aqua-voice", "Aqua Voice"), ("wispr-flow", "Wispr Flow"),
    ("superwhisper", "Superwhisper"), ("dragon", "Dragon"), ("nerd-dictation", "nerd-dictation"),
    ("talon", "Talon"), ("whisper", "Whisper"),
]
GUIDES = [
    ("/linux-voice-dictation/", "Voice dictation on Linux"),
    ("/offline-speech-to-text/", "Offline speech to text"),
    ("/whisper-dictation-wayland/", "Whisper dictation on Wayland"),
]

esc = html.escape


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def header(current: str = "") -> str:
    cur = ' aria-current="page"' if current == "compare" else ""
    return f"""<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <div class="wrap">
    <a class="brand" href="/" aria-label="voiceio home">VOICE<span class="io">IO</span><span class="flick">_</span></a>
    <nav aria-label="Site">
      <a class="opt" href="/#how">How it works</a>
      <a href="/vs/"{cur}>Compare</a>
      <a class="opt" href="/#install">Install</a>
    </nav>
    <a class="gh" href="https://github.com/Hugo0/voiceio">GitHub <span aria-hidden="true">↗</span></a>
  </div>
</header>"""


def footer() -> str:
    links = "\n        ".join(f'<a href="/vs/{s}/">{"and" if s == "whisper" else "vs"} {esc(l)}</a>' for s, l in COMPARE)
    guides = "\n        ".join(f'<a href="{p}">{esc(l)}</a>' for p, l in GUIDES)
    return f"""<footer>
  <div class="wrap">
    <nav class="compare label" aria-label="Compare">
        <a href="/vs/">Compare</a>
        {links}
    </nav>
    <nav class="compare label" aria-label="Guides">
        {guides}
    </nav>
    <div class="row">
      <span class="brand" style="color:var(--faint);font-size:14px">VOICE<span class="io">IO</span></span>
      <span class="label">local · private · learns you · made with ♥ by <a href="https://hugo0.com">hugo0.com</a></span>
    </div>
  </div>
</footer>"""


def crumbs_html(crumbs) -> str:
    items = []
    for i, (name, path) in enumerate(crumbs):
        if i == len(crumbs) - 1:
            items.append(f'<li aria-current="page">{esc(name)}</li>')
        else:
            items.append(f'<li><a href="{path}">{esc(name)}</a></li>')
    return f'<nav class="crumbs label" aria-label="Breadcrumb"><ol>{"".join(items)}</ol></nav>'


def faq_html(faq) -> str:
    out = []
    for q, a in faq:
        out.append(f"<details>\n  <summary>{esc(q)}</summary>\n  <p>{a}</p>\n</details>")
    return '<div class="faq">\n' + "\n".join(out) + "\n</div>"


def page(*, path: str, og: str, title: str, desc: str, crumbs, body: str, faq=None, app_ld=False, current="") -> str:
    assert len(title) <= 60, (path, len(title), title)
    assert len(desc) <= 155, (path, len(desc), desc)
    url = ORIGIN + path
    img = f"{ORIGIN}/og/{og}.png"
    ld = [{
        "@context": "https://schema.org", "@type": "BreadcrumbList",
        "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, "item": ORIGIN + p}
                            for i, (n, p) in enumerate(crumbs)],
    }]
    if faq:
        ld.append({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": strip_tags(a)}} for q, a in faq]})
    if app_ld:
        ld.append({"@context": "https://schema.org", "@type": "SoftwareApplication", "name": "voiceio",
                   "description": "Local, open-source voice dictation for Linux that fine-tunes itself on your voice.",
                   "url": ORIGIN + "/", "applicationCategory": "UtilitiesApplication", "operatingSystem": "Linux",
                   "license": "https://opensource.org/licenses/MIT", "isAccessibleForFree": True,
                   "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
                   "downloadUrl": "https://pypi.org/project/python-voiceio/",
                   "sameAs": ["https://github.com/Hugo0/voiceio", "https://pypi.org/project/python-voiceio/"]})
    ld_html = "\n".join(f'<script type="application/ld+json">\n{json.dumps(x, ensure_ascii=False)}\n</script>' for x in ld)
    t, d = esc(title), esc(desc)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{t}</title>
<meta name="description" content="{d}">
<meta name="color-scheme" content="light">
<meta name="referrer" content="no-referrer">
<meta name="theme-color" content="#f5f6f8">
<link rel="canonical" href="{url}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="voiceio">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{t}">
<meta property="og:description" content="{d}">
<meta property="og:image" content="{img}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{t}">
<meta name="twitter:description" content="{d}">
<meta name="twitter:image" content="{img}">
{ld_html}
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Spline+Sans+Mono:wght@400;500;600&family=Tiny5&display=swap">
<link rel="stylesheet" href="/assets/pages.css">
</head>
<body>
{header(current)}

<main id="main" class="wrap">
{crumbs_html(crumbs)}
{body}
</main>

{footer()}
</body>
</html>
"""


def cta(heading="Try voiceio", text=None) -> str:
    text = text or ("Free and MIT licensed. Install it with pipx on Linux, or try the dictation in your browser first: "
                    "the demo on the home page runs a small speech model inside the tab.")
    return f"""<section class="band" aria-labelledby="cta-h">
  <span class="label">// get it</span>
  <h2 id="cta-h" style="margin-top:10px">{heading}</h2>
  <p>{text}</p>
  <div class="cta">
    <a class="pill solid" href="/#install"><span class="d" aria-hidden="true">◇</span>Install voiceio</a>
    <a class="pill" href="/"><span class="d" aria-hidden="true">▸</span>Try it in your browser</a>
    <a class="pill" href="/vs/"><span class="d" aria-hidden="true">≡</span>All comparisons</a>
  </div>
</section>"""


def write(path: str, content: str):
    out = SITE / path.strip("/") / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print("wrote", out.relative_to(SITE))
