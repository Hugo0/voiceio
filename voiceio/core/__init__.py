"""Display-free dictation core: decoders, biasing, the text pipeline.

Everything here runs headless — the desktop daemon, the server, evaluation and
the learning loop all compose it. Nothing in `voiceio.core` may import audio
devices, hotkeys, typers, IBus or the tray.
"""
