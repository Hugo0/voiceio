"""Speech decoders behind one protocol (see `base`)."""
from voiceio.core.decoders.base import (
    Bias,
    BiasMode,
    Capabilities,
    DecodeError,
    DecodeResult,
    Decoder,
)

__all__ = [
    "Bias", "BiasMode", "Capabilities", "DecodeError", "DecodeResult", "Decoder",
    "build_decoder",
]


def build_decoder(model_cfg, lanes: int | None = None) -> Decoder:
    """The decoder for `[model] name`:

    - ``whispercpp:<url>``  a whisper.cpp server (e.g. Vulkan on an AMD iGPU)
    - ``parakeet-tdt-0.6b-v2`` and other sherpa-onnx catalog names: downloaded
                            on first use from the sherpa-onnx GitHub release
    - ``sherpa:<dir>``      a sherpa-onnx transducer (dir, or a name under
                            the models dir); `[model] decoding` overrides
                            the method (modified_beam_search where the
                            installed sherpa-onnx has the TDT fix)
    - anything else         faster-whisper (a size name, HF repo, or path to a
                            CTranslate2 model such as a `voiceio learn` run);
                            two worker processes when `lanes` >= 2
    """
    name = model_cfg.name
    if name.startswith("whispercpp:"):
        from voiceio.core.decoders.whispercpp import WhisperCppServer
        return WhisperCppServer(name.split(":", 1)[1], language=model_cfg.language)
    from voiceio.core.decoders.catalog import runtime_of
    if runtime_of(name) == "sherpa-onnx":
        return _sherpa(model_cfg)

    from voiceio.core.decoders.faster_whisper import Transcriber
    lanes = model_cfg.decode_lanes if lanes is None else lanes
    if not model_cfg.cpu_threads:
        import dataclasses

        from voiceio.core.decoders.faster_whisper.worker import auto_cpu_threads
        model_cfg = dataclasses.replace(model_cfg, cpu_threads=auto_cpu_threads(lanes))
    final = Transcriber(model_cfg)
    if lanes < 2:
        return final
    try:
        draft = Transcriber(model_cfg)
    except Exception:  # noqa: BLE001 — e.g. no memory for a second copy
        import logging
        logging.getLogger(__name__).warning(
            "Could not start a second decoder process; using one lane", exc_info=True)
        return final
    from voiceio.core.decoders.lanes import DualLane
    return DualLane(draft=draft, final=final)


def _sherpa(model_cfg) -> Decoder:
    from pathlib import Path

    from voiceio.core.decoders import fetch
    from voiceio.core.decoders.sherpa import SherpaTransducer, default_threads, features
    if not features().installed:   # before a 500 MB download it couldn't use
        raise DecodeError("sherpa-onnx not installed: pip install 'python-voiceio[sherpa]' "
                          "(or the build with the Parakeet fixes: voiceio models sherpa)")
    name = model_cfg.name
    if name.startswith("sherpa:"):
        model_dir = Path(name.split(":", 1)[1]).expanduser()
        if not model_dir.is_absolute():
            model_dir = fetch.models_dir() / model_dir
        label = model_dir.name
    else:  # a catalog model: fetched on first use, like Whisper's
        try:
            model_dir = fetch.download(name)
        except OSError as e:
            raise DecodeError(f"could not get model {name}: {e}") from e
        label = name
    return SherpaTransducer(model_dir, threads=model_cfg.cpu_threads or default_threads(),
                            method=model_cfg.decoding, label=label)
