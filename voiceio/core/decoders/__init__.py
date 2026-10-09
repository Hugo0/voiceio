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
    """The configured decoder: one faster-whisper worker, or two (a draft lane
    for interim previews and a final lane) when `lanes` >= 2."""
    from voiceio.transcriber import Transcriber

    lanes = model_cfg.decode_lanes if lanes is None else lanes
    final = Transcriber(model_cfg)
    if lanes < 2:
        return final
    from voiceio.core.decoders.lanes import DualLane
    return DualLane(draft=Transcriber(model_cfg), final=final)
