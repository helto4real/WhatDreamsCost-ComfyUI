from .ltx_keyframer import LTXKeyframer
from .multi_image_loader import MultiImageLoader
from .ltx_sequencer import LTXSequencer
from .speech_length_calculator import SpeechLengthCalculator
from .load_audio_ui import LoadAudioUI
from .load_video_ui import LoadVideoUI
from .ltx_director import LTXDirector
from .ltx_director_guide import LTXDirectorGuide
from .ltx_identity_anchor import (
    LTXDirectorApplyIdentityAnchor,
    LTXIdentityAnchorCombine,
    LTXIdentityAnchorFace,
    LTXIdentityAnchorLatentAware,
)
from . import timeline_image_routes  # noqa: F401
from . import timeline_audio_routes  # noqa: F401
from . import ltx_director_privacy_routes  # noqa: F401
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXDirector,
            LTXDirectorGuide,
            LTXIdentityAnchorLatentAware,
            LTXIdentityAnchorFace,
            LTXIdentityAnchorCombine,
            LTXDirectorApplyIdentityAnchor,
        ]

async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()
    
NODE_CLASS_MAPPINGS = {
    "LTXKeyframer": LTXKeyframer,
    "MultiImageLoader": MultiImageLoader,
    "LTXSequencer": LTXSequencer,
    "SpeechLengthCalculator": SpeechLengthCalculator,
    "LoadAudioUI": LoadAudioUI,
    "LoadVideoUI": LoadVideoUI,
    "LTXDirector": LTXDirector,
    "LTXDirectorGuide": LTXDirectorGuide,
    "LTXIdentityAnchorLatentAware": LTXIdentityAnchorLatentAware,
    "LTXIdentityAnchorFace": LTXIdentityAnchorFace,
    "LTXIdentityAnchorCombine": LTXIdentityAnchorCombine,
    "LTXDirectorApplyIdentityAnchor": LTXDirectorApplyIdentityAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXKeyframer": "LTX Keyframer",
    "MultiImageLoader": "Multi Image Loader",
    "LTXSequencer": "LTX Sequencer",
    "SpeechLengthCalculator": "Speech Length Calculator",
    "LoadAudioUI": "Load Audio UI",
    "LoadVideoUI": "Load Video UI",
    "LTXDirector": "LTX Director",
    "LTXDirectorGuide": "LTX Director Guide",
    "LTXIdentityAnchorLatentAware": "LTX Identity Anchor: Latent Aware",
    "LTXIdentityAnchorFace": "LTX Identity Anchor: Face",
    "LTXIdentityAnchorCombine": "LTX Identity Anchor: Combine",
    "LTXDirectorApplyIdentityAnchor": "LTX Director Apply Identity Anchor",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
