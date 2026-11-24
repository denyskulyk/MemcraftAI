"""MemcraftAI's own ComfyUI nodes (C:\Source\MemcraftAI). Added 2026-11-24 with Denys's OK.

LTXV Dual CFG Guider (audio until): the core LTXV Dual CFG Guider, but the audio CFG only works
while the noise level (sigma) is above `audio_until_sigma`. Below it, video and audio CFG are both
1, so the extra unconditional pass is skipped and those steps cost half. One sampler keeps the
first-frame and keyframe-guide masks intact (a split into two samplers rescales them 40x).
"""

from comfy_extras.nodes_lt import Guider_LTXAVDualCFG

version = "1.0.0"


class GuiderLTXAVDualCFGUntil(Guider_LTXAVDualCFG):
    def set_until(self, until):
        self.until = float(until)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        sigma = float(timestep.max())
        if sigma > self.until + 1e-4:
            self.cfg = max(self.video_cfg, self.audio_cfg)
            return super().predict_noise(x, timestep, model_options, seed)
        # Plain CFG at the video scale (1 in the app): the cfg-1 shortcut skips the uncond pass.
        self.cfg = self.video_cfg
        return super(Guider_LTXAVDualCFG, self).predict_noise(x, timestep, model_options, seed)


class LTXVDualCFGGuiderUntil:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "audio_cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "audio_until_sigma": (
                    "FLOAT",
                    {"default": 0.975, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg, audio_until_sigma):
        guider = GuiderLTXAVDualCFGUntil(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(video_cfg, audio_cfg)
        guider.set_until(audio_until_sigma)
        return (guider,)


NODE_CLASS_MAPPINGS = {"MemcraftLTXVDualCFGGuiderUntil": LTXVDualCFGGuiderUntil}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MemcraftLTXVDualCFGGuiderUntil": "LTXV Dual CFG Guider (audio until sigma)"
}
