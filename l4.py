from pathlib import Path
import time
import modal

app = modal.App(
    "stable-diffusion-xl-lightning-l4"
)  # Note: prior to April 2024, "app" was called "stub"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "diffusers==0.26.3", "transformers~=4.37.2", "accelerate==0.27.2"
)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"


with image.imports():
    import io
    import statistics

    import torch
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from fastapi import Response
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file


@app.cls(image=image, gpu="L4")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            "cuda", torch.float16
        )
        unet.load_state_dict(
            load_file(hf_hub_download(repo, ckpt), device="cuda")
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

    def _inference(self, prompt, n_steps=7):
        negative_prompt = "disfigured, ugly, deformed"
        start = time.monotonic_ns()
        image = self.pipe(
            prompt=prompt,
            guidance_scale=0,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
        ).images[0]

        duration_s = (time.monotonic_ns() - start) / 1e9
        return duration_s

    @modal.method()
    def inference(self, prompt, n_steps=7):
        prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
        return self._inference(
            prompt,
            n_steps=n_steps,
        )

@app.local_entrypoint()
def main():
    samples = range(33)
    model_runs = map(Model().inference.remote, samples)
    results = list(model_runs)
    print(f'Average L4 inference time: {sum(results) / len(results)}')
    print(f'Min L4 inference time: {min(results)}')
    print(f'Max L4 inference time: {max(results)}')
    print("Raw results")
    print(results)
