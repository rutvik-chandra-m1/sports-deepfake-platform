"""
Generates wholly synthetic ("fake" class) sports-scene images via a local
diffusion model, for the R2 sports-specific supplement.

Deliberately does NOT name or reference any real, identifiable athlete,
team, or event in any prompt -- every image depicts a fictional person in a
generic sporting context. This is a wholly-AI-generated-image detection
dataset (matching the PPT's actual framing: "Detection of AI-Generated
Sportsman Images"), not a deepfake/face-swap dataset of real people.

Model: segmind/tiny-sd -- a distilled, architecturally-smaller SD1.5
variant (not just fp16 -- fewer UNet blocks), chosen after checking actual
download sizes: stabilityai/sd-turbo's diffusers-format weights are
~2.5GB (fp16) to ~4.9GB (fp32); tiny-sd is ~1GB total. On this machine's
measured ~120-190kB/s connection, that's the difference between roughly
2 hours and 5-9 hours for the download alone, before any generation.
"Up to 80% faster than base SD1.5" per the model card also helps CPU
inference time (this machine has no GPU -- Intel i7-8665U, CPU-only).

Not a turbo/consistency model, so it needs normal multi-step sampling
with classifier-free guidance (steps=20, guidance_scale=7.5 -- standard
SD1.5 defaults; the model card doesn't recommend different values).

Run with --n 1 first to measure real per-image time on your machine
before committing to a full batch; see the printed timing.

Usage:
    python generate_synthetic.py --n 60 --out ../../datasets/sports_fake
"""

import argparse
import csv
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "segmind/tiny-sd"

# Varied sport + framing combinations for stratification (roadmap R2 calls
# for variety across sport and framing: close-up / mid / wide). No real
# person, team, league, or venue named anywhere.
PROMPTS = [
    # close-up
    "close-up portrait of a determined soccer player's face, sweat, stadium lights, photorealistic sports photography",
    "close-up of a boxer's gloved fists, intense expression, gym background, photorealistic",
    "close-up portrait of a swimmer adjusting goggles poolside, photorealistic",
    "close-up of a basketball player's hands gripping a basketball, photorealistic sports photography",
    "close-up portrait of a tennis player mid-serve, focused expression, photorealistic",
    "close-up of a runner's shoes on a track starting block, photorealistic",
    # mid action shot
    "a soccer player kicking a ball on a grass field, dynamic action shot, photorealistic sports photography",
    "a basketball player dunking during a game, arena lights, photorealistic action photo",
    "a tennis player serving on a clay court, photorealistic action photo",
    "a swimmer mid-stroke in an outdoor pool, splashing water, photorealistic sports photography",
    "a sprinter running on a track, motion blur, photorealistic action shot",
    "an American football quarterback throwing a pass, photorealistic action shot",
    "a volleyball player spiking the ball at the net, photorealistic sports photography",
    "a gymnast performing a balance beam routine, photorealistic action photo",
    "a cyclist racing on a mountain road, photorealistic action photo",
    "a baseball pitcher winding up to throw, photorealistic sports photography",
    # wide stadium / scene
    "a wide shot of a football stadium during a match, players on the field, photorealistic",
    "a wide shot of a crowded basketball arena during a game, photorealistic sports photography",
    "a wide shot of runners competing on an athletics track, stadium crowd in background, photorealistic",
    "a wide shot of a cricket match on a green pitch, photorealistic sports photography",
    "a wide shot of a swimming competition, multiple lanes, photorealistic",
    "a wide shot of a rugby match on a muddy field, photorealistic action photo",
]


def build_pipeline():
    import torch

    # Deliberately StableDiffusionPipeline directly, not AutoPipelineForText2Image:
    # the Auto* wrapper eagerly imports diffusers' entire pipeline registry (to map
    # arbitrary model configs to a pipeline class), including unrelated pipelines
    # like HunyuanDiT -- whose import chain is broken under transformers==5.14.1
    # (references transformers.MT5Tokenizer, removed in transformers 5.x). We
    # already know segmind/tiny-sd is a standard SD1.5-architecture checkpoint,
    # so there's nothing to auto-detect.
    from diffusers import StableDiffusionPipeline

    logger.info("Loading %s (CPU inference -- this is the slow part)...", MODEL_ID)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.to("cpu")
    return pipe


def generate(pipe, prompt: str, steps: int, seed: int, guidance_scale: float):
    import torch

    generator = torch.Generator("cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="total images to generate")
    parser.add_argument("--steps", type=int, default=20, help="inference steps (standard SD1.5 default)")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="classifier-free guidance scale")
    parser.add_argument("--out", type=str, default="../../datasets/sports_fake")
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline()

    manifest_rows = []
    generated = 0
    total_elapsed = 0.0
    for i in range(args.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = args.seed_start + i
        start = time.monotonic()
        image = generate(pipe, prompt, args.steps, seed, args.guidance_scale)
        elapsed = time.monotonic() - start
        total_elapsed += elapsed

        filename = f"synthetic_{seed}.png"
        image.save(out_dir / filename)
        manifest_rows.append(
            {
                "filename": filename,
                "source": "local-diffusion",
                "model_id": MODEL_ID,
                "prompt": prompt,
                "seed": seed,
                "steps": args.steps,
                "generation_seconds": round(elapsed, 2),
            }
        )
        generated += 1
        avg = total_elapsed / generated
        remaining = (args.n - generated) * avg
        logger.info(
            "[%d/%d] %s (%.1fs, avg %.1fs/image, ~%.0fs remaining)",
            generated, args.n, filename, elapsed, avg, remaining,
        )

    manifest_path = out_dir / "generation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "source", "model_id", "prompt", "seed", "steps", "generation_seconds"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("Done: %d synthetic images in %s (manifest: %s)", generated, out_dir, manifest_path)


if __name__ == "__main__":
    main()
