from diffusers import StableDiffusionPipeline
import torch

# Use Stable Diffusion 2
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" #if torch.cuda.is_available() else "cpu"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",  # Ensure you're loading the float16 weights
)
pipe = pipe.to(device)

prompt = "A serene landscape with mountains and a lake at sunset"
initial_image = pipe(prompt).images[0]

# Save or display the image
initial_image.save("initial_image.png")
