import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model ID and prompt
model_id = "stabilityai/stable-diffusion-2-base"
prompt = "A serene landscape with mountains and a lake at sunset"

# Load the tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

# Load the VAE and UNet models
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

# Inference parameters
num_inference_steps = 50
height = 512
width = 512
batch_size = 1
num_variations = 3
base_seed = 42

# Tokenize and encode the prompt
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# Function to convert latent to PIL
def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().cpu()
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_array = (image_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    return image

for i in range(num_variations):
    # Re-initialize the scheduler and set timesteps for each variation
    scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    
    # Create a new initial latent vector for each variation
    seed = base_seed #+ i
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
    )
    latents = latents * scheduler.init_noise_sigma

    # Define at which step to start adding random walk (e.g., after half of the steps)
    start_random_walk_step = 5 #num_inference_steps // 2
    random_walk_scale = 0.05 #0.05  # Adjust this for more or less variability

    # Denoising loop
    for step_index, t in enumerate(tqdm(scheduler.timesteps, desc=f"Generating variation {i+1}/{num_variations}")):
        # Scale the model input according to the scheduler
        latent_model_input = scheduler.scale_model_input(latents, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform the standard scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # After a certain number of steps, start adding random noise
        if step_index > start_random_walk_step:
            # Add a small random perturbation to latents
            random_noise = torch.randn_like(latents) * random_walk_scale
            latents = latents + random_noise

    # Decode the latents
    with torch.no_grad():
        scaled_latents = latents / 0.18215
        images = vae.decode(scaled_latents).sample

    # Convert to PIL and save
    img = tensor_to_pil(images)
    img.save(f"variation_{i+1}.png")
    print(f"Saved variation_{i+1}.png")
