import torch
from models.generator import Generator
import torchvision.utils as vutils

def generate_images(model_path, num_images, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator model
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    # Generate images
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, Z_DIM, 1, 1).to(device)
            fake_image = gen(noise)
            vutils.save_image(fake_image, f"{output_dir}/generated_image_{i}.png", normalize=True)
