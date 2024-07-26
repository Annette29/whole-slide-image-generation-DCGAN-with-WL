import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from preprocess_training_data import Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.initialize import initialize_weights
from models.gradient_penalty import gradient_penalty
from utils.visualization import plot_losses
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.training import train_critic, train_generator
from generate_images import generate_images

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 6000
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# Dataset
train_dataset = Dataset(img_dir="data/train", transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize generator and critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

initialize_weights(gen)
initialize_weights(critic)

# Optimizers
opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# Load checkpoint if available
if LOAD_MODEL:
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    load_checkpoint(CHECKPOINT_CRITIC, critic, opt_critic, LEARNING_RATE)

gen.train()
critic.train()

# For plotting loss curves
gen_losses = []
critic_losses = []

# Tensorboard setup
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(train_loader):
        real = real.to(device)
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            loss_critic = train_critic(critic, gen, real, fake, opt_critic, LAMBDA_GP, gradient_penalty)
        
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        loss_gen = train_generator(critic, gen, fake, opt_gen)

        # Save the losses for plotting
        critic_losses.append(loss_critic.item())
        gen_losses.append(loss_gen.item())

        # Print losses occasionally and print to tensorboard
        if batch_idx % 400 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

    # Save checkpoint
    if SAVE_MODEL and epoch % 600 == 0:
        save_checkpoint(gen, opt_gen, filename=f"generator_{epoch}.pth.tar")
        save_checkpoint(critic, opt_critic, filename=f"critic_{epoch}.pth.tar")


# Plot the generator and critic losses
plot_losses(critic_losses, gen_losses, NUM_EPOCHS, START_EPOCH, SAVE_PATH)
