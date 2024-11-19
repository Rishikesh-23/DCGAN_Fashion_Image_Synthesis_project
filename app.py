
import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image

# Generator class (same as defined earlier)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Function to generate images
def generate_images(generator, num_images=10, latent_dim=100):
    z = torch.randn(num_images, latent_dim)
    fake_images = generator(z)
    fake_images = (fake_images + 1) / 2  # Scale to [0, 1]
    return fake_images

# Streamlit Interface
st.title("Fashion Image Generator using DCGAN")
st.write("Generate new fashion items using a trained DCGAN model on the Fashion MNIST dataset.")

# User controls for image generation
num_images = st.slider("Number of Images to Generate", 1, 20, 10)
latent_dim = st.slider("Latent Vector Dimension", 10, 200, 100)

# Initialize and load the Generator model
generator = Generator(latent_dim=latent_dim)
# You can load pre-trained weights if available
# generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))

if st.button("Generate Images"):
    with st.spinner("Generating images..."):
        fake_images = generate_images(generator, num_images, latent_dim)

        # Display images in a grid
        grid = make_grid(fake_images, nrow=5, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).detach().numpy(), cmap="gray")
        plt.axis("off")
        
        # Convert to PIL image to display in Streamlit
        grid_image = Image.fromarray((grid.permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
        st.image(grid_image, caption="Generated Fashion Items", use_column_width=True)
