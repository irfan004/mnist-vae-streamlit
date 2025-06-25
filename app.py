import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# VAE Model (must match your trained architecture)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20)  # 10 for mean, 10 for log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h[:, :10], h[:, 10:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Load the trained model
device = torch.device('cpu')  # Local machine (no GPU needed)
model = VAE().to(device)
model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
model.eval()

# Streamlit app
st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0-9)", list(range(10)))

if st.button("Generate 5 Images"):
    with torch.no_grad():
        # Generate 5 random latent vectors
        z = torch.randn(5, 10).to(device)
        generated_images = model.decoder(z).view(-1, 1, 28, 28)
    # Display images
    st.write(f"Generated 5 images for digit {digit}:")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)