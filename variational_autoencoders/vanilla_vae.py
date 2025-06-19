"""
Vanilla Variational Autoencoder (VAE) Implementation
===================================================

Standard VAE with Gaussian prior and posterior distributions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class VanillaVAE(nn.Module):
    """
    Vanilla Variational Autoencoder
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): Hidden layer dimensions for encoder/decoder
        latent_dim (int): Latent space dimension
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super(VanillaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        dims_reversed = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(dims_reversed[i], dims_reversed[i+1]),
                nn.ReLU(inplace=True) if i < len(dims_reversed) - 2 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Complete forward pass
        
        Returns:
            recon_x: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Generate samples from the model
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples
        """
        with torch.no_grad():
            # Sample from prior distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Decode to get samples
            samples = self.decode(z)
            
        return samples


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction and KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


class VAETrainer:
    """
    Training class for VAE
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3, beta=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.beta = beta
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            recon_x, mu, logvar = self.model(data)
            
            # Flatten original data for loss calculation
            original = data.view(data.size(0), -1)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(recon_x, original, mu, logvar, self.beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {loss.item():.2f}, '
                      f'Recon: {recon_loss.item():.2f}, '
                      f'KL: {kl_loss.item():.2f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        self.train_losses.append(avg_loss)
        self.recon_losses.append(avg_recon_loss)
        self.kl_losses.append(avg_kl_loss)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                recon_x, mu, logvar = self.model(data)
                
                original = data.view(data.size(0), -1)
                loss, _, _ = vae_loss_function(recon_x, original, mu, logvar, self.beta)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training VAE on {self.device}")
        print(f"Beta (KL weight): {self.beta}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 70)
            
            # Train
            train_loss, recon_loss, kl_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f})')
            print(f'Val Loss: {val_loss:.4f}')
    
    def plot_losses(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total losses
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Decomposed losses
        axes[1].plot(self.recon_losses, label='Reconstruction Loss', color='green')
        axes[1].plot(self.kl_losses, label='KL Divergence Loss', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def visualize_reconstructions(model, test_loader, device, num_samples=8):
    """Visualize reconstruction results"""
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        
        recon_x, mu, logvar = model(data)
        
        # Convert to numpy for plotting
        original = data.cpu().numpy()
        reconstructed = recon_x.view(-1, 28, 28).cpu().numpy()
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed[i], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def visualize_generation(model, device, num_samples=16):
    """Visualize generated samples"""
    model.eval()
    
    # Generate samples
    samples = model.sample(num_samples, device)
    samples = samples.view(-1, 28, 28).cpu().numpy()
    
    # Plot generated samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].set_title(f'Generated {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_latent_space(model, test_loader, device, num_samples=1000):
    """Visualize latent space representation"""
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i * data.size(0) >= num_samples:
                break
                
            data = data.to(device)
            mu, logvar = model.encode(data.view(data.size(0), -1))
            
            # Use mean of latent distribution for visualization
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Use t-SNE for visualization if latent_dim > 2
    if latent_vectors.shape[1] > 2:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('VAE Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()


def latent_space_interpolation(model, test_loader, device, num_steps=10):
    """Visualize interpolation in latent space"""
    model.eval()
    
    with torch.no_grad():
        # Get two different samples
        data, _ = next(iter(test_loader))
        data = data[:2].to(device)
        
        # Encode to get latent representations
        mu, logvar = model.encode(data.view(data.size(0), -1))
        
        # Use means for interpolation
        z1, z2 = mu[0], mu[1]
        
        # Create interpolation
        alphas = np.linspace(0, 1, num_steps)
        interpolated_images = []
        
        for alpha in alphas:
            # Linear interpolation in latent space
            z_interpolated = (1 - alpha) * z1 + alpha * z2
            
            # Decode
            decoded = model.decode(z_interpolated.unsqueeze(0))
            interpolated_images.append(decoded.view(28, 28).cpu().numpy())
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'α={alphas[i]:.1f}')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        plt.show()


def analyze_latent_dimensions(model, test_loader, device):
    """Analyze individual latent dimensions"""
    model.eval()
    
    with torch.no_grad():
        # Get latent representations
        data, _ = next(iter(test_loader))
        data = data[:100].to(device)
        
        mu, logvar = model.encode(data.view(data.size(0), -1))
        
        # Analyze statistics of each latent dimension
        latent_means = torch.mean(mu, dim=0).cpu().numpy()
        latent_stds = torch.std(mu, dim=0).cpu().numpy()
        
        # Plot latent dimension statistics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Mean values
        axes[0].bar(range(len(latent_means)), latent_means, alpha=0.7)
        axes[0].set_xlabel('Latent Dimension')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Mean Values of Latent Dimensions')
        axes[0].grid(True, alpha=0.3)
        
        # Standard deviations
        axes[1].bar(range(len(latent_stds)), latent_stds, alpha=0.7, color='orange')
        axes[1].set_xlabel('Latent Dimension')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_title('Standard Deviations of Latent Dimensions')
        axes[1].grid(True, alpha=0.3)
        
        # Distribution of first latent dimension
        first_dim_values = mu[:, 0].cpu().numpy()
        axes[2].hist(first_dim_values, bins=20, alpha=0.7, color='green', density=True)
        
        # Overlay standard normal distribution
        x = np.linspace(-3, 3, 100)
        y = stats.norm.pdf(x, 0, 1)
        axes[2].plot(x, y, 'r-', label='Standard Normal')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Distribution of First Latent Dimension')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def get_mnist_loaders(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Main execution example"""
    print("🎯 Starting Vanilla VAE Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Create model
    print("🧠 Creating Vanilla VAE...")
    model = VanillaVAE(
        input_dim=784,
        hidden_dims=[512, 256],
        latent_dim=20
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VAETrainer(model, device, beta=1.0)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, test_loader, epochs=20)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize reconstructions
    print("🎨 Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device)
    
    # Visualize generation
    print("🎲 Visualizing generation...")
    visualize_generation(model, device)
    
    # Visualize latent space
    print("🔍 Visualizing latent space...")
    visualize_latent_space(model, test_loader, device)
    
    # Latent space interpolation
    print("🌈 Latent space interpolation...")
    latent_space_interpolation(model, test_loader, device)
    
    # Analyze latent dimensions
    print("📊 Analyzing latent dimensions...")
    analyze_latent_dimensions(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'vanilla_vae.pth')
    print("💾 Model saved as 'vanilla_vae.pth'")
    
    print("\n📋 Summary:")
    print("VAE learns a probabilistic latent representation that allows")
    print("both reconstruction and generation of new samples.")


if __name__ == "__main__":
    main()