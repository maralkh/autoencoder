"""
Denoising Autoencoder Implementation
===================================

Autoencoder that learns to remove noise from corrupted input data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder that learns to reconstruct clean data from noisy input
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): Hidden layer dimensions
        latent_dim (int): Latent space dimension
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=64, dropout_rate=0.2):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Encoder with dropout for denoising
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(dims_reversed[i], dims_reversed[i+1]),
                nn.ReLU(inplace=True) if i < len(dims_reversed) - 2 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def add_noise(self, x, noise_type='gaussian', noise_factor=0.3):
        """
        Add noise to input data
        
        Args:
            x: Input tensor
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'masking')
            noise_factor: Noise intensity
        """
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = torch.randn_like(x) * noise_factor
            noisy_x = x + noise
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            noise = torch.rand_like(x)
            noisy_x = x.clone()
            noisy_x[noise < noise_factor/2] = 0  # Salt
            noisy_x[noise > 1 - noise_factor/2] = 1  # Pepper
            
        elif noise_type == 'masking':
            # Random masking
            mask = torch.rand_like(x) > noise_factor
            noisy_x = x * mask.float()
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return torch.clamp(noisy_x, 0., 1.)
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x, add_noise_flag=True, noise_type='gaussian', noise_factor=0.3):
        """
        Forward pass with optional noise addition
        
        Args:
            x: Input tensor
            add_noise_flag: Whether to add noise during forward pass
            noise_type: Type of noise to add
            noise_factor: Noise intensity
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Add noise if specified
        if add_noise_flag:
            noisy_x = self.add_noise(x, noise_type, noise_factor)
        else:
            noisy_x = x
        
        # Encode and decode
        latent = self.encode(noisy_x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent, noisy_x


class DenoisingTrainer:
    """
    Training class for Denoising Autoencoder
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, noise_type='gaussian', noise_factor=0.3):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass with noise
            reconstructed, latent, noisy_input = self.model(
                data, add_noise_flag=True, noise_type=noise_type, noise_factor=noise_factor
            )
            
            # Original clean data (target)
            original = data.view(data.size(0), -1)
            
            # Loss between reconstructed and clean data
            loss = self.criterion(reconstructed, original)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, noise_type='gaussian', noise_factor=0.3):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                reconstructed, latent, noisy_input = self.model(
                    data, add_noise_flag=True, noise_type=noise_type, noise_factor=noise_factor
                )
                
                original = data.view(data.size(0), -1)
                loss = self.criterion(reconstructed, original)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs, noise_type='gaussian', noise_factor=0.3):
        """Complete training loop"""
        print(f"Training Denoising Autoencoder on {self.device}")
        print(f"Noise type: {noise_type}, Noise factor: {noise_factor}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, noise_type, noise_factor)
            
            # Validate
            val_loss = self.validate(val_loader, noise_type, noise_factor)
            
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
    
    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Denoising Autoencoder - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


def visualize_denoising(model, test_loader, device, noise_type='gaussian', noise_factor=0.3, num_samples=8):
    """Visualize denoising results"""
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        
        # Get reconstructed and noisy versions
        reconstructed, latent, noisy_input = model(
            data, add_noise_flag=True, noise_type=noise_type, noise_factor=noise_factor
        )
        
        # Convert to numpy for plotting
        original = data.cpu().numpy()
        noisy = noisy_input.view(-1, 28, 28).cpu().numpy()
        reconstructed = reconstructed.view(-1, 28, 28).cpu().numpy()
        
        # Plot original, noisy, and reconstructed
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Noisy image
            axes[1, i].imshow(noisy[i], cmap='gray')
            axes[1, i].set_title(f'Noisy ({noise_type})')
            axes[1, i].axis('off')
            
            # Reconstructed image
            axes[2, i].imshow(reconstructed[i], cmap='gray')
            axes[2, i].set_title('Denoised')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def compare_noise_types(model, test_loader, device, num_samples=4):
    """Compare different noise types and denoising results"""
    model.eval()
    noise_types = ['gaussian', 'salt_pepper', 'masking']
    noise_factor = 0.3
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        
        fig, axes = plt.subplots(len(noise_types) + 1, num_samples, figsize=(12, 10))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
        
        # For each noise type
        for noise_idx, noise_type in enumerate(noise_types):
            reconstructed, _, noisy_input = model(
                data, add_noise_flag=True, noise_type=noise_type, noise_factor=noise_factor
            )
            
            reconstructed = reconstructed.view(-1, 28, 28).cpu().numpy()
            
            for i in range(num_samples):
                axes[noise_idx + 1, i].imshow(reconstructed[i], cmap='gray')
                axes[noise_idx + 1, i].set_title(f'Denoised ({noise_type})')
                axes[noise_idx + 1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def get_mnist_loaders(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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
    print("🧹 Starting Denoising Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    
    # Create model
    print("🧠 Creating Denoising Autoencoder...")
    model = DenoisingAutoencoder(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        dropout_rate=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = DenoisingTrainer(model, device)
    
    # Train with different noise types
    noise_configs = [
        ('gaussian', 0.3),
        ('salt_pepper', 0.2),
        ('masking', 0.4)
    ]
    
    for noise_type, noise_factor in noise_configs:
        print(f"\n🏋️ Training with {noise_type} noise (factor: {noise_factor})...")
        trainer.train(train_loader, test_loader, epochs=10, 
                     noise_type=noise_type, noise_factor=noise_factor)
        
        # Visualize results for this noise type
        print(f"🎨 Visualizing {noise_type} denoising...")
        visualize_denoising(model, test_loader, device, 
                          noise_type=noise_type, noise_factor=noise_factor)
    
    # Plot losses
    trainer.plot_losses()
    
    # Compare all noise types
    print("🔍 Comparing different noise types...")
    compare_noise_types(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'denoising_autoencoder.pth')
    print("💾 Model saved as 'denoising_autoencoder.pth'")


if __name__ == "__main__":
    main()