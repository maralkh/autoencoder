"""
Vanilla Autoencoder Implementation
==================================

Simplest type of Autoencoder for dimensionality reduction and data reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class VanillaAutoencoder(nn.Module):
    """
    Vanilla Autoencoder with fully connected layers
    
    Args:
        input_dim (int): Input dimension (e.g., 784 for MNIST)
        hidden_dims (list): List of hidden layer dimensions
        latent_dim (int): Latent space dimension
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=64):
        super(VanillaAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True) if i < len(dims) - 2 else nn.Identity()
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(dims_reversed[i], dims_reversed[i+1]),
                nn.ReLU(inplace=True) if i < len(dims_reversed) - 2 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Complete forward pass"""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent


class AutoencoderTrainer:
    """
    Training class for Autoencoder
    
    Args:
        model: Autoencoder model
        device: Training device (cuda/cpu)
        lr: Learning rate
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.model(data)
            
            # Flatten original data for loss calculation
            original = data.view(data.size(0), -1)
            
            # Calculate reconstruction loss
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
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                reconstructed, latent = self.model(data)
                
                original = data.view(data.size(0), -1)
                loss = self.criterion(reconstructed, original)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
    
    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


def visualize_reconstructions(model, test_loader, device, num_samples=8):
    """Visualize reconstruction results"""
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        
        reconstructed, latent = model(data)
        
        # Convert to numpy for plotting
        original = data.cpu().numpy()
        reconstructed = reconstructed.view(-1, 28, 28).cpu().numpy()
        
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
            _, latent = model(data)
            
            latent_vectors.append(latent.cpu().numpy())
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
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
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
    print("🚀 Starting Vanilla Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    
    # Create model
    print("🧠 Creating Vanilla Autoencoder...")
    model = VanillaAutoencoder(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        latent_dim=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AutoencoderTrainer(model, device)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, test_loader, epochs=20)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize results
    print("🎨 Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device)
    
    # Visualize latent space
    print("🔍 Visualizing latent space...")
    visualize_latent_space(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'vanilla_autoencoder.pth')
    print("💾 Model saved as 'vanilla_autoencoder.pth'")


if __name__ == "__main__":
    main()