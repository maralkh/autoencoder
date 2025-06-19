"""
Sparse Autoencoder Implementation
=================================

Autoencoder with sparsity constraints to learn sparse representations
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


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 regularization and KL divergence sparsity penalty
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): Hidden layer dimensions
        latent_dim (int): Latent space dimension
        sparsity_target (float): Target sparsity level (rho)
        sparsity_weight (float): Weight for sparsity penalty (beta)
        l1_weight (float): Weight for L1 regularization
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=64,
                 sparsity_target=0.05, sparsity_weight=3.0, l1_weight=1e-4):
        super(SparseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        
        # Encoder layers
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True) if i < len(dims) - 2 else nn.Sigmoid()
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(dims_reversed[i], dims_reversed[i+1]),
                nn.ReLU(inplace=True) if i < len(dims_reversed) - 2 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store activations for sparsity penalty
        self.activations = None
    
    def encode(self, x):
        """Encode input to latent space"""
        latent = self.encoder(x)
        # Store activations for sparsity calculation
        self.activations = latent
        return latent
    
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
    
    def kl_divergence_sparsity_penalty(self, activations):
        """
        Calculate KL divergence sparsity penalty
        
        Args:
            activations: Latent activations
            
        Returns:
            KL divergence penalty
        """
        # Average activation for each neuron across batch
        rho_hat = torch.mean(activations, dim=0)
        
        # KL divergence between target sparsity and actual sparsity
        rho = self.sparsity_target
        kl_div = rho * torch.log(rho / (rho_hat + 1e-8)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
        
        return torch.sum(kl_div)
    
    def l1_penalty(self):
        """Calculate L1 regularization penalty on weights"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss
    
    def compute_sparsity_loss(self):
        """Compute total sparsity loss (KL + L1)"""
        if self.activations is None:
            return 0
        
        # KL divergence sparsity penalty
        kl_penalty = self.kl_divergence_sparsity_penalty(self.activations)
        
        # L1 weight penalty
        l1_penalty = self.l1_penalty()
        
        return self.sparsity_weight * kl_penalty + self.l1_weight * l1_penalty


class SparseTrainer:
    """
    Training class for Sparse Autoencoder
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.sparsity_losses = []
        self.reconstruction_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.model(data)
            
            # Flatten original data for loss calculation
            original = data.view(data.size(0), -1)
            
            # Reconstruction loss
            recon_loss = self.criterion(reconstructed, original)
            
            # Sparsity loss
            sparsity_loss = self.model.compute_sparsity_loss()
            
            # Total loss
            total_loss_batch = recon_loss + sparsity_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {total_loss_batch.item():.6f}, '
                      f'Recon: {recon_loss.item():.6f}, '
                      f'Sparsity: {sparsity_loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_sparsity_loss = total_sparsity_loss / len(train_loader)
        
        self.train_losses.append(avg_loss)
        self.reconstruction_losses.append(avg_recon_loss)
        self.sparsity_losses.append(avg_sparsity_loss)
        
        return avg_loss, avg_recon_loss, avg_sparsity_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                reconstructed, latent = self.model(data)
                
                original = data.view(data.size(0), -1)
                recon_loss = self.criterion(reconstructed, original)
                sparsity_loss = self.model.compute_sparsity_loss()
                
                total_loss += (recon_loss + sparsity_loss).item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training Sparse Autoencoder on {self.device}")
        print(f"Sparsity target: {self.model.sparsity_target}")
        print(f"Sparsity weight: {self.model.sparsity_weight}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 70)
            
            # Train
            train_loss, recon_loss, sparsity_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.6f} (Recon: {recon_loss:.6f}, '
                  f'Sparsity: {sparsity_loss:.6f})')
            print(f'Val Loss: {val_loss:.6f}')
    
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
        axes[1].plot(self.reconstruction_losses, label='Reconstruction Loss', color='green')
        axes[1].plot(self.sparsity_losses, label='Sparsity Loss', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def analyze_sparsity(model, test_loader, device):
    """Analyze sparsity of learned representations"""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:  # Analyze subset for efficiency
                break
                
            data = data.to(device)
            _, latent = model(data)
            all_activations.append(latent.cpu().numpy())
    
    # Concatenate all activations
    all_activations = np.concatenate(all_activations, axis=0)
    
    # Calculate sparsity metrics
    mean_activation = np.mean(all_activations, axis=0)
    sparsity_ratio = np.sum(mean_activation < 0.1) / len(mean_activation)
    
    # Visualize activation distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of mean activations
    axes[0].hist(mean_activation, bins=50, alpha=0.7, color='blue')
    axes[0].axvline(model.sparsity_target, color='red', linestyle='--', 
                   label=f'Target: {model.sparsity_target}')
    axes[0].set_xlabel('Mean Activation')
    axes[0].set_ylabel('Number of Neurons')
    axes[0].set_title('Distribution of Mean Activations')
    axes[0].legend()
    axes[0].grid(True)
    
    # Heatmap of activations (sample)
    sample_activations = all_activations[:100]  # First 100 samples
    im = axes[1].imshow(sample_activations.T, cmap='viridis', aspect='auto')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Neuron')
    axes[1].set_title('Activation Heatmap')
    plt.colorbar(im, ax=axes[1])
    
    # Sparsity per neuron
    sparsity_per_neuron = np.mean(all_activations < 0.1, axis=0)
    axes[2].plot(sparsity_per_neuron, 'o-', markersize=3)
    axes[2].axhline(0.9, color='red', linestyle='--', label='90% sparse')
    axes[2].set_xlabel('Neuron Index')
    axes[2].set_ylabel('Sparsity Ratio')
    axes[2].set_title('Sparsity per Neuron')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Overall sparsity ratio: {sparsity_ratio:.3f}")
    print(f"Mean activation: {np.mean(mean_activation):.6f}")
    print(f"Target sparsity: {model.sparsity_target}")


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
    print("✨ Starting Sparse Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    
    # Create model with different sparsity configurations
    sparsity_configs = [
        {"sparsity_target": 0.05, "sparsity_weight": 3.0, "l1_weight": 1e-4},
        {"sparsity_target": 0.1, "sparsity_weight": 2.0, "l1_weight": 1e-3},
    ]
    
    for i, config in enumerate(sparsity_configs):
        print(f"\n🧠 Creating Sparse Autoencoder {i+1}...")
        model = SparseAutoencoder(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            latent_dim=64,
            **config
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Configuration: {config}")
        
        # Create trainer
        trainer = SparseTrainer(model, device)
        
        # Train
        print(f"🏋️ Training model {i+1}...")
        trainer.train(train_loader, test_loader, epochs=15)
        
        # Plot losses
        trainer.plot_losses()
        
        # Analyze sparsity
        print(f"🔍 Analyzing sparsity for model {i+1}...")
        analyze_sparsity(model, test_loader, device)
        
        # Visualize reconstructions
        print(f"🎨 Visualizing reconstructions for model {i+1}...")
        visualize_reconstructions(model, test_loader, device)
        
        # Save model
        torch.save(model.state_dict(), f'sparse_autoencoder_{i+1}.pth')
        print(f"💾 Model {i+1} saved as 'sparse_autoencoder_{i+1}.pth'")


if __name__ == "__main__":
    main()