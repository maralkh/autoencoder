"""
Contractive Autoencoder Implementation
=====================================

Autoencoder with contractive penalty to learn robust representations
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


class ContractiveAutoencoder(nn.Module):
    """
    Contractive Autoencoder with Jacobian regularization
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): Hidden layer dimensions
        latent_dim (int): Latent space dimension
        contractive_weight (float): Weight for contractive penalty (lambda)
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=64, 
                 contractive_weight=1e-4):
        super(ContractiveAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contractive_weight = contractive_weight
        
        # Encoder layers
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.Sigmoid() if i < len(dims) - 2 else nn.Sigmoid()
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.extend([
                nn.Linear(dims_reversed[i], dims_reversed[i+1]),
                nn.Sigmoid() if i < len(dims_reversed) - 2 else nn.Sigmoid()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store encoder weights for Jacobian calculation
        self.encoder_weights = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture encoder weights"""
        def hook_fn(module, input, output):
            if isinstance(module, nn.Linear):
                self.encoder_weights = module.weight
        
        # Register hook on the last encoder layer
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(hook_fn)
    
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
    
    def compute_jacobian_penalty(self, x, latent):
        """
        Compute contractive penalty (Frobenius norm of Jacobian)
        
        Args:
            x: Input data
            latent: Latent representation
            
        Returns:
            Jacobian penalty
        """
        # Enable gradients for input
        x.requires_grad_(True)
        
        # Compute latent representation
        latent = self.encode(x)
        
        # Compute Jacobian matrix norm
        jacobian_norm = 0
        
        for i in range(latent.size(1)):  # For each latent dimension
            # Compute gradient of i-th latent unit w.r.t input
            grad_output = torch.zeros_like(latent)
            grad_output[:, i] = 1
            
            gradients = torch.autograd.grad(
                outputs=latent,
                inputs=x,
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Add squared norm of gradients
            jacobian_norm += torch.sum(gradients ** 2)
        
        return jacobian_norm / x.size(0)  # Average over batch
    
    def compute_contractive_loss(self, x, latent):
        """Compute contractive loss"""
        jacobian_penalty = self.compute_jacobian_penalty(x, latent)
        return self.contractive_weight * jacobian_penalty


class ContractiveTrainer:
    """
    Training class for Contractive Autoencoder
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.contractive_losses = []
        self.reconstruction_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_contractive_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.model(data)
            
            # Flatten original data for loss calculation
            original = data.view(data.size(0), -1)
            
            # Reconstruction loss
            recon_loss = self.criterion(reconstructed, original)
            
            # Contractive loss
            contractive_loss = self.model.compute_contractive_loss(data, latent)
            
            # Total loss
            total_loss_batch = recon_loss + contractive_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_contractive_loss += contractive_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {total_loss_batch.item():.6f}, '
                      f'Recon: {recon_loss.item():.6f}, '
                      f'Contractive: {contractive_loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_contractive_loss = total_contractive_loss / len(train_loader)
        
        self.train_losses.append(avg_loss)
        self.reconstruction_losses.append(avg_recon_loss)
        self.contractive_losses.append(avg_contractive_loss)
        
        return avg_loss, avg_recon_loss, avg_contractive_loss
    
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
                
                # Note: Contractive loss requires gradients, so skip in validation
                total_loss += recon_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training Contractive Autoencoder on {self.device}")
        print(f"Contractive weight: {self.model.contractive_weight}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 70)
            
            # Train
            train_loss, recon_loss, contractive_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.6f} (Recon: {recon_loss:.6f}, '
                  f'Contractive: {contractive_loss:.6f})')
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
        axes[1].plot(self.contractive_losses, label='Contractive Loss', color='purple')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def analyze_robustness(model, test_loader, device, noise_levels=[0.1, 0.2, 0.3]):
    """Analyze robustness of learned representations to input perturbations"""
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        # Get a batch of test data
        data, labels = next(iter(test_loader))
        data = data[:16].to(device)  # Use small batch for analysis
        
        # Original latent representations
        _, original_latent = model(data)
        
        # Test robustness to different noise levels
        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = torch.randn_like(data) * noise_level
            noisy_data = torch.clamp(data + noise, 0, 1)
            
            # Get latent representations of noisy data
            _, noisy_latent = model(noisy_data)
            
            # Calculate distance between original and noisy latent representations
            latent_distance = torch.norm(original_latent - noisy_latent, dim=1)
            avg_distance = torch.mean(latent_distance).item()
            
            results[noise_level] = avg_distance
    
    # Plot robustness analysis
    plt.figure(figsize=(10, 6))
    
    # Bar plot of average distances
    noise_levels_list = list(results.keys())
    distances = list(results.values())
    
    plt.subplot(1, 2, 1)
    plt.bar(noise_levels_list, distances, alpha=0.7, color='blue')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Latent Distance')
    plt.title('Robustness to Input Noise')
    plt.grid(True, alpha=0.3)
    
    # Visualization of original vs noisy reconstructions
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        original_recon, _ = model(data[:4])
        noise = torch.randn_like(data[:4]) * 0.2
        noisy_data = torch.clamp(data[:4] + noise, 0, 1)
        noisy_recon, _ = model(noisy_data)
    
    # Show difference in reconstructions
    diff = torch.abs(original_recon - noisy_recon)
    diff_mean = torch.mean(diff, dim=0).cpu().numpy().reshape(28, 28)
    
    plt.imshow(diff_mean, cmap='hot')
    plt.title('Reconstruction Difference Heatmap')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print("Robustness Analysis Results:")
    for noise_level, distance in results.items():
        print(f"Noise level {noise_level}: Average latent distance = {distance:.4f}")


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


def visualize_jacobian_analysis(model, test_loader, device):
    """Visualize Jacobian analysis"""
    model.eval()
    
    # Get a sample
    data, _ = next(iter(test_loader))
    sample = data[0:1].to(device)  # Single sample
    sample.requires_grad_(True)
    
    # Get latent representation
    latent = model.encode(sample)
    
    # Compute Jacobian for each latent dimension
    jacobians = []
    for i in range(latent.size(1)):
        grad_output = torch.zeros_like(latent)
        grad_output[0, i] = 1
        
        gradients = torch.autograd.grad(
            outputs=latent,
            inputs=sample,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        jacobians.append(gradients.cpu().numpy().reshape(28, 28))
    
    # Visualize first few Jacobian matrices
    num_to_show = min(8, len(jacobians))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_to_show):
        im = axes[i].imshow(jacobians[i], cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i].set_title(f'Jacobian {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def compare_contractive_weights(train_loader, test_loader, device, weights=[1e-5, 1e-4, 1e-3]):
    """Compare different contractive weights"""
    results = {}
    
    for weight in weights:
        print(f"\n🧠 Training with contractive weight: {weight}")
        
        # Create model
        model = ContractiveAutoencoder(
            input_dim=784,
            hidden_dims=[256, 128],
            latent_dim=32,
            contractive_weight=weight
        )
        
        # Train
        trainer = ContractiveTrainer(model, device)
        trainer.train(train_loader, test_loader, epochs=5)  # Quick training
        
        # Evaluate robustness
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data[:16].to(device)
            
            # Original latent
            _, original_latent = model(data)
            
            # Noisy latent
            noise = torch.randn_like(data) * 0.2
            noisy_data = torch.clamp(data + noise, 0, 1)
            _, noisy_latent = model(noisy_data)
            
            # Calculate robustness
            distance = torch.mean(torch.norm(original_latent - noisy_latent, dim=1)).item()
            
            results[weight] = {
                'final_loss': trainer.train_losses[-1],
                'robustness': distance
            }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    weights_list = list(results.keys())
    losses = [results[w]['final_loss'] for w in weights_list]
    robustness = [results[w]['robustness'] for w in weights_list]
    
    # Final loss comparison
    axes[0].bar(range(len(weights_list)), losses, alpha=0.7)
    axes[0].set_xlabel('Contractive Weight')
    axes[0].set_ylabel('Final Training Loss')
    axes[0].set_title('Training Loss vs Contractive Weight')
    axes[0].set_xticks(range(len(weights_list)))
    axes[0].set_xticklabels([f'{w:.0e}' for w in weights_list])
    axes[0].grid(True, alpha=0.3)
    
    # Robustness comparison
    axes[1].bar(range(len(weights_list)), robustness, alpha=0.7, color='orange')
    axes[1].set_xlabel('Contractive Weight')
    axes[1].set_ylabel('Latent Distance (Robustness)')
    axes[1].set_title('Robustness vs Contractive Weight')
    axes[1].set_xticks(range(len(weights_list)))
    axes[1].set_xticklabels([f'{w:.0e}' for w in weights_list])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


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
    print("🔒 Starting Contractive Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)  # Smaller batch for gradient computation
    
    # Create model
    print("🧠 Creating Contractive Autoencoder...")
    model = ContractiveAutoencoder(
        input_dim=784,
        hidden_dims=[256, 128],
        latent_dim=32,
        contractive_weight=1e-4
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ContractiveTrainer(model, device)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, test_loader, epochs=15)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize reconstructions
    print("🎨 Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device)
    
    # Analyze robustness
    print("🔍 Analyzing robustness...")
    analyze_robustness(model, test_loader, device)
    
    # Visualize Jacobian
    print("📈 Visualizing Jacobian analysis...")
    visualize_jacobian_analysis(model, test_loader, device)
    
    # Compare different contractive weights
    print("⚖️ Comparing different contractive weights...")
    comparison_results = compare_contractive_weights(train_loader, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'contractive_autoencoder.pth')
    print("💾 Model saved as 'contractive_autoencoder.pth'")
    
    print("\n📋 Summary:")
    print("Contractive Autoencoder learns robust representations by minimizing")
    print("the Frobenius norm of the Jacobian matrix of the encoder.")
    print("This encourages the model to be less sensitive to small input changes.")


if __name__ == "__main__":
    main()