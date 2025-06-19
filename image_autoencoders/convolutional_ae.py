"""
Convolutional Autoencoder Implementation
=======================================

Autoencoder using convolutional layers for image data
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


class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image data
    
    Args:
        input_channels (int): Number of input channels
        latent_dim (int): Latent space dimension
        image_size (int): Input image size (assumes square images)
    """
    
    def __init__(self, input_channels=1, latent_dim=128, image_size=28):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate the size after convolutions for fully connected layer
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            # First conv block: 28x28 -> 14x14
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block: 7x7 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Latent space
        self.encoder_fc = nn.Linear(128 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 7 * 7)
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            # First deconv block: 7x7 -> 7x7
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Second deconv block: 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Third deconv block: 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def _calculate_conv_output_size(self):
        """Calculate the output size after convolution layers"""
        # For MNIST (28x28), after two MaxPool2d operations: 28 -> 14 -> 7
        return 7 * 7
    
    def encode(self, x):
        """Encode input to latent space"""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        latent = self.encoder_fc(x)
        return latent
    
    def decode(self, z):
        """Decode latent representation to output"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape for conv layers
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        """Complete forward pass"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class ConvAutoencoderTrainer:
    """Training class for Convolutional Autoencoder"""
    
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
            
            # Calculate loss
            loss = self.criterion(reconstructed, data)
            
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
                loss = self.criterion(reconstructed, data)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training Convolutional Autoencoder on {self.device}")
        
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
        plt.title('Convolutional Autoencoder - Training and Validation Loss')
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
        reconstructed = reconstructed.cpu().numpy()
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(original[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def visualize_feature_maps(model, test_loader, device, layer_idx=0):
    """Visualize feature maps from encoder layers"""
    model.eval()
    
    # Hook to capture feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.cpu().detach())
    
    # Register hook on specified layer
    layers = list(model.encoder_conv.children())
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    # Get a sample
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        sample = data[0:1].to(device)
        
        # Forward pass to trigger hook
        model.encode(sample)
        
        # Get feature maps
        if feature_maps:
            fmaps = feature_maps[0].squeeze().numpy()
            
            # Plot feature maps
            num_features = min(16, fmaps.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(num_features):
                axes[i].imshow(fmaps[i], cmap='viridis')
                axes[i].set_title(f'Feature Map {i+1}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_features, 16):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps from Layer {layer_idx}')
            plt.tight_layout()
            plt.show()
    
    # Remove hook
    handle.remove()


def analyze_latent_space(model, test_loader, device, num_samples=1000):
    """Analyze the learned latent space"""
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
    
    # Analyze latent space properties
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Latent dimension statistics
    latent_means = np.mean(latent_vectors, axis=0)
    latent_stds = np.std(latent_vectors, axis=0)
    
    axes[0, 0].plot(latent_means, 'o-', alpha=0.7, label='Mean')
    axes[0, 0].plot(latent_stds, 'o-', alpha=0.7, label='Std')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Latent Dimension Statistics')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    scatter = axes[0, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0, 1].set_title('PCA of Latent Space')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. Latent activation histogram
    axes[1, 0].hist(latent_vectors.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Latent Activations')
    axes[1, 0].grid(True)
    
    # 4. Correlation matrix (subset)
    subset_size = min(20, latent_vectors.shape[1])
    corr_matrix = np.corrcoef(latent_vectors[:, :subset_size].T)
    im = axes[1, 1].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title(f'Correlation Matrix (First {subset_size} dims)')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Latent space analysis:")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    print(f"Mean activation: {np.mean(latent_vectors):.4f}")
    print(f"Std activation: {np.std(latent_vectors):.4f}")
    print(f"PCA variance explained (2 components): {sum(pca.explained_variance_ratio_):.2%}")


def compare_architectures(train_loader, test_loader, device):
    """Compare different convolutional architectures"""
    
    # Define different architectures
    architectures = {
        'Simple': {
            'encoder': nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            'latent_size': 32 * 7 * 7,
            'latent_dim': 64
        },
        'Deep': {
            'encoder': nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            ),
            'latent_size': 128 * 7 * 7,
            'latent_dim': 128
        }
    }
    
    results = {}
    
    for name, config in architectures.items():
        print(f"\n🧠 Training {name} Architecture...")
        
        # Create custom model
        class CustomConvAE(nn.Module):
            def __init__(self, encoder, latent_size, latent_dim):
                super().__init__()
                self.encoder_conv = encoder
                self.encoder_fc = nn.Linear(latent_size, latent_dim)
                self.decoder_fc = nn.Linear(latent_dim, latent_size)
                
                # Simple decoder (reverse of encoder)
                self.decoder_conv = nn.Sequential(
                    nn.Conv2d(128 if 'Deep' in name else 32, 64, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
                )
            
            def encode(self, x):
                x = self.encoder_conv(x)
                x = x.view(x.size(0), -1)
                return self.encoder_fc(x)
            
            def decode(self, z):
                x = self.decoder_fc(z)
                if 'Deep' in name:
                    x = x.view(x.size(0), 128, 7, 7)
                else:
                    x = x.view(x.size(0), 32, 7, 7)
                return self.decoder_conv(x)
            
            def forward(self, x):
                latent = self.encode(x)
                reconstructed = self.decode(latent)
                return reconstructed, latent
        
        model = CustomConvAE(
            config['encoder'],
            config['latent_size'],
            config['latent_dim']
        )
        
        # Quick training
        trainer = ConvAutoencoderTrainer(model, device)
        trainer.train(train_loader, test_loader, epochs=5)
        
        results[name] = {
            'final_loss': trainer.train_losses[-1],
            'params': sum(p.numel() for p in model.parameters()),
            'model': model
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results.keys())
    losses = [results[name]['final_loss'] for name in names]
    params = [results[name]['params'] for name in names]
    
    # Final loss comparison
    axes[0].bar(names, losses, alpha=0.7)
    axes[0].set_ylabel('Final Training Loss')
    axes[0].set_title('Architecture Comparison - Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Parameter count comparison
    axes[1].bar(names, params, alpha=0.7, color='orange')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Architecture Comparison - Model Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def noise_robustness_test(model, test_loader, device, noise_levels=[0.1, 0.2, 0.3, 0.4]):
    """Test robustness to different noise levels"""
    model.eval()
    
    # Get a batch of test data
    data, _ = next(iter(test_loader))
    data = data[:8].to(device)
    
    results = {}
    
    with torch.no_grad():
        # Original reconstruction
        orig_recon, _ = model(data)
        
        for noise_level in noise_levels:
            # Add noise
            noise = torch.randn_like(data) * noise_level
            noisy_data = torch.clamp(data + noise, 0, 1)
            
            # Reconstruct noisy data
            noisy_recon, _ = model(noisy_data)
            
            # Calculate reconstruction error
            mse_orig = F.mse_loss(orig_recon, data).item()
            mse_noisy = F.mse_loss(noisy_recon, data).item()
            
            results[noise_level] = {
                'mse_increase': mse_noisy - mse_orig,
                'noisy_data': noisy_data.cpu(),
                'noisy_recon': noisy_recon.cpu()
            }
    
    # Visualize robustness
    fig, axes = plt.subplots(len(noise_levels) + 1, 8, figsize=(16, 2 * (len(noise_levels) + 1)))
    
    # Original images
    for i in range(8):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
    
    # Noisy reconstructions
    for row, noise_level in enumerate(noise_levels, 1):
        for i in range(8):
            axes[row, i].imshow(results[noise_level]['noisy_recon'][i].squeeze(), cmap='gray')
            axes[row, i].set_title(f'Noise {noise_level}')
            axes[row, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot robustness curve
    plt.figure(figsize=(10, 6))
    noise_levels_list = list(results.keys())
    mse_increases = [results[nl]['mse_increase'] for nl in noise_levels_list]
    
    plt.plot(noise_levels_list, mse_increases, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level')
    plt.ylabel('MSE Increase')
    plt.title('Model Robustness to Input Noise')
    plt.grid(True)
    plt.show()
    
    return results


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


def get_cifar10_loaders(batch_size=128):
    """Load CIFAR-10 dataset for color images"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Main execution example"""
    print("🖼️ Starting Convolutional Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Create model
    print("🧠 Creating Convolutional Autoencoder...")
    model = ConvolutionalAutoencoder(
        input_channels=1,
        latent_dim=128,
        image_size=28
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ConvAutoencoderTrainer(model, device)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, test_loader, epochs=20)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize reconstructions
    print("🎨 Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device)
    
    # Visualize feature maps
    print("🔍 Visualizing feature maps...")
    for layer_idx in [0, 3, 6]:  # Different conv layers
        print(f"Layer {layer_idx}:")
        visualize_feature_maps(model, test_loader, device, layer_idx)
    
    # Analyze latent space
    print("📊 Analyzing latent space...")
    analyze_latent_space(model, test_loader, device)
    
    # Compare architectures
    print("⚖️ Comparing different architectures...")
    arch_results = compare_architectures(train_loader, test_loader, device)
    
    # Test noise robustness
    print("🛡️ Testing noise robustness...")
    robustness_results = noise_robustness_test(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'convolutional_autoencoder.pth')
    print("💾 Model saved as 'convolutional_autoencoder.pth'")
    
    print("\n📋 Summary:")
    print("Convolutional Autoencoders use conv/deconv layers to preserve")
    print("spatial structure in images, making them ideal for image tasks.")
    print("They learn hierarchical features from low-level edges to")
    print("high-level semantic concepts.")


if __name__ == "__main__":
    main()