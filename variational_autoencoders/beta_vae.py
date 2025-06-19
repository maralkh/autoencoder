"""
Beta-VAE Implementation
======================

VAE with adjustable beta parameter for better disentanglement
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class BetaVAE(nn.Module):
    """
    Beta-VAE with controllable disentanglement
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): Hidden layer dimensions
        latent_dim (int): Latent space dimension
        beta (float): Beta parameter for KL divergence weighting
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=10, beta=4.0):
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
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
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Complete forward pass"""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar, z
    
    def sample(self, num_samples, device):
        """Generate samples from the model"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
        return samples


def beta_vae_loss_function(recon_x, x, mu, logvar, beta):
    """
    Beta-VAE loss function with adjustable beta
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Beta parameter for KL weighting
    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta weighting
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


class BetaVAETrainer:
    """Training class for Beta-VAE"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
            recon_x, mu, logvar, z = self.model(data)
            
            # Flatten original data
            original = data.view(data.size(0), -1)
            
            # Calculate loss
            loss, recon_loss, kl_loss = beta_vae_loss_function(
                recon_x, original, mu, logvar, self.model.beta
            )
            
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
                recon_x, mu, logvar, z = self.model(data)
                
                original = data.view(data.size(0), -1)
                loss, _, _ = beta_vae_loss_function(
                    recon_x, original, mu, logvar, self.model.beta
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training Beta-VAE on {self.device}")
        print(f"Beta parameter: {self.model.beta}")
        
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
        axes[0].set_title(f'Total Loss (β={self.model.beta})')
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


def visualize_latent_traversal(model, device, latent_dim_idx=0, num_steps=10, range_val=3):
    """
    Visualize latent space traversal for disentanglement analysis
    
    Args:
        model: Trained Beta-VAE model
        device: Device
        latent_dim_idx: Index of latent dimension to traverse
        num_steps: Number of steps in traversal
        range_val: Range of values to traverse
    """
    model.eval()
    
    with torch.no_grad():
        # Create base latent vector (zeros)
        base_z = torch.zeros(1, model.latent_dim).to(device)
        
        # Create traversal values
        traversal_values = np.linspace(-range_val, range_val, num_steps)
        
        # Generate images for each traversal value
        traversal_images = []
        for val in traversal_values:
            z = base_z.clone()
            z[0, latent_dim_idx] = val
            
            decoded = model.decode(z)
            traversal_images.append(decoded.view(28, 28).cpu().numpy())
        
        # Plot traversal
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        
        for i, img in enumerate(traversal_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{traversal_values[i]:.1f}')
            axes[i].axis('off')
        
        plt.suptitle(f'Latent Dimension {latent_dim_idx} Traversal')
        plt.tight_layout()
        plt.show()


def analyze_disentanglement(model, test_loader, device, num_samples=1000):
    """
    Analyze disentanglement quality using various metrics
    """
    model.eval()
    
    # Collect latent representations and labels
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i * data.size(0) >= num_samples:
                break
                
            data = data.to(device)
            mu, logvar, _, _ = model(data)
            
            # Use mean for analysis
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Calculate mutual information between latent dimensions and labels
    mutual_info_scores = []
    
    for dim in range(model.latent_dim):
        # Calculate mutual information approximation using linear regression
        X = latent_vectors[:, dim].reshape(-1, 1)
        y = labels
        
        # Use R² as proxy for mutual information
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        mutual_info_scores.append(max(0, r2))  # Ensure non-negative
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Mutual information scores
    axes[0, 0].bar(range(len(mutual_info_scores)), mutual_info_scores, alpha=0.7)
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mutual Information (R²)')
    axes[0, 0].set_title('Disentanglement: Latent Dims vs Labels')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Latent dimension variances
    latent_vars = np.var(latent_vectors, axis=0)
    axes[0, 1].bar(range(len(latent_vars)), latent_vars, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].set_title('Latent Dimension Variances')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Correlation matrix of latent dimensions
    correlation_matrix = np.corrcoef(latent_vectors.T)
    im = axes[1, 0].imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_title('Latent Dimension Correlations')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Distribution of most informative latent dimension
    most_informative_dim = np.argmax(mutual_info_scores)
    for label in range(10):  # MNIST has 10 classes
        mask = labels == label
        if np.sum(mask) > 0:
            axes[1, 1].hist(latent_vectors[mask, most_informative_dim], 
                           alpha=0.7, label=f'Digit {label}', bins=20)
    
    axes[1, 1].set_xlabel('Latent Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Most Informative Dimension ({most_informative_dim})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Disentanglement Analysis Summary:")
    print(f"Most informative latent dimension: {most_informative_dim}")
    print(f"Max mutual information score: {max(mutual_info_scores):.4f}")
    print(f"Average mutual information: {np.mean(mutual_info_scores):.4f}")
    print(f"Latent dimension utilization: {np.sum(np.array(latent_vars) > 0.1)} / {len(latent_vars)}")
    
    return mutual_info_scores, most_informative_dim


def compare_beta_values(train_loader, test_loader, device, beta_values=[1.0, 4.0, 10.0]):
    """Compare different beta values"""
    results = {}
    
    for beta in beta_values:
        print(f"\n🧠 Training Beta-VAE with β={beta}")
        
        # Create model
        model = BetaVAE(
            input_dim=784,
            hidden_dims=[400, 200],
            latent_dim=10,
            beta=beta
        )
        
        # Train
        trainer = BetaVAETrainer(model, device)
        trainer.train(train_loader, test_loader, epochs=10)  # Quick training
        
        # Analyze disentanglement
        mutual_info_scores, most_informative_dim = analyze_disentanglement(
            model, test_loader, device, num_samples=500
        )
        
        results[beta] = {
            'final_loss': trainer.train_losses[-1],
            'recon_loss': trainer.recon_losses[-1],
            'kl_loss': trainer.kl_losses[-1],
            'max_mutual_info': max(mutual_info_scores),
            'avg_mutual_info': np.mean(mutual_info_scores),
            'most_informative_dim': most_informative_dim
        }
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    beta_list = list(results.keys())
    recon_losses = [results[b]['recon_loss'] for b in beta_list]
    kl_losses = [results[b]['kl_loss'] for b in beta_list]
    max_mi = [results[b]['max_mutual_info'] for b in beta_list]
    avg_mi = [results[b]['avg_mutual_info'] for b in beta_list]
    
    # Reconstruction loss vs Beta
    axes[0, 0].plot(beta_list, recon_losses, 'o-', color='blue')
    axes[0, 0].set_xlabel('Beta Value')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('Reconstruction Loss vs Beta')
    axes[0, 0].grid(True)
    
    # KL loss vs Beta
    axes[0, 1].plot(beta_list, kl_losses, 'o-', color='red')
    axes[0, 1].set_xlabel('Beta Value')
    axes[0, 1].set_ylabel('KL Divergence Loss')
    axes[0, 1].set_title('KL Loss vs Beta')
    axes[0, 1].grid(True)
    
    # Max mutual information vs Beta
    axes[1, 0].plot(beta_list, max_mi, 'o-', color='green')
    axes[1, 0].set_xlabel('Beta Value')
    axes[1, 0].set_ylabel('Max Mutual Information')
    axes[1, 0].set_title('Disentanglement vs Beta')
    axes[1, 0].grid(True)
    
    # Average mutual information vs Beta
    axes[1, 1].plot(beta_list, avg_mi, 'o-', color='purple')
    axes[1, 1].set_xlabel('Beta Value')
    axes[1, 1].set_ylabel('Average Mutual Information')
    axes[1, 1].set_title('Average Disentanglement vs Beta')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
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


def main():
    """Main execution example"""
    print("⚖️ Starting Beta-VAE Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Create model with higher beta for better disentanglement
    print("🧠 Creating Beta-VAE...")
    model = BetaVAE(
        input_dim=784,
        hidden_dims=[400, 200],
        latent_dim=10,
        beta=4.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = BetaVAETrainer(model, device)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, test_loader, epochs=20)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize latent traversals
    print("🎨 Visualizing latent traversals...")
    for dim in range(min(5, model.latent_dim)):  # Show first 5 dimensions
        visualize_latent_traversal(model, device, latent_dim_idx=dim)
    
    # Analyze disentanglement
    print("🔍 Analyzing disentanglement...")
    analyze_disentanglement(model, test_loader, device)
    
    # Compare different beta values
    print("⚖️ Comparing different beta values...")
    beta_comparison = compare_beta_values(train_loader, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'beta_vae.pth')
    print("💾 Model saved as 'beta_vae.pth'")
    
    print("\n📋 Summary:")
    print("Beta-VAE uses a weighted KL divergence term to control the")
    print("trade-off between reconstruction and disentanglement.")
    print("Higher beta values encourage better disentanglement but")
    print("may hurt reconstruction quality.")


if __name__ == "__main__":
    main()