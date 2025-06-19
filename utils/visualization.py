"""
Visualization Utilities
=======================

Common visualization functions for autoencoders
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class AutoencoderVisualizer:
    """Comprehensive visualization class for autoencoders"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def plot_training_curves(self, train_losses, val_losses=None, additional_losses=None, 
                           save_path=None, figsize=(12, 8)):
        """
        Plot training curves with multiple loss components
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            additional_losses: Dict of additional loss components
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Main loss curve
        epochs = range(1, len(train_losses) + 1)
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale
        axes[0, 1].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            axes[0, 1].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_title('Training Progress (Log Scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Additional losses
        if additional_losses:
            for i, (name, losses) in enumerate(additional_losses.items()):
                axes[1, i % 2].plot(epochs[:len(losses)], losses, 
                                  label=name, linewidth=2)
                axes[1, i % 2].set_xlabel('Epoch')
                axes[1, i % 2].set_ylabel('Loss')
                axes[1, i % 2].set_title(f'{name} Over Time')
                axes[1, i % 2].legend()
                axes[1, i % 2].grid(True, alpha=0.3)
        
        # If no additional losses, plot loss smoothing
        else:
            # Smoothed loss
            if len(train_losses) > 10:
                window_size = max(1, len(train_losses) // 20)
                smoothed = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
                axes[1, 0].plot(range(window_size, len(train_losses) + 1), smoothed, 
                              'g-', label='Smoothed Training Loss', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Smoothed Training Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Loss derivative (learning progress)
            if len(train_losses) > 2:
                loss_diff = np.diff(train_losses)
                axes[1, 1].plot(range(2, len(train_losses) + 1), loss_diff, 
                              'purple', label='Loss Change', linewidth=2)
                axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss Change')
                axes[1, 1].set_title('Learning Progress')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_reconstructions(self, data_loader, num_samples=8, save_path=None):
        """Visualize original vs reconstructed images"""
        with torch.no_grad():
            data, _ = next(iter(data_loader))
            data = data[:num_samples].to(self.device)
            
            # Get reconstructions
            if hasattr(self.model, 'forward'):
                output = self.model(data)
                if isinstance(output, tuple):
                    reconstructed = output[0]
                else:
                    reconstructed = output
            else:
                reconstructed = self.model.decode(self.model.encode(data))
            
            # Convert to numpy
            original = data.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()
            
            # Plot
            fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
            
            for i in range(num_samples):
                # Original
                if len(original.shape) == 4 and original.shape[1] == 1:
                    axes[0, i].imshow(original[i].squeeze(), cmap='gray')
                elif len(original.shape) == 4 and original.shape[1] == 3:
                    img = np.transpose(original[i], (1, 2, 0))
                    img = np.clip((img + 1) / 2, 0, 1)  # Denormalize
                    axes[0, i].imshow(img)
                else:
                    axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
                
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed
                if len(reconstructed.shape) == 4 and reconstructed.shape[1] == 1:
                    axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
                elif len(reconstructed.shape) == 4 and reconstructed.shape[1] == 3:
                    img = np.transpose(reconstructed[i], (1, 2, 0))
                    img = np.clip(img, 0, 1)
                    axes[1, i].imshow(img)
                else:
                    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
                
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
                
                # Difference
                diff = np.abs(original[i] - reconstructed[i])
                if len(diff.shape) == 3 and diff.shape[0] == 1:
                    axes[2, i].imshow(diff.squeeze(), cmap='hot')
                elif len(diff.shape) == 3 and diff.shape[0] == 3:
                    diff_img = np.mean(diff, axis=0)
                    axes[2, i].imshow(diff_img, cmap='hot')
                else:
                    axes[2, i].imshow(diff.reshape(28, 28), cmap='hot')
                
                axes[2, i].set_title('Difference')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def visualize_latent_space(self, data_loader, method='tsne', num_samples=1000, 
                              save_path=None, interactive=False):
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            data_loader: DataLoader
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            num_samples: Number of samples to visualize
            save_path: Path to save plot
            interactive: Whether to create interactive plot
        """
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for i, (data, label) in enumerate(data_loader):
                if i * data.size(0) >= num_samples:
                    break
                
                data = data.to(self.device)
                
                # Get latent representation
                if hasattr(self.model, 'encode'):
                    if hasattr(self.model, 'reparameterize'):  # VAE
                        mu, logvar = self.model.encode(data.view(data.size(0), -1))
                        latent = mu
                    else:  # Regular AE
                        latent = self.model.encode(data.view(data.size(0), -1))
                else:
                    # Try forward pass and extract latent
                    output = self.model(data)
                    if isinstance(output, tuple) and len(output) > 1:
                        latent = output[1]
                    else:
                        continue
                
                latent_vectors.append(latent.cpu().numpy())
                labels.append(label.numpy())
        
        if not latent_vectors:
            print("Could not extract latent representations")
            return
        
        # Concatenate data
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
        labels = np.concatenate(labels, axis=0)[:num_samples]
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = reducer.fit_transform(latent_vectors)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_vectors)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                latent_2d = reducer.fit_transform(latent_vectors)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                reducer = TSNE(n_components=2, random_state=42)
                latent_2d = reducer.fit_transform(latent_vectors)
        
        # Create visualization
        if interactive:
            # Interactive plot with Plotly
            fig = px.scatter(
                x=latent_2d[:, 0], y=latent_2d[:, 1], 
                color=labels.astype(str),
                title=f'Latent Space Visualization ({method.upper()})',
                labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Label'}
            )
            fig.show()
        else:
            # Static plot with Matplotlib
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=labels, cmap='tab10', alpha=0.7, s=20)
            plt.colorbar(scatter, label='Class')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(f'Latent Space Visualization ({method.upper()})')
            
            if method == 'pca':
                plt.xlabel(f'PC1 ({reducer.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({reducer.explained_variance_ratio_[1]:.2%} variance)')
            
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_latent_traversal(self, latent_dim_idx, num_steps=10, range_val=3, 
                            image_shape=(28, 28), save_path=None):
        """
        Visualize latent space traversal for a specific dimension
        
        Args:
            latent_dim_idx: Index of latent dimension to traverse
            num_steps: Number of steps in traversal
            range_val: Range of values to traverse
            image_shape: Shape of output images
            save_path: Path to save plot
        """
        if not hasattr(self.model, 'decode'):
            print("Model must have a decode method for latent traversal")
            return
        
        with torch.no_grad():
            # Get latent dimension size
            if hasattr(self.model, 'latent_dim'):
                latent_dim = self.model.latent_dim
            else:
                # Try to infer from model
                dummy_input = torch.zeros(1, 784).to(self.device)
                if hasattr(self.model, 'encode'):
                    dummy_latent = self.model.encode(dummy_input)
                    latent_dim = dummy_latent.shape[1]
                else:
                    print("Cannot determine latent dimension size")
                    return
            
            # Create base latent vector (zeros or random)
            base_z = torch.zeros(1, latent_dim).to(self.device)
            
            # Create traversal values
            traversal_values = np.linspace(-range_val, range_val, num_steps)
            
            # Generate images for each traversal value
            traversal_images = []
            for val in traversal_values:
                z = base_z.clone()
                z[0, latent_dim_idx] = val
                
                decoded = self.model.decode(z)
                
                # Reshape if needed
                if decoded.shape[1:] != image_shape:
                    if len(image_shape) == 2:
                        decoded = decoded.view(-1, *image_shape)
                    else:
                        decoded = decoded.view(-1, *image_shape)
                
                traversal_images.append(decoded[0].cpu().numpy())
            
            # Plot traversal
            fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
            
            for i, img in enumerate(traversal_images):
                if len(img.shape) == 3 and img.shape[0] == 1:
                    axes[i].imshow(img.squeeze(), cmap='gray')
                elif len(img.shape) == 3 and img.shape[0] == 3:
                    img_show = np.transpose(img, (1, 2, 0))
                    img_show = np.clip(img_show, 0, 1)
                    axes[i].imshow(img_show)
                else:
                    axes[i].imshow(img, cmap='gray')
                
                axes[i].set_title(f'{traversal_values[i]:.1f}')
                axes[i].axis('off')
            
            plt.suptitle(f'Latent Dimension {latent_dim_idx} Traversal')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_reconstruction_error_distribution(self, data_loader, save_path=None):
        """Plot distribution of reconstruction errors"""
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                
                # Get reconstruction
                output = self.model(data)
                if isinstance(output, tuple):
                    reconstructed = output[0]
                else:
                    reconstructed = output
                
                # Calculate reconstruction error per sample
                original = data.view(data.size(0), -1)
                reconstructed = reconstructed.view(reconstructed.size(0), -1)
                
                # MSE per sample
                mse_per_sample = torch.mean((original - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(mse_per_sample.cpu().numpy())
        
        # Plot distribution
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(reconstruction_errors, bins=50, alpha=0.7, density=True, edgecolor='black')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(reconstruction_errors, vert=True)
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error Statistics')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        errors = np.array(reconstruction_errors)
        print(f"Reconstruction Error Statistics:")
        print(f"Mean: {np.mean(errors):.6f}")
        print(f"Std: {np.std(errors):.6f}")
        print(f"Min: {np.min(errors):.6f}")
        print(f"Max: {np.max(errors):.6f}")
        print(f"Median: {np.median(errors):.6f}")
        print(f"95th percentile: {np.percentile(errors, 95):.6f}")


def plot_model_architecture(model, input_shape=(1, 28, 28), save_path=None):
    """
    Visualize model architecture
    
    Args:
        model: PyTorch model
        input_shape: Input shape for the model
        save_path: Path to save the plot
    """
    try:
        from torchsummary import summary
        print("Model Architecture Summary:")
        print("=" * 50)
        summary(model, input_shape)
    except ImportError:
        print("torchsummary not installed. Install with: pip install torchsummary")
        
        # Alternative: manual architecture description
        print("Model Architecture:")
        print("=" * 50)
        total_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                num_params = sum(p.numel() for p in module.parameters())
                total_params += num_params
                print(f"{name}: {module} - Parameters: {num_params:,}")
        
        print(f"\nTotal Parameters: {total_params:,}")


def plot_weight_distributions(model, save_path=None):
    """Plot weight distributions for each layer"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2 and layer_idx < 6:
            weights = param.data.cpu().numpy().flatten()
            
            axes[layer_idx].hist(weights, bins=50, alpha=0.7, edgecolor='black')
            axes[layer_idx].set_title(f'{name}\nMean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}')
            axes[layer_idx].set_xlabel('Weight Value')
            axes[layer_idx].set_ylabel('Frequency')
            axes[layer_idx].grid(True, alpha=0.3)
            
            layer_idx += 1
    
    # Hide unused subplots
    for i in range(layer_idx, 6):
        axes[i].axis('off')
    
    plt.suptitle('Weight Distributions Across Layers')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_gradient_flow(model, save_path=None):
    """Plot gradient flow through the model"""
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and 'bias' not in name:
            layers.append(name.replace('.weight', ''))
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7, lw=1, color="c", label="max gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7, lw=1, color="b", label="mean gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads)*1.1)
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comparison_plot(models_dict, data_loader, metric='reconstruction_error', 
                          save_path=None, figsize=(12, 8)):
    """
    Compare multiple models on a given metric
    
    Args:
        models_dict: Dict of {model_name: model} pairs
        data_loader: DataLoader for evaluation
        metric: Metric to compare ('reconstruction_error', 'latent_variance')
        save_path: Path to save plot
        figsize: Figure size
    """
    results = {}
    
    for name, model in models_dict.items():
        model.eval()
        
        if metric == 'reconstruction_error':
            errors = []
            with torch.no_grad():
                for data, _ in data_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    output = model(data)
                    if isinstance(output, tuple):
                        reconstructed = output[0]
                    else:
                        reconstructed = output
                    
                    original = data.view(data.size(0), -1)
                    reconstructed = reconstructed.view(reconstructed.size(0), -1)
                    
                    mse = torch.mean((original - reconstructed) ** 2, dim=1)
                    errors.extend(mse.cpu().numpy())
            
            results[name] = errors
        
        elif metric == 'latent_variance':
            latent_vectors = []
            with torch.no_grad():
                for data, _ in data_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    if hasattr(model, 'encode'):
                        latent = model.encode(data.view(data.size(0), -1))
                        if isinstance(latent, tuple):  # VAE case
                            latent = latent[0]
                        latent_vectors.append(latent.cpu().numpy())
            
            if latent_vectors:
                all_latent = np.concatenate(latent_vectors, axis=0)
                variance_per_dim = np.var(all_latent, axis=0)
                results[name] = variance_per_dim
    
    # Create comparison plot
    plt.figure(figsize=figsize)
    
    if metric == 'reconstruction_error':
        # Box plot comparison
        plt.boxplot([results[name] for name in results.keys()], 
                   labels=list(results.keys()))
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error Comparison')
        plt.yscale('log')
        
    elif metric == 'latent_variance':
        # Line plot comparison
        for name, variances in results.items():
            plt.plot(variances, 'o-', label=name, alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Variance')
        plt.title('Latent Space Variance Comparison')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results


def plot_interpolation_grid(model, data_loader, num_interpolations=5, num_pairs=3, 
                           save_path=None):
    """
    Create a grid of interpolations between different samples
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader
        num_interpolations: Number of interpolation steps
        num_pairs: Number of sample pairs to interpolate
        save_path: Path to save plot
    """
    model.eval()
    
    with torch.no_grad():
        # Get sample pairs
        data, _ = next(iter(data_loader))
        if torch.cuda.is_available():
            data = data.cuda()
        
        # Select pairs
        pairs = []
        for i in range(0, min(num_pairs * 2, data.size(0)), 2):
            pairs.append((data[i:i+1], data[i+1:i+2]))
        
        fig, axes = plt.subplots(num_pairs, num_interpolations + 2, 
                                figsize=(2 * (num_interpolations + 2), 2 * num_pairs))
        
        for pair_idx, (img1, img2) in enumerate(pairs):
            # Encode images
            if hasattr(model, 'encode'):
                if hasattr(model, 'reparameterize'):  # VAE
                    mu1, logvar1 = model.encode(img1.view(1, -1))
                    mu2, logvar2 = model.encode(img2.view(1, -1))
                    z1, z2 = mu1, mu2
                else:  # Regular AE
                    z1 = model.encode(img1.view(1, -1))
                    z2 = model.encode(img2.view(1, -1))
            else:
                continue
            
            # Create interpolations
            alphas = np.linspace(0, 1, num_interpolations + 2)
            
            for i, alpha in enumerate(alphas):
                if i == 0:
                    # Original image 1
                    img_show = img1[0].cpu().numpy()
                elif i == len(alphas) - 1:
                    # Original image 2
                    img_show = img2[0].cpu().numpy()
                else:
                    # Interpolated
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img_interp = model.decode(z_interp)
                    img_show = img_interp[0].cpu().numpy()
                
                # Plot
                if len(img_show.shape) == 3 and img_show.shape[0] == 1:
                    axes[pair_idx, i].imshow(img_show.squeeze(), cmap='gray')
                elif len(img_show.shape) == 3 and img_show.shape[0] == 3:
                    img_show = np.transpose(img_show, (1, 2, 0))
                    img_show = np.clip(img_show, 0, 1)
                    axes[pair_idx, i].imshow(img_show)
                else:
                    axes[pair_idx, i].imshow(img_show.reshape(28, 28), cmap='gray')
                
                axes[pair_idx, i].axis('off')
                if pair_idx == 0:
                    axes[pair_idx, i].set_title(f'α={alpha:.2f}')
        
        plt.suptitle('Latent Space Interpolations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_loss_landscape_plot(model, data_loader, param1_name, param2_name, 
                              param_range=0.1, num_points=20, save_path=None):
    """
    Create a 2D loss landscape plot by varying two parameters
    
    Args:
        model: Model to analyze
        data_loader: DataLoader for computing loss
        param1_name: Name of first parameter to vary
        param2_name: Name of second parameter to vary
        param_range: Range to vary parameters (as fraction of current value)
        num_points: Number of points in each dimension
        save_path: Path to save plot
    """
    model.eval()
    
    # Get original parameters
    param1 = None
    param2 = None
    for name, param in model.named_parameters():
        if param1_name in name:
            param1 = param.data.clone()
        if param2_name in name:
            param2 = param.data.clone()
    
    if param1 is None or param2 is None:
        print(f"Could not find parameters {param1_name} or {param2_name}")
        return
    
    # Create parameter grids
    p1_range = torch.linspace(-param_range, param_range, num_points)
    p2_range = torch.linspace(-param_range, param_range, num_points)
    
    loss_landscape = np.zeros((num_points, num_points))
    
    # Compute loss for each parameter combination
    for i, p1_delta in enumerate(p1_range):
        for j, p2_delta in enumerate(p2_range):
            # Modify parameters
            for name, param in model.named_parameters():
                if param1_name in name:
                    param.data = param1 + p1_delta * param1.abs().mean()
                if param2_name in name:
                    param.data = param2 + p2_delta * param2.abs().mean()
            
            # Compute loss
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for data, _ in data_loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    output = model(data)
                    if isinstance(output, tuple):
                        reconstructed = output[0]
                    else:
                        reconstructed = output
                    
                    loss = torch.nn.functional.mse_loss(reconstructed, data)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches >= 5:  # Limit for speed
                        break
            
            loss_landscape[i, j] = total_loss / num_batches
    
    # Restore original parameters
    for name, param in model.named_parameters():
        if param1_name in name:
            param.data = param1
        if param2_name in name:
            param.data = param2
    
    # Plot landscape
    plt.figure(figsize=(10, 8))
    plt.contourf(p1_range.numpy(), p2_range.numpy(), loss_landscape, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel(f'{param1_name} (relative change)')
    plt.ylabel(f'{param2_name} (relative change)')
    plt.title('Loss Landscape')
    plt.scatter([0], [0], color='red', s=100, marker='*', label='Current parameters')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Demo of visualization utilities"""
    print("🎨 Visualization Utilities Demo")
    
    # This would typically be used with a trained model
    # For demo purposes, we'll create dummy data
    
    # Demo training curves
    epochs = 20
    train_losses = [1.0 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in range(epochs)]
    val_losses = [1.1 * np.exp(-0.09 * i) + 0.1 * np.random.random() for i in range(epochs)]
    
    # Create dummy visualizer
    class DummyModel:
        def eval(self): pass
    
    visualizer = AutoencoderVisualizer(DummyModel())
    
    # Plot training curves
    print("📈 Plotting training curves...")
    visualizer.plot_training_curves(
        train_losses, 
        val_losses,
        additional_losses={'KL Loss': [0.5 * np.exp(-0.2 * i) for i in range(epochs)]}
    )
    
    print("\n✅ Visualization utilities demo completed!")
    print("Use these functions with your trained models for comprehensive analysis.")


if __name__ == "__main__":
    main()