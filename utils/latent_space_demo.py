"""
Latent Space Analysis Example
============================

Complete example of how to use the Latent Space Analyzer
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import autoencoder implementations
from basic_autoencoders.vanilla_autoencoder import VanillaAutoencoder, AutoencoderTrainer
from variational_autoencoders.vanilla_vae import VanillaVAE, VAETrainer
from variational_autoencoders.beta_vae import BetaVAE, BetaVAETrainer
from utils.data_loaders import get_mnist_loaders
from utils.latent_space_analyzer import LatentSpaceAnalyzer, compare_latent_spaces


def train_sample_models():
    """Train sample models for analysis"""
    print("🏋️ Training sample models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=128, train_size=5000, test_size=1000)
    train_loader_vae, test_loader_vae = get_mnist_loaders(batch_size=128, train_size=5000, test_size=1000, normalize=False)
    
    models = {}
    
    # Train Vanilla Autoencoder
    print("\n📦 Training Vanilla Autoencoder...")
    vanilla_ae = VanillaAutoencoder(input_dim=784, hidden_dims=[256, 128], latent_dim=20)
    vanilla_trainer = AutoencoderTrainer(vanilla_ae, device)
    vanilla_trainer.train(train_loader, test_loader, epochs=15)
    models['Vanilla AE'] = vanilla_ae
    
    # Train VAE
    print("\n🎯 Training VAE...")
    vae = VanillaVAE(input_dim=784, hidden_dims=[256, 128], latent_dim=20)
    vae_trainer = VAETrainer(vae, device)
    vae_trainer.train(train_loader_vae, test_loader_vae, epochs=15)
    models['VAE'] = vae
    
    # Train β-VAE
    print("\n⚖️ Training β-VAE...")
    beta_vae = BetaVAE(input_dim=784, hidden_dims=[256, 128], latent_dim=20, beta=4.0)
    beta_vae_trainer = BetaVAETrainer(beta_vae, device)
    beta_vae_trainer.train(train_loader_vae, test_loader_vae, epochs=15)
    models['β-VAE'] = beta_vae
    
    return models, test_loader, device


def analyze_single_model(model, model_name, test_loader, device):
    """Comprehensive analysis of a single model"""
    print(f"\n🔬 Analyzing {model_name} latent space...")
    
    # Create analyzer
    analyzer = LatentSpaceAnalyzer(model, device)
    
    # Extract latent representations
    analyzer.extract_latent_representations(test_loader, max_samples=2000)
    
    # Perform individual analyses
    print("📊 Computing basic statistics...")
    basic_stats = analyzer.basic_statistics()
    
    print("📐 Analyzing dimensionality...")
    dim_analysis = analyzer.dimensionality_analysis()
    
    print("🎯 Performing clustering analysis...")
    clustering_results = analyzer.clustering_analysis()
    
    print("🔀 Analyzing disentanglement...")
    disentanglement_results = analyzer.disentanglement_analysis()
    
    print("🌈 Analyzing interpolation properties...")
    interpolation_results = analyzer.interpolation_analysis(num_pairs=5, num_steps=8)
    
    print("🏘️ Analyzing neighborhood structure...")
    neighborhood_results = analyzer.neighborhood_analysis(k=10)
    
    print("🌀 Analyzing manifold structure...")
    manifold_results = analyzer.manifold_analysis()
    
    # Generate comprehensive report
    print("📋 Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    return analyzer, report


def demonstrate_advanced_analysis(analyzer, model_name):
    """Demonstrate advanced analysis capabilities"""
    print(f"\n🚀 Advanced Analysis for {model_name}")
    print("="*50)
    
    # 1. Dimension importance ranking
    print("\n1️⃣ Dimension Importance Analysis:")
    disentanglement = analyzer.disentanglement_analysis()
    mi_scores = disentanglement['mutual_information_scores']
    
    # Rank dimensions by importance
    dim_rankings = np.argsort(mi_scores)[::-1]
    print("Top 5 most informative dimensions:")
    for i, dim_idx in enumerate(dim_rankings[:5]):
        print(f"   {i+1}. Dimension {dim_idx}: MI = {mi_scores[dim_idx]:.4f}")
    
    # 2. Latent space quality metrics
    print("\n2️⃣ Latent Space Quality Metrics:")
    dim_analysis = analyzer.dimensionality_analysis()
    neighborhood = analyzer.neighborhood_analysis()
    
    quality_metrics = {
        'Dimension Utilization': dim_analysis['utilization_ratio'],
        'Effective Rank (normalized)': dim_analysis['effective_rank'] / analyzer.metadata['latent_dim'],
        'Independence Score': disentanglement['independence_score'],
        'Neighborhood Purity': neighborhood['average_purity'],
        'Max Mutual Information': disentanglement['max_mutual_information']
    }
    
    for metric, value in quality_metrics.items():
        print(f"   • {metric}: {value:.3f}")
    
    # 3. Clustering performance
    print("\n3️⃣ Clustering Performance:")
    clustering = analyzer.clustering_analysis()
    
    best_method = max(clustering.items(), 
                     key=lambda x: x[1].get('silhouette_score', -1) if isinstance(x[1], dict) else -1)
    
    if isinstance(best_method[1], dict):
        print(f"   • Best clustering method: {best_method[0]}")
        print(f"   • Silhouette score: {best_method[1]['silhouette_score']:.3f}")
        print(f"   • Adjusted Rand Index: {best_method[1]['adjusted_rand_index']:.3f}")
    
    # 4. Interpolation quality
    print("\n4️⃣ Interpolation Quality:")
    interpolation = analyzer.interpolation_analysis()
    print(f"   • Average smoothness: {interpolation['average_smoothness']:.4f}")
    print(f"   • Average linearity: {interpolation['average_linearity']:.3f}")
    
    # 5. Create custom visualizations
    create_custom_visualizations(analyzer, model_name)


def create_custom_visualizations(analyzer, model_name):
    """Create custom visualization plots"""
    print(f"\n🎨 Creating custom visualizations for {model_name}...")
    
    # Get data
    latent_vectors = analyzer.latent_vectors
    labels = analyzer.labels
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Custom Latent Space Analysis', fontsize=16)
    
    # 1. Dimension activation heatmap
    ax = axes[0, 0]
    
    # Sample data for visualization
    sample_indices = np.random.choice(len(latent_vectors), min(100, len(latent_vectors)), replace=False)
    sample_latents = latent_vectors[sample_indices]
    sample_labels = labels[sample_indices]
    
    # Sort by labels for better visualization
    sort_indices = np.argsort(sample_labels)
    sorted_latents = sample_latents[sort_indices]
    sorted_labels = sample_labels[sort_indices]
    
    im = ax.imshow(sorted_latents.T, cmap='RdBu', aspect='auto')
    ax.set_xlabel('Sample (sorted by label)')
    ax.set_ylabel('Latent Dimension')
    ax.set_title('Latent Activations Heatmap')
    plt.colorbar(im, ax=ax)
    
    # 2. Per-class latent statistics
    ax = axes[0, 1]
    
    class_means = []
    class_stds = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            class_latents = latent_vectors[mask]
            class_means.append(np.mean(class_latents, axis=0))
            class_stds.append(np.std(class_latents, axis=0))
    
    if class_means:
        class_means = np.array(class_means)
        class_stds = np.array(class_stds)
        
        # Plot mean activation per class
        for i, label in enumerate(unique_labels[:5]):  # Show first 5 classes
            ax.plot(class_means[i], 'o-', label=f'Class {int(label)}', alpha=0.7)
        
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Per-Class Mean Activations')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Latent dimension correlation network
    ax = axes[0, 2]
    
    disentanglement = analyzer.disentanglement_analysis()
    corr_matrix = disentanglement['correlation_matrix']
    
    # Show only strong correlations
    threshold = 0.3
    strong_corr = np.abs(corr_matrix) > threshold
    
    # Create network-style visualization
    n_dims = corr_matrix.shape[0]
    angles = np.linspace(0, 2*np.pi, n_dims, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot nodes
    ax.scatter(x, y, s=100, c='lightblue', edgecolors='black', zorder=3)
    
    # Add dimension labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(f'{i}', (xi, yi), ha='center', va='center', fontweight='bold')
    
    # Plot edges for strong correlations
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            if strong_corr[i, j]:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'r-', 
                       alpha=abs(corr_matrix[i, j]), linewidth=2)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Dimension Correlation Network')
    ax.axis('off')
    
    # 4. Latent space density map
    ax = axes[1, 0]
    
    # Use PCA for 2D projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Create density plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(latent_2d.T)
    
    # Create grid for density estimation
    x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
    y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()
    
    xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    
    # Plot density
    im = ax.contourf(xx, yy, density, levels=20, cmap='viridis', alpha=0.8)
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, 
                        cmap='tab10', s=20, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Latent Space Density Map')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    # 5. Dimension utilization analysis
    ax = axes[1, 1]
    
    basic_stats = analyzer.basic_statistics()
    dim_stds = basic_stats['std']
    dim_means = np.abs(basic_stats['mean'])
    
    # Create utilization score (combination of std and mean)
    utilization_score = dim_stds * (1 + dim_means)
    
    bars = ax.bar(range(len(utilization_score)), utilization_score, alpha=0.7)
    
    # Color bars by utilization level
    colors = ['red' if score < 0.1 else 'orange' if score < 0.5 else 'green' 
              for score in utilization_score]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Utilization Score')
    ax.set_title('Dimension Utilization Analysis')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Unused (<0.1)'),
        Patch(facecolor='orange', label='Low (0.1-0.5)'),
        Patch(facecolor='green', label='Active (>0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 6. Interpolation smoothness analysis
    ax = axes[1, 2]
    
    interpolation = analyzer.interpolation_analysis(num_pairs=10)
    smoothness_scores = [r['smoothness'] for r in interpolation['interpolation_results']]
    linearity_scores = [r['linearity_score'] for r in interpolation['interpolation_results']]
    
    # Scatter plot of smoothness vs linearity
    scatter = ax.scatter(smoothness_scores, linearity_scores, 
                        s=100, alpha=0.7, c='purple', edgecolors='black')
    
    # Add trend line
    if len(smoothness_scores) > 1:
        z = np.polyfit(smoothness_scores, linearity_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(smoothness_scores), max(smoothness_scores), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Interpolation Smoothness')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Interpolation Quality Analysis')
    ax.grid(True, alpha=0.3)
    
    # Add quality quadrants
    if smoothness_scores and linearity_scores:
        med_smooth = np.median(smoothness_scores)
        med_linear = np.median(linearity_scores)
        ax.axvline(med_smooth, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(med_linear, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 Custom Analysis Summary for {model_name}:")
    print(f"   • Active dimensions: {np.sum(utilization_score > 0.1)}/{len(utilization_score)}")
    print(f"   • Average interpolation smoothness: {np.mean(smoothness_scores):.4f}")
    print(f"   • Average linearity: {np.mean(linearity_scores):.3f}")


def compare_multiple_models(models, test_loader, device):
    """Compare multiple models comprehensively"""
    print("\n⚖️ Comprehensive Model Comparison")
    print("="*50)
    
    # Quick comparison using built-in function
    comparison_results = compare_latent_spaces(models, test_loader, device, max_samples=1500)
    
    # Detailed comparison table
    print("\n📊 Detailed Comparison Table:")
    print("-" * 80)
    print(f"{'Model':<12} {'Eff.Rank':<10} {'Util.%':<8} {'Indep.':<8} {'Max MI':<8} {'Quality':<8}")
    print("-" * 80)
    
    for name, results in comparison_results.items():
        eff_rank = results['dim_analysis']['effective_rank']
        util_ratio = results['dim_analysis']['utilization_ratio'] * 100
        independence = results['disentanglement']['independence_score']
        max_mi = results['disentanglement']['max_mutual_information']
        quality = (util_ratio/100 + independence + max_mi) / 3
        
        print(f"{name:<12} {eff_rank:<10.2f} {util_ratio:<8.1f} {independence:<8.3f} "
              f"{max_mi:<8.3f} {quality:<8.3f}")
    
    print("-" * 80)
    
    return comparison_results


def main():
    """Main execution example"""
    print("🔬 Comprehensive Latent Space Analysis Example")
    print("="*60)
    
    # Check if we should load pre-trained models or train new ones
    import os
    
    if os.path.exists('sample_models.pth'):
        print("📦 Loading pre-trained models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('sample_models.pth', map_location=device)
        models = checkpoint['models']
        _, test_loader = get_mnist_loaders(batch_size=128, train_size=1000, test_size=1000)
    else:
        print("🏋️ Training new models...")
        models, test_loader, device = train_sample_models()
        
        # Save models for future use
        checkpoint = {
            'models': models,
            'model_configs': {
                'Vanilla AE': {'type': 'vanilla', 'latent_dim': 20},
                'VAE': {'type': 'vae', 'latent_dim': 20},
                'β-VAE': {'type': 'beta_vae', 'latent_dim': 20, 'beta': 4.0}
            }
        }
        torch.save(checkpoint, 'sample_models.pth')
        print("💾 Models saved to 'sample_models.pth'")
    
    # Analyze each model individually
    analyzers = {}
    reports = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔍 INDIVIDUAL ANALYSIS: {model_name}")
        print(f"{'='*60}")
        
        analyzer, report = analyze_single_model(model, model_name, test_loader, device)
        analyzers[model_name] = analyzer
        reports[model_name] = report
        
        # Demonstrate advanced analysis
        demonstrate_advanced_analysis(analyzer, model_name)
    
    # Compare all models
    print(f"\n{'='*60}")
    print("🏆 MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_results = compare_multiple_models(models, test_loader, device)
    
    # Final recommendations
    print(f"\n🎯 Final Recommendations:")
    
    # Find best model for different criteria
    best_utilization = max(comparison_results.items(), 
                          key=lambda x: x[1]['dim_analysis']['utilization_ratio'])
    
    best_independence = max(comparison_results.items(),
                           key=lambda x: x[1]['disentanglement']['independence_score'])
    
    best_mi = max(comparison_results.items(),
                  key=lambda x: x[1]['disentanglement']['max_mutual_information'])
    
    print(f"   • Best dimension utilization: {best_utilization[0]} "
          f"({best_utilization[1]['dim_analysis']['utilization_ratio']:.1%})")
    
    print(f"   • Best independence: {best_independence[0]} "
          f"({best_independence[1]['disentanglement']['independence_score']:.3f})")
    
    print(f"   • Most informative: {best_mi[0]} "
          f"(MI: {best_mi[1]['disentanglement']['max_mutual_information']:.3f})")
    
    print(f"\n✨ Analysis completed! Check the generated visualizations and reports.")
    print(f"💡 Use these insights to improve your autoencoder architectures and training.")


if __name__ == "__main__":
    main()