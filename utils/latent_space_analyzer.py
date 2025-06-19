"""
Latent Space Analyzer
====================

Comprehensive analysis tools for autoencoder latent spaces
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, pearsonr, spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class LatentSpaceAnalyzer:
    """
    Comprehensive latent space analysis toolkit
    
    Args:
        model: Trained autoencoder model
        device: Device for computations
        extract_method: Method to extract latent representations ('encode', 'forward')
    """
    
    def __init__(self, model, device='cpu', extract_method='encode'):
        self.model = model
        self.device = device
        self.extract_method = extract_method
        self.model.eval()
        
        # Storage for extracted representations
        self.latent_vectors = None
        self.labels = None
        self.original_data = None
        self.metadata = {}
    
    def extract_latent_representations(self, data_loader, max_samples=5000):
        """
        Extract latent representations from data loader
        
        Args:
            data_loader: PyTorch DataLoader
            max_samples: Maximum number of samples to extract
        """
        print(f"🔍 Extracting latent representations...")
        
        latent_vectors = []
        labels = []
        original_data = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if len(batch) == 2:
                    data, label = batch
                else:
                    data = batch
                    label = torch.zeros(data.size(0))
                
                data = data.to(self.device)
                
                # Extract latent representation
                if hasattr(self.model, 'encode'):
                    if hasattr(self.model, 'reparameterize'):  # VAE
                        mu, logvar = self.model.encode(data.view(data.size(0), -1))
                        latent = mu  # Use mean for analysis
                    else:  # Regular autoencoder
                        latent = self.model.encode(data.view(data.size(0), -1))
                elif self.extract_method == 'forward':
                    output = self.model(data)
                    if isinstance(output, tuple):
                        latent = output[1] if len(output) > 1 else output[0]
                    else:
                        continue
                else:
                    raise ValueError("Model must have 'encode' method or use 'forward' extract_method")
                
                latent_vectors.append(latent.cpu().numpy())
                labels.append(label.numpy())
                original_data.append(data.cpu().numpy())
                
                if (i + 1) * data.size(0) >= max_samples:
                    break
        
        # Concatenate all data
        self.latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
        self.labels = np.concatenate(labels, axis=0)[:max_samples]
        self.original_data = np.concatenate(original_data, axis=0)[:max_samples]
        
        # Store metadata
        self.metadata = {
            'num_samples': len(self.latent_vectors),
            'latent_dim': self.latent_vectors.shape[1],
            'num_classes': len(np.unique(self.labels)),
            'data_shape': self.original_data.shape[1:]
        }
        
        print(f"✅ Extracted {self.metadata['num_samples']} samples")
        print(f"📊 Latent dimension: {self.metadata['latent_dim']}")
        print(f"🏷️ Number of classes: {self.metadata['num_classes']}")
        
        return self.latent_vectors, self.labels
    
    def basic_statistics(self):
        """Compute basic statistics of the latent space"""
        if self.latent_vectors is None:
            raise ValueError("Extract latent representations first!")
        
        stats = {
            'mean': np.mean(self.latent_vectors, axis=0),
            'std': np.std(self.latent_vectors, axis=0),
            'min': np.min(self.latent_vectors, axis=0),
            'max': np.max(self.latent_vectors, axis=0),
            'median': np.median(self.latent_vectors, axis=0),
            'skewness': self._compute_skewness(self.latent_vectors),
            'kurtosis': self._compute_kurtosis(self.latent_vectors)
        }
        
        # Overall statistics
        stats['overall_mean'] = np.mean(self.latent_vectors)
        stats['overall_std'] = np.std(self.latent_vectors)
        stats['sparsity'] = np.mean(np.abs(self.latent_vectors) < 0.1)
        stats['active_dimensions'] = np.sum(stats['std'] > 0.01)
        
        return stats
    
    def _compute_skewness(self, data):
        """Compute skewness for each dimension"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.mean(((data - mean) / (std + 1e-8)) ** 3, axis=0)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis for each dimension"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.mean(((data - mean) / (std + 1e-8)) ** 4, axis=0) - 3
    
    def dimensionality_analysis(self):
        """Analyze effective dimensionality of latent space"""
        print("📐 Analyzing latent space dimensionality...")
        
        # PCA analysis
        pca = PCA()
        pca.fit(self.latent_vectors)
        
        # Compute intrinsic dimensionality metrics
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Effective rank (participation ratio)
        eigenvalues = pca.explained_variance_
        effective_rank = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        # 90% and 95% variance thresholds
        dim_90 = np.argmax(cumulative_variance >= 0.90) + 1
        dim_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Dimension utilization (non-zero variance dimensions)
        utilized_dims = np.sum(np.var(self.latent_vectors, axis=0) > 1e-6)
        
        results = {
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance,
            'effective_rank': effective_rank,
            'intrinsic_dim_90': dim_90,
            'intrinsic_dim_95': dim_95,
            'utilized_dimensions': utilized_dims,
            'total_dimensions': self.latent_vectors.shape[1],
            'utilization_ratio': utilized_dims / self.latent_vectors.shape[1]
        }
        
        return results
    
    def clustering_analysis(self):
        """Perform clustering analysis on latent space"""
        print("🎯 Performing clustering analysis...")
        
        results = {}
        
        # Try different clustering algorithms
        clusterers = {
            'KMeans': KMeans(n_clusters=self.metadata['num_classes'], random_state=42),
            'KMeans_optimal': None,  # Will find optimal k
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(n_clusters=self.metadata['num_classes'])
        }
        
        # Find optimal k for KMeans
        k_range = range(2, min(20, self.metadata['num_samples'] // 50))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.latent_vectors)
            score = silhouette_score(self.latent_vectors, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        clusterers['KMeans_optimal'] = KMeans(n_clusters=optimal_k, random_state=42)
        
        # Perform clustering
        for name, clusterer in clusterers.items():
            if clusterer is not None:
                cluster_labels = clusterer.fit_predict(self.latent_vectors)
                
                # Compute metrics
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(self.latent_vectors, cluster_labels)
                    ari = adjusted_rand_score(self.labels, cluster_labels)
                else:
                    silhouette = -1
                    ari = 0
                
                results[name] = {
                    'labels': cluster_labels,
                    'n_clusters': len(np.unique(cluster_labels)),
                    'silhouette_score': silhouette,
                    'adjusted_rand_index': ari,
                    'clusterer': clusterer
                }
        
        results['optimal_k'] = optimal_k
        results['k_silhouette_scores'] = silhouette_scores
        
        return results
    
    def disentanglement_analysis(self):
        """Analyze disentanglement properties of latent space"""
        print("🔀 Analyzing disentanglement properties...")
        
        # Mutual Information between latent dimensions and labels
        mi_scores = []
        for dim in range(self.latent_vectors.shape[1]):
            # Use regression R² as proxy for mutual information
            X = self.latent_vectors[:, dim].reshape(-1, 1)
            y = self.labels
            
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            mi_scores.append(max(0, r2))
        
        # Classification accuracy per dimension
        classification_scores = []
        for dim in range(self.latent_vectors.shape[1]):
            X = self.latent_vectors[:, dim].reshape(-1, 1)
            y = self.labels.astype(int)
            
            if len(np.unique(y)) > 1:
                clf = LogisticRegression(random_state=42, max_iter=200)
                clf.fit(X, y)
                acc = clf.score(X, y)
                classification_scores.append(acc)
            else:
                classification_scores.append(0)
        
        # Compute correlation matrix between latent dimensions
        correlation_matrix = np.corrcoef(self.latent_vectors.T)
        
        # Disentanglement metrics
        most_informative_dim = np.argmax(mi_scores)
        max_mutual_info = np.max(mi_scores)
        avg_mutual_info = np.mean(mi_scores)
        
        # Independence measure (average absolute correlation)
        off_diagonal_corr = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        independence_score = 1 - np.mean(np.abs(off_diagonal_corr))
        
        results = {
            'mutual_information_scores': mi_scores,
            'classification_scores': classification_scores,
            'correlation_matrix': correlation_matrix,
            'most_informative_dimension': most_informative_dim,
            'max_mutual_information': max_mutual_info,
            'average_mutual_information': avg_mutual_info,
            'independence_score': independence_score,
            'dimension_rankings': np.argsort(mi_scores)[::-1]
        }
        
        return results
    
    def interpolation_analysis(self, num_pairs=5, num_steps=10):
        """Analyze interpolation properties in latent space"""
        print("🌈 Analyzing interpolation properties...")
        
        # Select random pairs
        indices = np.random.choice(len(self.latent_vectors), num_pairs * 2, replace=False)
        pairs = indices.reshape(num_pairs, 2)
        
        interpolation_results = []
        
        for i, (idx1, idx2) in enumerate(pairs):
            z1 = self.latent_vectors[idx1]
            z2 = self.latent_vectors[idx2]
            
            # Linear interpolation
            alphas = np.linspace(0, 1, num_steps)
            interpolated_latents = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_latents.append(z_interp)
            
            interpolated_latents = np.array(interpolated_latents)
            
            # Analyze interpolation smoothness
            differences = np.diff(interpolated_latents, axis=0)
            smoothness = np.mean(np.linalg.norm(differences, axis=1))
            
            # Distance in latent space vs interpolation parameter
            distances_from_z1 = [np.linalg.norm(z - z1) for z in interpolated_latents]
            linearity_score = np.corrcoef(alphas, distances_from_z1)[0, 1]
            
            interpolation_results.append({
                'pair_index': i,
                'start_idx': idx1,
                'end_idx': idx2,
                'interpolated_latents': interpolated_latents,
                'smoothness': smoothness,
                'linearity_score': linearity_score,
                'alphas': alphas
            })
        
        # Overall interpolation metrics
        avg_smoothness = np.mean([r['smoothness'] for r in interpolation_results])
        avg_linearity = np.mean([r['linearity_score'] for r in interpolation_results])
        
        return {
            'interpolation_results': interpolation_results,
            'average_smoothness': avg_smoothness,
            'average_linearity': avg_linearity,
            'num_pairs_analyzed': num_pairs
        }
    
    def neighborhood_analysis(self, k=10):
        """Analyze local neighborhood structure in latent space"""
        print("🏘️ Analyzing neighborhood structure...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(self.latent_vectors)
        distances, indices = nbrs.kneighbors(self.latent_vectors)
        
        # Analyze neighborhood purity (same class neighbors)
        neighborhood_purity = []
        for i in range(len(self.latent_vectors)):
            neighbor_labels = self.labels[indices[i][1:]]  # Exclude self
            same_class = np.sum(neighbor_labels == self.labels[i])
            purity = same_class / k
            neighborhood_purity.append(purity)
        
        # Analyze distance distributions
        neighbor_distances = distances[:, 1:]  # Exclude self-distance
        avg_neighbor_distance = np.mean(neighbor_distances, axis=1)
        
        # Local density estimation
        local_density = 1 / (np.mean(neighbor_distances, axis=1) + 1e-8)
        
        # Continuity and trustworthiness metrics
        # (Simplified versions)
        continuity_errors = []
        for i in range(min(1000, len(self.latent_vectors))):  # Sample for efficiency
            # Find neighbors in original space if available
            if hasattr(self, 'original_data') and self.original_data is not None:
                orig_nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.original_data.reshape(len(self.original_data), -1))
                _, orig_indices = orig_nbrs.kneighbors(self.original_data[i:i+1].reshape(1, -1))
                
                # Check how many latent neighbors are also original neighbors
                latent_neighbors = set(indices[i][1:])
                original_neighbors = set(orig_indices[0][1:])
                overlap = len(latent_neighbors.intersection(original_neighbors))
                continuity_errors.append(1 - overlap / k)
        
        results = {
            'k': k,
            'neighborhood_indices': indices,
            'neighborhood_distances': distances,
            'neighborhood_purity': neighborhood_purity,
            'average_purity': np.mean(neighborhood_purity),
            'average_neighbor_distance': avg_neighbor_distance,
            'local_density': local_density,
            'continuity_errors': continuity_errors if continuity_errors else None,
            'average_continuity_error': np.mean(continuity_errors) if continuity_errors else None
        }
        
        return results
    
    def manifold_analysis(self):
        """Analyze manifold structure of latent space"""
        print("🌀 Analyzing manifold structure...")
        
        # Apply different manifold learning techniques
        manifold_methods = {}
        
        # t-SNE
        if len(self.latent_vectors) <= 10000:  # t-SNE is slow for large datasets
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_embedding = tsne.fit_transform(self.latent_vectors[:5000])
            manifold_methods['t-SNE'] = tsne_embedding
        
        # PCA
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(self.latent_vectors)
        manifold_methods['PCA'] = pca_embedding
        
        # MDS
        if len(self.latent_vectors) <= 5000:  # MDS is computationally expensive
            mds = MDS(n_components=2, random_state=42)
            mds_embedding = mds.fit_transform(self.latent_vectors[:2000])
            manifold_methods['MDS'] = mds_embedding
        
        # ICA
        ica = FastICA(n_components=min(2, self.latent_vectors.shape[1]), random_state=42)
        ica_embedding = ica.fit_transform(self.latent_vectors)
        manifold_methods['ICA'] = ica_embedding
        
        # Estimate intrinsic dimensionality using correlation dimension
        intrinsic_dim = self._estimate_correlation_dimension()
        
        results = {
            'manifold_embeddings': manifold_methods,
            'estimated_intrinsic_dimension': intrinsic_dim,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
        
        return results
    
    def _estimate_correlation_dimension(self, max_points=2000):
        """Estimate intrinsic dimensionality using correlation dimension"""
        if len(self.latent_vectors) > max_points:
            indices = np.random.choice(len(self.latent_vectors), max_points, replace=False)
            sample_data = self.latent_vectors[indices]
        else:
            sample_data = self.latent_vectors
        
        # Compute pairwise distances
        distances = pdist(sample_data)
        
        # Range of radius values
        r_values = np.logspace(-2, 1, 20)
        
        # Count pairs within each radius
        counts = []
        for r in r_values:
            count = np.sum(distances < r)
            counts.append(count / len(distances))
        
        # Fit line to log-log plot to estimate dimension
        log_r = np.log(r_values[1:])  # Avoid log(0)
        log_counts = np.log(np.array(counts[1:]) + 1e-10)
        
        # Remove infinite values
        valid_mask = np.isfinite(log_r) & np.isfinite(log_counts)
        if np.sum(valid_mask) > 2:
            slope, _ = np.polyfit(log_r[valid_mask], log_counts[valid_mask], 1)
            estimated_dim = max(0, slope)
        else:
            estimated_dim = self.latent_vectors.shape[1]
        
        return estimated_dim
    
    def generate_comprehensive_report(self, save_path=None):
        """Generate comprehensive latent space analysis report"""
        print("📊 Generating comprehensive analysis report...")
        
        if self.latent_vectors is None:
            raise ValueError("Extract latent representations first!")
        
        # Perform all analyses
        basic_stats = self.basic_statistics()
        dim_analysis = self.dimensionality_analysis()
        clustering_results = self.clustering_analysis()
        disentanglement_results = self.disentanglement_analysis()
        interpolation_results = self.interpolation_analysis()
        neighborhood_results = self.neighborhood_analysis()
        manifold_results = self.manifold_analysis()
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(
            basic_stats, dim_analysis, clustering_results, 
            disentanglement_results, interpolation_results,
            neighborhood_results, manifold_results, save_path
        )
        
        # Create summary report
        report = {
            'metadata': self.metadata,
            'basic_statistics': basic_stats,
            'dimensionality_analysis': dim_analysis,
            'clustering_analysis': clustering_results,
            'disentanglement_analysis': disentanglement_results,
            'interpolation_analysis': interpolation_results,
            'neighborhood_analysis': neighborhood_results,
            'manifold_analysis': manifold_results
        }
        
        # Print summary
        self._print_summary_report(report)
        
        return report
    
    def _create_comprehensive_visualization(self, basic_stats, dim_analysis, clustering_results, 
                                          disentanglement_results, interpolation_results,
                                          neighborhood_results, manifold_results, save_path):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Basic statistics
        ax1 = plt.subplot(4, 3, 1)
        dims = range(len(basic_stats['mean']))
        plt.plot(dims, basic_stats['mean'], 'o-', label='Mean', alpha=0.7)
        plt.plot(dims, basic_stats['std'], 's-', label='Std', alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Value')
        plt.title('Basic Statistics per Dimension')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. PCA explained variance
        ax2 = plt.subplot(4, 3, 2)
        plt.plot(dim_analysis['explained_variance_ratio'], 'o-', alpha=0.7)
        plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Cumulative variance
        ax3 = plt.subplot(4, 3, 3)
        plt.plot(dim_analysis['cumulative_variance'], 'o-', alpha=0.7)
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
        plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Correlation matrix
        ax4 = plt.subplot(4, 3, 4)
        corr_matrix = disentanglement_results['correlation_matrix']
        im = plt.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        plt.title('Latent Dimension Correlations')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Latent Dimension')
        plt.colorbar(im, ax=ax4)
        
        # 5. Mutual information scores
        ax5 = plt.subplot(4, 3, 5)
        mi_scores = disentanglement_results['mutual_information_scores']
        plt.bar(range(len(mi_scores)), mi_scores, alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Mutual Information (R²)')
        plt.title('Disentanglement: MI with Labels')
        plt.grid(True, alpha=0.3)
        
        # 6. Clustering results
        ax6 = plt.subplot(4, 3, 6)
        if 'PCA' in manifold_results['manifold_embeddings']:
            pca_embedding = manifold_results['manifold_embeddings']['PCA']
            best_clustering = max(clustering_results.items(), 
                                key=lambda x: x[1].get('silhouette_score', -1) if isinstance(x[1], dict) else -1)
            if isinstance(best_clustering[1], dict):
                cluster_labels = best_clustering[1]['labels']
                scatter = plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                                    c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
                plt.title(f'Clustering Results ({best_clustering[0]})')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.colorbar(scatter, ax=ax6)
        
        # 7. Neighborhood purity
        ax7 = plt.subplot(4, 3, 7)
        purity = neighborhood_results['neighborhood_purity']
        plt.hist(purity, bins=30, alpha=0.7, density=True)
        plt.axvline(np.mean(purity), color='r', linestyle='--', label=f'Mean: {np.mean(purity):.3f}')
        plt.xlabel('Neighborhood Purity')
        plt.ylabel('Density')
        plt.title('Neighborhood Purity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Local density
        ax8 = plt.subplot(4, 3, 8)
        density = neighborhood_results['local_density']
        plt.hist(density, bins=30, alpha=0.7, density=True)
        plt.xlabel('Local Density')
        plt.ylabel('Frequency')
        plt.title('Local Density Distribution')
        plt.grid(True, alpha=0.3)
        
        # 9. t-SNE or PCA visualization with true labels
        ax9 = plt.subplot(4, 3, 9)
        if 't-SNE' in manifold_results['manifold_embeddings']:
            embedding = manifold_results['manifold_embeddings']['t-SNE']
            title = 't-SNE Visualization'
        else:
            embedding = manifold_results['manifold_embeddings']['PCA']
            title = 'PCA Visualization'
        
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=self.labels[:len(embedding)], cmap='tab10', alpha=0.7, s=20)
        plt.title(f'{title} (True Labels)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, ax=ax9)
        
        # 10. Interpolation smoothness
        ax10 = plt.subplot(4, 3, 10)
        smoothness_scores = [r['smoothness'] for r in interpolation_results['interpolation_results']]
        linearity_scores = [r['linearity_score'] for r in interpolation_results['interpolation_results']]
        plt.scatter(smoothness_scores, linearity_scores, alpha=0.7)
        plt.xlabel('Interpolation Smoothness')
        plt.ylabel('Linearity Score')
        plt.title('Interpolation Quality')
        plt.grid(True, alpha=0.3)
        
        # 11. Dimension utilization
        ax11 = plt.subplot(4, 3, 11)
        utilization = basic_stats['std'] > 0.01
        plt.bar(['Utilized', 'Unused'], 
                [np.sum(utilization), np.sum(~utilization)], 
                alpha=0.7, color=['green', 'red'])
        plt.title('Dimension Utilization')
        plt.ylabel('Number of Dimensions')
        
        # 12. Overall summary metrics
        ax12 = plt.subplot(4, 3, 12)
        metrics = {
            'Effective Rank': dim_analysis['effective_rank'] / self.metadata['latent_dim'],
            'Independence': disentanglement_results['independence_score'],
            'Avg Purity': np.mean(neighborhood_results['neighborhood_purity']),
            'Utilization': dim_analysis['utilization_ratio'],
            'Max MI': disentanglement_results['max_mutual_information']
        }
        
        bars = plt.bar(range(len(metrics)), list(metrics.values()), alpha=0.7)
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
        plt.ylabel('Score (0-1)')
        plt.title('Summary Metrics')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _print_summary_report(self, report):
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE LATENT SPACE ANALYSIS REPORT")
        print("="*80)
        
        # Basic Info
        print(f"\n📋 Dataset Information:")
        print(f"   • Samples analyzed: {report['metadata']['num_samples']:,}")
        print(f"   • Latent dimensions: {report['metadata']['latent_dim']}")
        print(f"   • Number of classes: {report['metadata']['num_classes']}")
        print(f"   • Data shape: {report['metadata']['data_shape']}")
        
        # Dimensionality Analysis
        dim_analysis = report['dimensionality_analysis']
        print(f"\n📐 Dimensionality Analysis:")
        print(f"   • Effective rank: {dim_analysis['effective_rank']:.2f}")
        print(f"   • Utilized dimensions: {dim_analysis['utilized_dimensions']}/{dim_analysis['total_dimensions']}")
        print(f"   • Utilization ratio: {dim_analysis['utilization_ratio']:.1%}")
        print(f"   • Dimensions for 90% variance: {dim_analysis['intrinsic_dim_90']}")
        print(f"   • Dimensions for 95% variance: {dim_analysis['intrinsic_dim_95']}")
        
        # Basic Statistics
        basic_stats = report['basic_statistics']
        print(f"\n📊 Basic Statistics:")
        print(f"   • Overall mean: {basic_stats['overall_mean']:.4f}")
        print(f"   • Overall std: {basic_stats['overall_std']:.4f}")
        print(f"   • Sparsity ratio: {basic_stats['sparsity']:.1%}")
        print(f"   • Active dimensions: {basic_stats['active_dimensions']}")
        
        # Clustering Analysis
        clustering = report['clustering_analysis']
        best_clustering = max(clustering.items(), 
                            key=lambda x: x[1].get('silhouette_score', -1) if isinstance(x[1], dict) else -1)
        if isinstance(best_clustering[1], dict):
            print(f"\n🎯 Clustering Analysis:")
            print(f"   • Best method: {best_clustering[0]}")
            print(f"   • Silhouette score: {best_clustering[1]['silhouette_score']:.3f}")
            print(f"   • Adjusted Rand Index: {best_clustering[1]['adjusted_rand_index']:.3f}")
            print(f"   • Optimal k (KMeans): {clustering['optimal_k']}")
        
        # Disentanglement Analysis
        disentanglement = report['disentanglement_analysis']
        print(f"\n🔀 Disentanglement Analysis:")
        print(f"   • Most informative dimension: {disentanglement['most_informative_dimension']}")
        print(f"   • Max mutual information: {disentanglement['max_mutual_information']:.3f}")
        print(f"   • Average mutual information: {disentanglement['average_mutual_information']:.3f}")
        print(f"   • Independence score: {disentanglement['independence_score']:.3f}")
        
        # Neighborhood Analysis
        neighborhood = report['neighborhood_analysis']
        print(f"\n🏘️ Neighborhood Analysis:")
        print(f"   • Average neighborhood purity: {neighborhood['average_purity']:.3f}")
        print(f"   • k-neighbors used: {neighborhood['k']}")
        if neighborhood['average_continuity_error'] is not None:
            print(f"   • Average continuity error: {neighborhood['average_continuity_error']:.3f}")
        
        # Interpolation Analysis
        interpolation = report['interpolation_analysis']
        print(f"\n🌈 Interpolation Analysis:")
        print(f"   • Average smoothness: {interpolation['average_smoothness']:.4f}")
        print(f"   • Average linearity: {interpolation['average_linearity']:.3f}")
        print(f"   • Pairs analyzed: {interpolation['num_pairs_analyzed']}")
        
        # Manifold Analysis
        manifold = report['manifold_analysis']
        print(f"\n🌀 Manifold Analysis:")
        print(f"   • Estimated intrinsic dimension: {manifold['estimated_intrinsic_dimension']:.2f}")
        print(f"   • Available embeddings: {list(manifold['manifold_embeddings'].keys())}")
        
        # Overall Assessment
        print(f"\n🎯 Overall Assessment:")
        self._provide_latent_space_assessment(report)
        
        print("="*80)
    
    def _provide_latent_space_assessment(self, report):
        """Provide qualitative assessment of latent space quality"""
        dim_analysis = report['dimensionality_analysis']
        disentanglement = report['disentanglement_analysis']
        neighborhood = report['neighborhood_analysis']
        
        # Quality scores (0-1)
        utilization_score = dim_analysis['utilization_ratio']
        disentanglement_score = disentanglement['independence_score']
        purity_score = neighborhood['average_purity']
        
        overall_score = (utilization_score + disentanglement_score + purity_score) / 3
        
        if overall_score >= 0.8:
            quality = "Excellent 🌟"
        elif overall_score >= 0.6:
            quality = "Good 👍"
        elif overall_score >= 0.4:
            quality = "Fair 👌"
        else:
            quality = "Poor 👎"
        
        print(f"   • Overall quality: {quality} (Score: {overall_score:.3f})")
        
        # Specific recommendations
        print(f"   • Recommendations:")
        
        if utilization_score < 0.5:
            print(f"     - Consider reducing latent dimensions (low utilization: {utilization_score:.1%})")
        
        if disentanglement_score < 0.5:
            print(f"     - Latent dimensions are highly correlated (independence: {disentanglement_score:.3f})")
            print(f"     - Consider using β-VAE or other disentanglement methods")
        
        if purity_score < 0.5:
            print(f"     - Poor neighborhood structure (purity: {purity_score:.3f})")
            print(f"     - Model may need more training or different architecture")
        
        if dim_analysis['effective_rank'] < dim_analysis['total_dimensions'] * 0.3:
            print(f"     - Low effective rank suggests redundant dimensions")
        
        if disentanglement['max_mutual_information'] < 0.1:
            print(f"     - Weak relationship between latent dims and labels")
            print(f"     - Consider supervised or semi-supervised training")


def compare_latent_spaces(models_dict, data_loader, device='cpu', max_samples=2000):
    """
    Compare latent spaces of multiple models
    
    Args:
        models_dict: Dictionary of {model_name: model} pairs
        data_loader: DataLoader for analysis
        device: Device for computations
        max_samples: Maximum samples to analyze
    """
    print("🔍 Comparing latent spaces across models...")
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n📊 Analyzing {name}...")
        analyzer = LatentSpaceAnalyzer(model, device)
        analyzer.extract_latent_representations(data_loader, max_samples)
        
        # Quick analysis
        basic_stats = analyzer.basic_statistics()
        dim_analysis = analyzer.dimensionality_analysis()
        disentanglement = analyzer.disentanglement_analysis()
        
        results[name] = {
            'analyzer': analyzer,
            'basic_stats': basic_stats,
            'dim_analysis': dim_analysis,
            'disentanglement': disentanglement
        }
    
    # Create comparison visualization
    _create_comparison_visualization(results)
    
    return results


def _create_comparison_visualization(results):
    """Create comparison visualization for multiple models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(results.keys())
    
    # 1. Effective Rank Comparison
    ax = axes[0, 0]
    effective_ranks = [results[name]['dim_analysis']['effective_rank'] for name in model_names]
    latent_dims = [results[name]['dim_analysis']['total_dimensions'] for name in model_names]
    normalized_ranks = [er / ld for er, ld in zip(effective_ranks, latent_dims)]
    
    bars = ax.bar(model_names, normalized_ranks, alpha=0.7)
    ax.set_ylabel('Normalized Effective Rank')
    ax.set_title('Effective Rank Comparison')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 2. Utilization Ratio Comparison
    ax = axes[0, 1]
    utilization_ratios = [results[name]['dim_analysis']['utilization_ratio'] for name in model_names]
    
    bars = ax.bar(model_names, utilization_ratios, alpha=0.7, color='orange')
    ax.set_ylabel('Utilization Ratio')
    ax.set_title('Dimension Utilization')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 3. Independence Score Comparison
    ax = axes[0, 2]
    independence_scores = [results[name]['disentanglement']['independence_score'] for name in model_names]
    
    bars = ax.bar(model_names, independence_scores, alpha=0.7, color='green')
    ax.set_ylabel('Independence Score')
    ax.set_title('Disentanglement Quality')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 4. Sparsity Comparison
    ax = axes[1, 0]
    sparsity_ratios = [results[name]['basic_stats']['sparsity'] for name in model_names]
    
    bars = ax.bar(model_names, sparsity_ratios, alpha=0.7, color='purple')
    ax.set_ylabel('Sparsity Ratio')
    ax.set_title('Activation Sparsity')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 5. Max Mutual Information Comparison
    ax = axes[1, 1]
    max_mi_scores = [results[name]['disentanglement']['max_mutual_information'] for name in model_names]
    
    bars = ax.bar(model_names, max_mi_scores, alpha=0.7, color='red')
    ax.set_ylabel('Max Mutual Information')
    ax.set_title('Label Informativeness')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 6. Overall Quality Score
    ax = axes[1, 2]
    quality_scores = []
    for name in model_names:
        util = results[name]['dim_analysis']['utilization_ratio']
        indep = results[name]['disentanglement']['independence_score']
        # Add more metrics as available
        quality = (util + indep) / 2
        quality_scores.append(quality)
    
    bars = ax.bar(model_names, quality_scores, alpha=0.7, color='gold')
    ax.set_ylabel('Overall Quality Score')
    ax.set_title('Overall Assessment')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Add value labels on bars
    for axes_row in axes:
        for ax in axes_row:
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*60)
    print("📊 LATENT SPACE COMPARISON SUMMARY")
    print("="*60)
    
    for i, name in enumerate(model_names):
        quality_score = quality_scores[i]
        print(f"\n{name}:")
        print(f"  • Overall Quality: {quality_score:.3f}")
        print(f"  • Effective Rank: {effective_ranks[i]:.2f}")
        print(f"  • Utilization: {utilization_ratios[i]:.1%}")
        print(f"  • Independence: {independence_scores[i]:.3f}")
        print(f"  • Max MI: {max_mi_scores[i]:.3f}")
    
    # Rank models
    best_model_idx = np.argmax(quality_scores)
    print(f"\n🏆 Best performing model: {model_names[best_model_idx]}")
    print(f"   Quality score: {quality_scores[best_model_idx]:.3f}")


def main():
    """Demo of latent space analysis"""
    print("🔍 Latent Space Analyzer Demo")
    
    # This would typically be used with trained models
    # For demo purposes, we'll create dummy data
    
    # Simulate latent representations
    np.random.seed(42)
    n_samples = 1000
    latent_dim = 20
    n_classes = 10
    
    # Create analyzer with dummy data
    class DummyAnalyzer(LatentSpaceAnalyzer):
        def __init__(self):
            super().__init__(None, 'cpu')
            
            # Generate synthetic latent vectors
            self.latent_vectors = np.random.randn(n_samples, latent_dim)
            
            # Add some structure
            for i in range(n_classes):
                mask = slice(i * n_samples // n_classes, (i + 1) * n_samples // n_classes)
                self.latent_vectors[mask, i % latent_dim] += 2  # Make some dims informative
            
            self.labels = np.repeat(range(n_classes), n_samples // n_classes)
            self.original_data = np.random.randn(n_samples, 28, 28)
            
            self.metadata = {
                'num_samples': n_samples,
                'latent_dim': latent_dim,
                'num_classes': n_classes,
                'data_shape': (28, 28)
            }
    
    # Create dummy analyzer
    analyzer = DummyAnalyzer()
    
    # Generate comprehensive report
    print("📊 Generating comprehensive analysis...")
    report = analyzer.generate_comprehensive_report()
    
    print("\n✅ Latent space analysis completed!")
    print("🔬 Use this analyzer with your trained models for detailed insights.")


if __name__ == "__main__":
    main()