"""
Attention Data Analyzer
=======================

Tool to load and analyze saved attention data
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class AttentionDataAnalyzer:
    """Analyzer for saved attention data"""
    
    def __init__(self, data_dir="attention_captures"):
        self.data_dir = data_dir
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.summary = None
        
        self.load_data()
    
    def load_data(self):
        """Load all saved attention data"""
        print(f"📂 Loading attention data from {self.data_dir}...")
        
        # Load inputs
        inputs_file = os.path.join(self.data_dir, "attention_data_inputs.pkl")
        if os.path.exists(inputs_file):
            with open(inputs_file, 'rb') as f:
                self.inputs = pickle.load(f)
            print(f"✅ Loaded {len(self.inputs)} input captures")
        
        # Load outputs
        outputs_file = os.path.join(self.data_dir, "attention_data_outputs.pkl")
        if os.path.exists(outputs_file):
            with open(outputs_file, 'rb') as f:
                self.outputs = pickle.load(f)
            print(f"✅ Loaded {len(self.outputs)} output captures")
        
        # Load weights
        weights_file = os.path.join(self.data_dir, "attention_data_weights.pkl")
        if os.path.exists(weights_file):
            with open(weights_file, 'rb') as f:
                self.weights = pickle.load(f)
            print(f"✅ Loaded {len(self.weights)} weight captures")
        
        # Load summary
        summary_file = os.path.join(self.data_dir, "attention_data_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
            print(f"✅ Loaded summary data")
    
    def create_comprehensive_report(self):
        """Create comprehensive analysis report"""
        if not all([self.inputs, self.outputs]):
            print("❌ Missing required data!")
            return
        
        print("📊 Creating comprehensive analysis report...")
        
        # Create analysis dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Layer-wise statistics
        self._plot_layer_statistics(plt.subplot(3, 4, 1))
        
        # 2. Activation distributions
        self._plot_activation_distributions(plt.subplot(3, 4, 2))
        
        # 3. Input-Output correlations
        self._plot_io_correlations(plt.subplot(3, 4, 3))
        
        # 4. Transformation magnitudes
        self._plot_transformation_magnitudes(plt.subplot(3, 4, 4))
        
        # 5. Layer depth analysis
        self._plot_depth_analysis(plt.subplot(3, 4, 5))
        
        # 6. Attention weight analysis (if available)
        if self.weights:
            self._plot_attention_weight_stats(plt.subplot(3, 4, 6))
        
        # 7. Hidden state evolution
        self._plot_hidden_state_evolution(plt.subplot(3, 4, 7))
        
        # 8. Dimensionality analysis
        self._plot_dimensionality_analysis(plt.subplot(3, 4, 8))
        
        # 9. Layer comparison heatmap
        self._plot_layer_comparison_heatmap(plt.subplot(3, 4, 9))
        
        # 10. Sequence position analysis
        self._plot_sequence_position_analysis(plt.subplot(3, 4, 10))
        
        # 11. Attention head analysis (if available)
        if self.weights:
            self._plot_attention_head_analysis(plt.subplot(3, 4, 11))
        
        # 12. Overall summary metrics
        self._plot_summary_metrics(plt.subplot(3, 4, 12))
        
        plt.tight_layout()
        
        # Save comprehensive report
        report_file = os.path.join(self.data_dir, "comprehensive_attention_analysis.png")
        plt.savefig(report_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Comprehensive report saved to {report_file}")
        
        # Generate text report
        self._generate_text_report()
    
    def _plot_layer_statistics(self, ax):
        """Plot basic layer statistics"""
        layer_names = []
        input_means = []
        output_means = []
        
        for inp, out in zip(self.inputs, self.outputs):
            layer_names.append(inp['layer_name'].split('.')[-1])
            input_means.append(inp['mean'])
            output_means.append(out['mean'])
        
        x_pos = range(len(layer_names))
        ax.bar([x - 0.2 for x in x_pos], input_means, 0.4, label='Input', alpha=0.7)
        ax.bar([x + 0.2 for x in x_pos], output_means, 0.4, label='Output', alpha=0.7)
        
        ax.set_title('Layer-wise Mean Activations')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Activation')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_activation_distributions(self, ax):
        """Plot activation distributions"""
        all_input_means = [inp['mean'] for inp in self.inputs]
        all_output_means = [out['mean'] for out in self.outputs]
        
        ax.hist(all_input_means, bins=20, alpha=0.7, label='Inputs', density=True)
        ax.hist(all_output_means, bins=20, alpha=0.7, label='Outputs', density=True)
        
        ax.set_title('Activation Distribution')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    def _plot_io_correlations(self, ax):
        """Plot input-output correlations"""
        input_means = [inp['mean'] for inp in self.inputs]
        output_means = [out['mean'] for out in self.outputs]
        
        ax.scatter(input_means, output_means, alpha=0.7)
        
        # Add correlation line
        if len(input_means) > 1:
            correlation, _ = pearsonr(input_means, output_means)
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Input-Output Correlation')
        ax.set_xlabel('Input Mean')
        ax.set_ylabel('Output Mean')
    
    def _plot_transformation_magnitudes(self, ax):
        """Plot transformation magnitudes"""
        transformations = []
        for inp, out in zip(self.inputs, self.outputs):
            transformations.append(abs(out['mean'] - inp['mean']))
        
        layer_indices = range(len(transformations))
        ax.plot(layer_indices, transformations, 'o-', linewidth=2, markersize=6)
        
        ax.set_title('Transformation Magnitude')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('|Output - Input|')
        ax.grid(True, alpha=0.3)
    
    def _plot_depth_analysis(self, ax):
        """Analyze how representations change with depth"""
        layer_depths = []
        std_values = []
        
        for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
            layer_depths.append(i)
            std_values.append(out['std'])
        
        ax.plot(layer_depths, std_values, 'o-', linewidth=2, markersize=6, color='purple')
        
        ax.set_title('Representation Diversity by Depth')
        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Output Std')
        ax.grid(True, alpha=0.3)
    
    def _plot_attention_weight_stats(self, ax):
        """Plot attention weight statistics"""
        if not self.weights:
            ax.text(0.5, 0.5, 'No attention weights\navailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Attention Weight Stats')
            return
        
        # Analyze attention weight distributions
        entropy_values = []
        sparsity_values = []
        
        for weight_data in self.weights:
            weights = weight_data['weights'].numpy()
            
            # Calculate entropy for each head
            for head in range(weights.shape[1]):
                head_weights = weights[0, head]  # First batch
                
                # Calculate entropy
                entropy = -np.sum(head_weights * np.log(head_weights + 1e-8), axis=-1)
                entropy_values.extend(entropy.mean(axis=0))
                
                # Calculate sparsity (percentage of weights < threshold)
                sparsity = np.mean(head_weights < 0.1)
                sparsity_values.append(sparsity)
        
        ax.scatter(entropy_values, sparsity_values, alpha=0.7)
        ax.set_title('Attention Weight Properties')
        ax.set_xlabel('Average Entropy')
        ax.set_ylabel('Sparsity')
    
    def _plot_hidden_state_evolution(self, ax):
        """Plot hidden state evolution"""
        # Track how hidden states evolve through layers
        norms = []
        for out in self.outputs:
            if 'attention_output' in out:
                # Calculate norm of hidden states
                hidden_states = out['attention_output']
                norm = np.linalg.norm(hidden_states.numpy(), axis=-1).mean()
                norms.append(norm)
        
        if norms:
            ax.plot(range(len(norms)), norms, 'o-', linewidth=2, markersize=6, color='green')
            ax.set_title('Hidden State Norm Evolution')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('L2 Norm')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hidden state\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hidden State Evolution')
    
    def _plot_dimensionality_analysis(self, ax):
        """Analyze effective dimensionality of representations"""
        # Collect all output representations
        all_outputs = []
        for out in self.outputs:
            if 'attention_output' in out:
                output_tensor = out['attention_output'].numpy()
                # Flatten batch and sequence dimensions
                reshaped = output_tensor.reshape(-1, output_tensor.shape[-1])
                all_outputs.append(reshaped)
        
        if all_outputs:
            # Concatenate and perform PCA
            combined_outputs = np.concatenate(all_outputs, axis=0)
            
            # Sample if too large
            if combined_outputs.shape[0] > 5000:
                indices = np.random.choice(combined_outputs.shape[0], 5000, replace=False)
                combined_outputs = combined_outputs[indices]
            
            pca = PCA()
            pca.fit(combined_outputs)
            
            # Plot explained variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', linewidth=2)
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90%')
            ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95%')
            
            ax.set_title('Effective Dimensionality')
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor PCA', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dimensionality Analysis')
    
    def _plot_layer_comparison_heatmap(self, ax):
        """Create heatmap comparing layer properties"""
        # Create comparison matrix
        layer_data = []
        layer_names = []
        
        for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
            layer_names.append(f"L{i}")
            layer_data.append([
                inp['mean'],
                inp['std'],
                out['mean'],
                out['std'],
                abs(out['mean'] - inp['mean'])  # transformation magnitude
            ])
        
        if layer_data:
            layer_matrix = np.array(layer_data)
            
            # Normalize each column
            layer_matrix_norm = (layer_matrix - layer_matrix.min(axis=0)) / (layer_matrix.max(axis=0) - layer_matrix.min(axis=0) + 1e-8)
            
            im = ax.imshow(layer_matrix_norm.T, cmap='viridis', aspect='auto')
            
            ax.set_title('Layer Property Comparison')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Property')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names)
            ax.set_yticks(range(5))
            ax.set_yticklabels(['In Mean', 'In Std', 'Out Mean', 'Out Std', 'Transform'])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_sequence_position_analysis(self, ax):
        """Analyze how attention varies by sequence position"""
        if not self.weights:
            ax.text(0.5, 0.5, 'No attention weights\nfor position analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sequence Position Analysis')
            return
        
        # Analyze attention patterns by position
        position_attention = []
        
        for weight_data in self.weights[:3]:  # First few layers
            weights = weight_data['weights'].numpy()  # [batch, heads, seq_len, seq_len]
            
            # Average across batch and heads
            avg_weights = weights.mean(axis=(0, 1))  # [seq_len, seq_len]
            
            # Sum attention received by each position
            attention_received = avg_weights.sum(axis=0)
            position_attention.append(attention_received)
        
        if position_attention:
            # Plot attention by position for different layers
            for i, attn in enumerate(position_attention):
                ax.plot(range(len(attn)), attn, 'o-', label=f'Layer {i}', alpha=0.7)
            
            ax.set_title('Attention by Sequence Position')
            ax.set_xlabel('Position')
            ax.set_ylabel('Total Attention Received')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_attention_head_analysis(self, ax):
        """Analyze attention head diversity"""
        if not self.weights:
            ax.text(0.5, 0.5, 'No attention weights\nfor head analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Attention Head Analysis')
            return
        
        # Analyze head diversity
        head_entropies = []
        layer_names = []
        
        for weight_data in self.weights:
            weights = weight_data['weights'].numpy()  # [batch, heads, seq_len, seq_len]
            layer_names.append(weight_data['layer_name'].split('.')[-1])
            
            # Calculate entropy for each head
            layer_entropies = []
            for head in range(weights.shape[1]):
                head_weights = weights[0, head]  # First batch
                
                # Calculate entropy across sequence positions
                entropy = -np.sum(head_weights * np.log(head_weights + 1e-8), axis=-1)
                layer_entropies.append(entropy.mean())
            
            head_entropies.append(layer_entropies)
        
        if head_entropies:
            # Create box plot of head entropies by layer
            ax.boxplot(head_entropies, labels=layer_names)
            ax.set_title('Attention Head Entropy Distribution')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Entropy')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_summary_metrics(self, ax):
        """Plot overall summary metrics"""
        metrics = {}
        
        # Calculate summary metrics
        if self.inputs and self.outputs:
            # Average transformation magnitude
            transformations = [abs(out['mean'] - inp['mean']) for inp, out in zip(self.inputs, self.outputs)]
            metrics['Avg Transform'] = np.mean(transformations)
            
            # Layer diversity
            output_stds = [out['std'] for out in self.outputs]
            metrics['Representation Diversity'] = np.mean(output_stds)
            
            # Input-output correlation
            input_means = [inp['mean'] for inp in self.inputs]
            output_means = [out['mean'] for out in self.outputs]
            if len(input_means) > 1:
                corr, _ = pearsonr(input_means, output_means)
                metrics['I/O Correlation'] = abs(corr)
        
        if self.weights:
            # Attention sparsity
            all_sparsity = []
            for weight_data in self.weights:
                weights = weight_data['weights'].numpy()
                sparsity = np.mean(weights < 0.1)
                all_sparsity.append(sparsity)
            metrics['Attention Sparsity'] = np.mean(all_sparsity)
        
        # Create bar plot
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.7, color='skyblue')
            ax.set_title('Summary Metrics')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _generate_text_report(self):
        """Generate detailed text report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE ATTENTION ANALYSIS REPORT")
        report_lines.append("="*80)
        
        # Basic statistics
        if self.summary:
            report_lines.append(f"\n📊 BASIC STATISTICS:")
            report_lines.append(f"   • Total layers analyzed: {self.summary['num_layers']}")
            report_lines.append(f"   • Total attention captures: {self.summary['total_captures']}")
            report_lines.append(f"   • Unique layer names: {len(self.summary['unique_layers'])}")
        
        # Layer analysis
        if self.inputs and self.outputs:
            report_lines.append(f"\n🔍 LAYER ANALYSIS:")
            
            # Calculate statistics
            input_means = [inp['mean'] for inp in self.inputs]
            output_means = [out['mean'] for out in self.outputs]
            transformations = [abs(out['mean'] - inp['mean']) for inp, out in zip(self.inputs, self.outputs)]
            
            report_lines.append(f"   • Average input activation: {np.mean(input_means):.6f}")
            report_lines.append(f"   • Average output activation: {np.mean(output_means):.6f}")
            report_lines.append(f"   • Average transformation magnitude: {np.mean(transformations):.6f}")
            report_lines.append(f"   • Max transformation: {np.max(transformations):.6f}")
            report_lines.append(f"   • Min transformation: {np.min(transformations):.6f}")
            
            # Layer-by-layer breakdown
            report_lines.append(f"\n📋 LAYER-BY-LAYER BREAKDOWN:")
            for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
                layer_name = inp['layer_name'].split('.')[-1]
                transform_mag = abs(out['mean'] - inp['mean'])
                report_lines.append(f"   Layer {i} ({layer_name}):")
                report_lines.append(f"     - Input:  mean={inp['mean']:.6f}, std={inp['std']:.6f}")
                report_lines.append(f"     - Output: mean={out['mean']:.6f}, std={out['std']:.6f}")
                report_lines.append(f"     - Transform: {transform_mag:.6f}")
        
        # Attention weight analysis
        if self.weights:
            report_lines.append(f"\n🎯 ATTENTION WEIGHT ANALYSIS:")
            report_lines.append(f"   • Number of layers with weights: {len(self.weights)}")
            
            # Analyze weight properties
            all_entropies = []
            all_sparsities = []
            
            for weight_data in self.weights:
                weights = weight_data['weights'].numpy()
                
                # Calculate metrics
                for head in range(weights.shape[1]):
                    head_weights = weights[0, head]
                    
                    # Entropy
                    entropy = -np.sum(head_weights * np.log(head_weights + 1e-8), axis=-1)
                    all_entropies.extend(entropy.mean(axis=0))
                    
                    # Sparsity
                    sparsity = np.mean(head_weights < 0.1)
                    all_sparsities.append(sparsity)
            
            if all_entropies and all_sparsities:
                report_lines.append(f"   • Average attention entropy: {np.mean(all_entropies):.6f}")
                report_lines.append(f"   • Average attention sparsity: {np.mean(all_sparsities):.6f}")
                report_lines.append(f"   • Entropy std: {np.std(all_entropies):.6f}")
                report_lines.append(f"   • Sparsity std: {np.std(all_sparsities):.6f}")
        
        # Recommendations
        report_lines.append(f"\n💡 RECOMMENDATIONS:")
        
        if self.inputs and self.outputs:
            avg_transform = np.mean([abs(out['mean'] - inp['mean']) for inp, out in zip(self.inputs, self.outputs)])
            
            if avg_transform < 0.001:
                report_lines.append(f"   • Low transformation magnitude suggests potential vanishing gradients")
            elif avg_transform > 1.0:
                report_lines.append(f"   • High transformation magnitude suggests potential exploding gradients")
            else:
                report_lines.append(f"   • Transformation magnitudes appear healthy")
        
        if self.weights:
            avg_sparsity = np.mean([np.mean(wd['weights'].numpy() < 0.1) for wd in self.weights])
            
            if avg_sparsity > 0.8:
                report_lines.append(f"   • High attention sparsity may indicate under-utilization")
            elif avg_sparsity < 0.2:
                report_lines.append(f"   • Low attention sparsity suggests good attention diversity")
        
        report_lines.append("="*80)
        
        # Save text report
        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.data_dir, "attention_analysis_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📝 Text report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("📊 ATTENTION ANALYSIS SUMMARY")
        print("="*60)
        for line in report_lines[3:15]:  # Print first few lines
            print(line)
        print("...")
        print(f"📝 Full report available in: {report_file}")


def main():
    """Main function to analyze saved attention data"""
    print("🔍 Attention Data Analyzer")
    print("="*40)
    
    # Initialize analyzer
    analyzer = AttentionDataAnalyzer("attention_captures")
    
    if not any([analyzer.inputs, analyzer.outputs, analyzer.weights]):
        print("❌ No attention data found!")
        print("💡 Run the transformer model first to generate attention data.")
        return
    
    # Create comprehensive analysis
    analyzer.create_comprehensive_report()
    
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()