"""
Main Autoencoder Demo
====================

Comprehensive demo showcasing all autoencoder types
"""

import torch
import torch.nn as nn
import argparse
import sys
import os

# Import all autoencoder implementations
from basic_autoencoders.vanilla_autoencoder import VanillaAutoencoder, AutoencoderTrainer
from basic_autoencoders.denoising_autoencoder import DenoisingAutoencoder, DenoisingTrainer
from basic_autoencoders.sparse_autoencoder import SparseAutoencoder, SparseTrainer
from basic_autoencoders.contractive_autoencoder import ContractiveAutoencoder, ContractiveTrainer
from variational_autoencoders.vanilla_vae import VanillaVAE, VAETrainer
from variational_autoencoders.beta_vae import BetaVAE, BetaVAETrainer
from image_autoencoders.convolutional_ae import ConvolutionalAutoencoder, ConvAutoencoderTrainer
from text_autoencoders.text_autoencoder import TextAutoencoder, TextAutoencoderTrainer, get_text_data_loaders
from utils.data_loaders import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders
from utils.visualization import AutoencoderVisualizer
from utils.latent_space_analyzer import LatentSpaceAnalyzer, compare_latent_spaces


class AutoencoderDemo:
    """Main demo class for all autoencoder types"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Autoencoder Demo initialized on {self.device}")
    
    def run_vanilla_autoencoder_demo(self, epochs=10, batch_size=128):
        """Run Vanilla Autoencoder demo"""
        print("\n" + "="*60)
        print("🔥 VANILLA AUTOENCODER DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000)
        
        # Create model
        model = VanillaAutoencoder(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            latent_dim=64
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = AutoencoderTrainer(model, self.device)
        trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Visualize
        visualizer = AutoencoderVisualizer(model, self.device)
        visualizer.visualize_reconstructions(test_loader)
        visualizer.visualize_latent_space(test_loader)
        
        # Comprehensive latent space analysis
        print("🔬 Performing comprehensive latent space analysis...")
        latent_analyzer = LatentSpaceAnalyzer(model, self.device)
        latent_analyzer.extract_latent_representations(test_loader, max_samples=2000)
        latent_report = latent_analyzer.generate_comprehensive_report()
        
        return model, trainer
    
    def run_denoising_autoencoder_demo(self, epochs=10, batch_size=128):
        """Run Denoising Autoencoder demo"""
        print("\n" + "="*60)
        print("🧹 DENOISING AUTOENCODER DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000)
        
        # Create model
        model = DenoisingAutoencoder(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            latent_dim=64,
            dropout_rate=0.2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = DenoisingTrainer(model, self.device)
        trainer.train(train_loader, test_loader, epochs=epochs, 
                     noise_type='gaussian', noise_factor=0.3)
        
        # Visualize denoising
        from basic_autoencoders.denoising_autoencoder import visualize_denoising
        visualize_denoising(model, test_loader, self.device)
        
        return model, trainer
    
    def run_sparse_autoencoder_demo(self, epochs=10, batch_size=128):
        """Run Sparse Autoencoder demo"""
        print("\n" + "="*60)
        print("✨ SPARSE AUTOENCODER DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000)
        
        # Create model
        model = SparseAutoencoder(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            latent_dim=64,
            sparsity_target=0.05,
            sparsity_weight=3.0
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = SparseTrainer(model, self.device)
        trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Analyze sparsity
        from basic_autoencoders.sparse_autoencoder import analyze_sparsity
        analyze_sparsity(model, test_loader, self.device)
        
        return model, trainer
    
    def run_vae_demo(self, epochs=10, batch_size=128):
        """Run Variational Autoencoder demo"""
        print("\n" + "="*60)
        print("🎯 VARIATIONAL AUTOENCODER DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000, normalize=False)
        
        # Create model
        model = VanillaVAE(
            input_dim=784,
            hidden_dims=[512, 256],
            latent_dim=20
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = VAETrainer(model, self.device, beta=1.0)
        trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Visualize
        from variational_autoencoders.vanilla_vae import visualize_generation, latent_space_interpolation
        visualize_generation(model, self.device)
        latent_space_interpolation(model, test_loader, self.device)
        
        return model, trainer
    
    def run_beta_vae_demo(self, epochs=10, batch_size=128):
        """Run Beta-VAE demo"""
        print("\n" + "="*60)
        print("⚖️ BETA-VAE DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000, normalize=False)
        
        # Create model
        model = BetaVAE(
            input_dim=784,
            hidden_dims=[400, 200],
            latent_dim=10,
            beta=4.0
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = BetaVAETrainer(model, self.device)
        trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Analyze disentanglement
        from variational_autoencoders.beta_vae import analyze_disentanglement, visualize_latent_traversal
        analyze_disentanglement(model, test_loader, self.device, num_samples=500)
        
        # Show latent traversals
        for dim in range(min(3, model.latent_dim)):
            visualize_latent_traversal(model, self.device, latent_dim_idx=dim)
        
        return model, trainer
    
    def run_convolutional_ae_demo(self, epochs=10, batch_size=128):
        """Run Convolutional Autoencoder demo"""
        print("\n" + "="*60)
        print("🖼️ CONVOLUTIONAL AUTOENCODER DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, train_size=5000, test_size=1000, normalize=False)
        
        # Create model
        model = ConvolutionalAutoencoder(
            input_channels=1,
            latent_dim=128,
            image_size=28
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = ConvAutoencoderTrainer(model, self.device)
        trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Visualize
        from image_autoencoders.convolutional_ae import visualize_feature_maps, analyze_latent_space
        visualize_feature_maps(model, test_loader, self.device, layer_idx=0)
        analyze_latent_space(model, test_loader, self.device)
        
        return model, trainer
    
    def run_text_autoencoder_demo(self, epochs=10, batch_size=32):
        """Run Text Autoencoder demo"""
        print("\n" + "="*60)
        print("📝 TEXT AUTOENCODER DEMO")
        print("="*60)
        
        # Generate sample text data
        from text_autoencoders.text_autoencoder import generate_sample_texts
        texts = generate_sample_texts(1000)
        
        # Create data loaders
        train_loader, val_loader, dataset = get_text_data_loaders(
            texts, batch_size=batch_size, max_length=15
        )
        
        print(f"Vocabulary size: {dataset.vocab_size}")
        
        # Create model
        model = TextAutoencoder(
            vocab_size=dataset.vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            latent_dim=32,
            num_layers=2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = TextAutoencoderTrainer(model, self.device)
        trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Visualize
        from text_autoencoders.text_autoencoder import visualize_text_reconstructions, text_interpolation
        visualize_text_reconstructions(model, dataset, val_loader)
        
        # Text interpolation
        text1 = "artificial intelligence is powerful"
        text2 = "machine learning models are useful"
        text_interpolation(model, dataset, text1, text2)
        
        return model, trainer
    
    def run_comparison_demo(self, epochs=5):
        """Run comparison of multiple autoencoder types"""
        print("\n" + "="*60)
        print("📊 AUTOENCODER COMPARISON DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=128, train_size=2000, test_size=500)
        
        models = {}
        trainers = {}
        
        # Train different models
        autoencoder_configs = [
            ('Vanilla AE', VanillaAutoencoder, AutoencoderTrainer, 
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32}),
            ('Sparse AE', SparseAutoencoder, SparseTrainer,
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32, 'sparsity_target': 0.1}),
            ('Denoising AE', DenoisingAutoencoder, DenoisingTrainer,
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32})
        ]
        
        for name, model_class, trainer_class, config in autoencoder_configs:
            print(f"\n🧠 Training {name}...")
            model = model_class(**config)
            trainer = trainer_class(model, self.device)
            trainer.train(train_loader, test_loader, epochs=epochs)
            
            models[name] = model
            trainers[name] = trainer
        
        # Compare results
        from utils.visualization import create_comparison_plot
        create_comparison_plot(models, test_loader, metric='reconstruction_error')
        
        # Comprehensive latent space comparison
        print("🔬 Performing comprehensive latent space comparison...")
        latent_comparison = compare_latent_spaces(models, test_loader, self.device, max_samples=1000)
        
        # Plot all training curves
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for name, trainer in trainers.items():
            plt.plot(trainer.train_losses, label=f'{name} Train', linewidth=2)
            plt.plot(trainer.val_losses, label=f'{name} Val', linestyle='--', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return models, trainers
    
    def run_interactive_demo(self):
        """Interactive demo allowing user to choose autoencoder type"""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE AUTOENCODER DEMO")
        print("="*60)
        
        demos = {
            '1': ('Vanilla Autoencoder', self.run_vanilla_autoencoder_demo),
            '2': ('Denoising Autoencoder', self.run_denoising_autoencoder_demo),
            '3': ('Sparse Autoencoder', self.run_sparse_autoencoder_demo),
            '4': ('Variational Autoencoder (VAE)', self.run_vae_demo),
            '5': ('Beta-VAE', self.run_beta_vae_demo),
            '6': ('Convolutional Autoencoder', self.run_convolutional_ae_demo),
            '7': ('Text Autoencoder', self.run_text_autoencoder_demo),
            '8': ('Comparison Demo', self.run_comparison_demo),
            '9': ('Latent Space Analysis Demo', self.run_latent_analysis_demo),
            '10': ('Run All Demos', self.run_all_demos)
        }
        
        while True:
            print("\nChoose an autoencoder demo:")
            for key, (name, _) in demos.items():
                print(f"{key}. {name}")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-10): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice in demos:
                name, demo_func = demos[choice]
                print(f"\n🚀 Running {name}...")
                try:
                    demo_func()
                    print(f"✅ {name} completed successfully!")
                except Exception as e:
                    print(f"❌ Error in {name}: {e}")
            else:
                print("❌ Invalid choice. Please try again.")
    
    def run_all_demos(self, quick_mode=True):
        """Run all demos sequentially"""
        print("\n" + "="*60)
        print("🏃 RUNNING ALL AUTOENCODER DEMOS")
        print("="*60)
        
        epochs = 3 if quick_mode else 10
        batch_size = 256 if quick_mode else 128
        
        print(f"Quick mode: {quick_mode} (epochs: {epochs}, batch_size: {batch_size})")
        
        results = {}
        
        try:
            print("\n1️⃣ Vanilla Autoencoder...")
            results['vanilla'] = self.run_vanilla_autoencoder_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ Vanilla AE failed: {e}")
        
        try:
            print("\n2️⃣ Denoising Autoencoder...")
            results['denoising'] = self.run_denoising_autoencoder_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ Denoising AE failed: {e}")
        
        try:
            print("\n3️⃣ Sparse Autoencoder...")
            results['sparse'] = self.run_sparse_autoencoder_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ Sparse AE failed: {e}")
        
        try:
            print("\n4️⃣ Variational Autoencoder...")
            results['vae'] = self.run_vae_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ VAE failed: {e}")
        
        try:
            print("\n5️⃣ Beta-VAE...")
            results['beta_vae'] = self.run_beta_vae_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ Beta-VAE failed: {e}")
        
        try:
            print("\n6️⃣ Convolutional Autoencoder...")
            results['conv'] = self.run_convolutional_ae_demo(epochs, batch_size)
        except Exception as e:
            print(f"❌ Conv AE failed: {e}")
        
        try:
            print("\n7️⃣ Text Autoencoder...")
            results['text'] = self.run_text_autoencoder_demo(epochs, 32)
        except Exception as e:
            print(f"❌ Text AE failed: {e}")
        
        print(f"\n✅ Completed {len(results)} out of 7 demos!")
        return results
    
    def run_latent_analysis_demo(self, epochs=10):
        """Run dedicated latent space analysis demo"""
        print("\n" + "="*60)
        print("🔬 LATENT SPACE ANALYSIS DEMO")
        print("="*60)
        
        # Load data
        train_loader, test_loader = get_mnist_loaders(batch_size=128, train_size=3000, test_size=1000)
        
        print("Training multiple models for latent space comparison...")
        
        models = {}
        
        # Train different autoencoder types for comparison
        configs = [
            ('Vanilla AE', VanillaAutoencoder, AutoencoderTrainer, 
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32}),
            ('VAE', VanillaVAE, VAETrainer,
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32}),
            ('β-VAE', BetaVAE, BetaVAETrainer,
             {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32, 'beta': 4.0})
        ]
        
        for name, model_class, trainer_class, config in configs:
            print(f"\n🧠 Training {name}...")
            
            # Adjust data loaders for VAE (no normalization)
            if 'VAE' in name:
                train_loader_vae, test_loader_vae = get_mnist_loaders(
                    batch_size=128, train_size=3000, test_size=1000, normalize=False
                )
                model = model_class(**config)
                trainer = trainer_class(model, self.device)
                trainer.train(train_loader_vae, test_loader_vae, epochs=epochs)
                models[name] = model
            else:
                model = model_class(**config)
                trainer = trainer_class(model, self.device)
                trainer.train(train_loader, test_loader, epochs=epochs)
                models[name] = model
        
        # Comprehensive latent space analysis
        print("\n🔍 Performing comprehensive latent space analysis...")
        
        # Individual analysis for each model
        individual_reports = {}
        for name, model in models.items():
            print(f"\n📊 Analyzing {name} latent space...")
            analyzer = LatentSpaceAnalyzer(model, self.device)
            
            # Use appropriate data loader
            if 'VAE' in name:
                analyzer.extract_latent_representations(test_loader_vae, max_samples=1000)
            else:
                analyzer.extract_latent_representations(test_loader, max_samples=1000)
            
            report = analyzer.generate_comprehensive_report()
            individual_reports[name] = report
        
        # Comparative analysis
        print("\n⚖️ Performing comparative latent space analysis...")
        comparison_results = compare_latent_spaces(models, test_loader, self.device, max_samples=1000)
        
        # Detailed analysis for best model
        best_model_name = max(comparison_results.keys(), 
                             key=lambda x: comparison_results[x]['dim_analysis']['utilization_ratio'])
        
        print(f"\n🏆 Detailed analysis of best model: {best_model_name}")
        best_analyzer = comparison_results[best_model_name]['analyzer']
        
        # Advanced analyses
        print("🎯 Performing advanced analyses...")
        
        # Interpolation analysis
        interpolation_results = best_analyzer.interpolation_analysis(num_pairs=8, num_steps=10)
        print(f"   ✅ Interpolation analysis: {interpolation_results['average_smoothness']:.4f} smoothness")
        
        # Neighborhood analysis  
        neighborhood_results = best_analyzer.neighborhood_analysis(k=10)
        print(f"   ✅ Neighborhood analysis: {neighborhood_results['average_purity']:.3f} purity")
        
        # Manifold analysis
        manifold_results = best_analyzer.manifold_analysis()
        print(f"   ✅ Manifold analysis: {manifold_results['estimated_intrinsic_dimension']:.2f} intrinsic dim")
        
        print("\n✨ Latent space analysis completed!")
        print("💡 Key insights:")
        print(f"   • Best model: {best_model_name}")
        print(f"   • Utilization: {comparison_results[best_model_name]['dim_analysis']['utilization_ratio']:.1%}")
        print(f"   • Independence: {comparison_results[best_model_name]['disentanglement']['independence_score']:.3f}")
        
        return models, individual_reports, comparison_results


def create_project_structure():
    """Create the project directory structure"""
    dirs = [
        'basic_autoencoders',
        'variational_autoencoders', 
        'image_autoencoders',
        'text_autoencoders',
        'audio_autoencoders',
        'multimodal_autoencoders',
        'utils',
        'data',
        'models',
        'results'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        
        # Create __init__.py files
        init_file = os.path.join(dir_name, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""{"="*50}\n{dir_name.replace("_", " ").title()} Module\n{"="*50}\n"""\n')
    
    print("📁 Project structure created!")


def print_repo_summary():
    """Print a summary of the repository"""
    print("\n" + "="*80)
    print("🧠 AUTOENCODER REPOSITORY SUMMARY")
    print("="*80)
    
    summary = """
    📚 This repository contains comprehensive implementations of various autoencoder types:
    
    🔥 BASIC AUTOENCODERS:
    ├── Vanilla Autoencoder - Simple encoder-decoder architecture
    ├── Denoising Autoencoder - Learns to remove noise from corrupted inputs
    ├── Sparse Autoencoder - Enforces sparsity constraints on latent representations
    └── Contractive Autoencoder - Uses Jacobian regularization for robustness
    
    🎯 VARIATIONAL AUTOENCODERS:
    ├── Vanilla VAE - Probabilistic autoencoder with KL divergence regularization
    ├── β-VAE - VAE with controllable disentanglement via beta parameter
    ├── Conditional VAE - VAE with conditional generation capabilities
    └── VQ-VAE - Vector Quantized VAE for discrete latent representations
    
    🖼️ IMAGE AUTOENCODERS:
    ├── Convolutional Autoencoder - Uses conv/deconv layers for images
    ├── U-Net Autoencoder - Skip connections for better reconstruction
    └── Image Colorization AE - Converts grayscale to color images
    
    📝 TEXT AUTOENCODERS:
    ├── Text Autoencoder - RNN/LSTM based for sequential text data
    ├── Sequence-to-Sequence AE - Advanced seq2seq architecture
    └── Transformer Autoencoder - Uses attention mechanisms
    
    🔊 AUDIO AUTOENCODERS:
    ├── Audio Autoencoder - For raw audio waveforms
    └── Spectrogram Autoencoder - For frequency domain representations
    
    🌐 MULTIMODAL AUTOENCODERS:
    ├── Multimodal VAE - Handles multiple data types simultaneously
    └── Cross-Modal Autoencoder - Translates between different modalities
    
    🛠️ UTILITIES:
    ├── Data Loaders - Common datasets and preprocessing utilities
    ├── Visualization - Comprehensive plotting and analysis tools
    └── Training Utils - Common training loops and evaluation metrics
    
    ✨ FEATURES:
    • Complete implementations with detailed documentation
    • Comprehensive visualization and analysis tools
    • Multiple dataset support (MNIST, CIFAR-10, text, audio)
    • GPU acceleration support
    • Modular and extensible design
    • Interactive demos and comparisons
    • Pre-trained model saving/loading
    • Extensive hyperparameter exploration
    """
    
    print(summary)
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Autoencoder Repository Demo')
    parser.add_argument('--demo', type=str, choices=[
        'vanilla', 'denoising', 'sparse', 'contractive', 'vae', 'beta_vae', 
        'conv', 'text', 'comparison', 'latent_analysis', 'interactive', 'all'
    ], default='interactive', help='Type of demo to run')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (fewer epochs)')
    parser.add_argument('--setup', action='store_true', help='Setup project structure')
    parser.add_argument('--summary', action='store_true', help='Show repository summary')
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("🚀 WELCOME TO THE AUTOENCODER REPOSITORY!")
    print("="*80)
    print("A comprehensive collection of autoencoder implementations")
    print("for various modalities and use cases.")
    print("="*80)
    
    # Setup project structure if requested
    if args.setup:
        create_project_structure()
        return
    
    # Show summary if requested
    if args.summary:
        print_repo_summary()
        return
    
    # Initialize demo
    demo = AutoencoderDemo()
    
    # Run specified demo
    if args.demo == 'interactive':
        demo.run_interactive_demo()
    elif args.demo == 'all':
        demo.run_all_demos(quick_mode=args.quick)
    elif args.demo == 'vanilla':
        demo.run_vanilla_autoencoder_demo(args.epochs, args.batch_size)
    elif args.demo == 'denoising':
        demo.run_denoising_autoencoder_demo(args.epochs, args.batch_size)
    elif args.demo == 'sparse':
        demo.run_sparse_autoencoder_demo(args.epochs, args.batch_size)
    elif args.demo == 'vae':
        demo.run_vae_demo(args.epochs, args.batch_size)
    elif args.demo == 'beta_vae':
        demo.run_beta_vae_demo(args.epochs, args.batch_size)
    elif args.demo == 'conv':
        demo.run_convolutional_ae_demo(args.epochs, args.batch_size)
    elif args.demo == 'text':
        demo.run_text_autoencoder_demo(args.epochs, args.batch_size)
    elif args.demo == 'comparison':
        demo.run_comparison_demo(args.epochs)
    elif args.demo == 'latent_analysis':
        demo.run_latent_analysis_demo(args.epochs)
    
    print("\n🎉 Demo completed! Thank you for exploring autoencoders!")
    print("💡 Tip: Try different demos to see various autoencoder capabilities.")


if __name__ == "__main__":
    main()