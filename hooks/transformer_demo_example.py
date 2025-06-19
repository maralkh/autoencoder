"""
Transformer Demo Example
=======================

Complete example demonstrating transformer attention capture and analysis
"""

import torch
import os
import sys
from transformer_autoencoders.transformer_autoencoder import TransformerRunner, AttentionHook
from transformer_autoencoders.attention_data_analyzer import AttentionDataAnalyzer


def run_transformer_demo():
    """Run complete transformer attention analysis demo"""
    print("🚀 Transformer Attention Analysis Demo")
    print("="*60)
    
    # Check if transformers library is available
    try:
        import transformers
        print(f"✅ Transformers library version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers library not found!")
        print("💡 Install with: pip install transformers")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create sample texts for analysis
    sample_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Deep learning requires large datasets and computational resources.",
        "Machine learning algorithms learn patterns from training data.",
        "Artificial intelligence systems can process and understand text.",
        "Neural networks use backpropagation to update their parameters.",
        "Language models generate text by predicting next tokens.",
        "Self-attention computes relationships between all input positions.",
        "Encoder-decoder architectures are used for sequence-to-sequence tasks.",
        "Transformers use positional encodings to understand word order."
    ]
    
    print(f"📝 Prepared {len(sample_texts)} sample texts")
    
    # Initialize transformer runner
    print("\n🤖 Initializing transformer model...")
    runner = TransformerRunner(model_name="gpt2", device=device)
    
    # Prepare input data
    print("\n📊 Preparing input data...")
    input_ids, attention_mask = runner.prepare_data(sample_texts, max_length=32)
    
    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Attention mask shape: {attention_mask.shape}")
    
    # Run model with attention hooks
    print("\n🔗 Running model with attention capture...")
    save_dir = "demo_attention_captures"
    
    try:
        outputs = runner.run_with_hooks(input_ids, attention_mask, save_dir)
        print("✅ Model execution completed successfully!")
    except Exception as e:
        print(f"❌ Error during model execution: {e}")
        return
    
    # Analyze captured attention data
    print("\n📊 Analyzing captured attention patterns...")
    try:
        analysis = runner.analyze_attention_patterns(save_dir)
        print("✅ Attention pattern analysis completed!")
    except Exception as e:
        print(f"❌ Error during attention analysis: {e}")
        return
    
    # Create visualizations
    print("\n🎨 Creating attention visualizations...")
    try:
        runner.visualize_attention_flow(save_dir)
        print("✅ Attention flow visualization created!")
    except Exception as e:
        print(f"❌ Error creating flow visualization: {e}")
    
    # Visualize attention weights for first few layers/heads
    print("\n🎯 Creating attention weight visualizations...")
    try:
        max_layers = min(2, len(runner.attention_hook.attention_weights))
        max_heads = 2
        
        for layer_idx in range(max_layers):
            num_heads = runner.attention_hook.attention_weights[layer_idx]['weights'].shape[1]
            for head_idx in range(min(max_heads, num_heads)):
                runner.visualize_attention_weights(save_dir, layer_idx, head_idx)
        
        print(f"✅ Created visualizations for {max_layers} layers x {max_heads} heads")
    except Exception as e:
        print(f"❌ Error creating weight visualizations: {e}")
    
    # Load and analyze saved data
    print("\n🔍 Loading and analyzing saved data...")
    try:
        analyzer = AttentionDataAnalyzer(save_dir)
        print("✅ Data loaded successfully!")
        
        # Create comprehensive report
        print("\n📋 Creating comprehensive analysis report...")
        analyzer.create_comprehensive_report()
        print("✅ Comprehensive report created!")
        
    except Exception as e:
        print(f"❌ Error during comprehensive analysis: {e}")
    
    # Print summary of captured data
    print(f"\n📊 CAPTURE SUMMARY:")
    print(f"   • Save directory: {save_dir}")
    
    if os.path.exists(os.path.join(save_dir, "attention_data_summary.json")):
        import json
        with open(os.path.join(save_dir, "attention_data_summary.json"), 'r') as f:
            summary = json.load(f)
        
        print(f"   • Layers captured: {summary.get('num_layers', 'N/A')}")
        print(f"   • Total captures: {summary.get('total_captures', 'N/A')}")
        print(f"   • Layer names: {summary.get('unique_layers', [])[:3]}...")
    
    # List generated files
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"\n📁 Generated files ({len(files)} total):")
        for file in sorted(files)[:10]:  # Show first 10 files
            print(f"   • {file}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🔍 Check the '{save_dir}' directory for all generated files and visualizations.")


def demo_custom_text_analysis():
    """Demo with custom text analysis"""
    print("\n" + "="*60)
    print("📝 CUSTOM TEXT ANALYSIS DEMO")
    print("="*60)
    
    # Custom texts focusing on different topics
    custom_texts = {
        "Technology": [
            "Artificial intelligence is transforming industries worldwide.",
            "Machine learning algorithms require massive computational power.",
            "Deep neural networks can recognize complex patterns in data."
        ],
        "Science": [
            "The scientific method involves hypothesis testing and experimentation.",
            "Quantum mechanics describes behavior at the atomic scale.",
            "Climate change affects global weather patterns significantly."
        ],
        "Literature": [
            "Shakespeare's sonnets explore themes of love and mortality.",
            "Modern poetry often experiments with form and structure.",
            "Narrative techniques vary across different literary genres."
        ]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for topic, texts in custom_texts.items():
        print(f"\n🔍 Analyzing {topic} texts...")
        
        try:
            # Initialize runner
            runner = TransformerRunner(model_name="gpt2", device=device)
            
            # Prepare data
            input_ids, attention_mask = runner.prepare_data(texts, max_length=24)
            
            # Run with hooks
            save_dir = f"attention_analysis_{topic.lower()}"
            outputs = runner.run_with_hooks(input_ids, attention_mask, save_dir)
            
            # Quick analysis
            runner.analyze_attention_patterns(save_dir)
            
            print(f"✅ {topic} analysis completed!")
            print(f"📁 Results saved to: {save_dir}")
            
        except Exception as e:
            print(f"❌ Error analyzing {topic}: {e}")


def demo_attention_comparison():
    """Demo comparing attention patterns across different text types"""
    print("\n" + "="*60)
    print("🔀 ATTENTION PATTERN COMPARISON DEMO")
    print("="*60)
    
    # Different text structures
    text_types = {
        "Short_Sentences": [
            "The cat sat on the mat.",
            "Birds fly in the sky.",
            "Water flows down the river."
        ],
        "Long_Sentences": [
            "The magnificent cat with beautiful stripes sat comfortably on the soft, warm mat near the fireplace.",
            "Colorful birds with vibrant feathers fly gracefully through the clear blue sky above the mountains.",
            "Crystal clear water flows gently down the winding river through the peaceful valley."
        ],
        "Questions": [
            "What is the meaning of life?",
            "How do birds learn to fly?",
            "Why does water flow downhill?"
        ]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comparison_results = {}
    
    for text_type, texts in text_types.items():
        print(f"\n📊 Processing {text_type}...")
        
        try:
            runner = TransformerRunner(model_name="gpt2", device=device)
            input_ids, attention_mask = runner.prepare_data(texts, max_length=32)
            
            save_dir = f"comparison_{text_type.lower()}"
            outputs = runner.run_with_hooks(input_ids, attention_mask, save_dir)
            
            # Store results for comparison
            comparison_results[text_type] = {
                'save_dir': save_dir,
                'num_inputs': len(runner.attention_hook.attention_inputs),
                'num_outputs': len(runner.attention_hook.attention_outputs),
                'num_weights': len(runner.attention_hook.attention_weights)
            }
            
            print(f"✅ {text_type} processed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing {text_type}: {e}")
    
    # Print comparison summary
    if comparison_results:
        print(f"\n📋 COMPARISON SUMMARY:")
        print("-" * 50)
        for text_type, results in comparison_results.items():
            print(f"{text_type}:")
            print(f"  • Attention inputs captured: {results['num_inputs']}")
            print(f"  • Attention outputs captured: {results['num_outputs']}")
            print(f"  • Attention weights captured: {results['num_weights']}")
            print(f"  • Save directory: {results['save_dir']}")
        print("-" * 50)


def main():
    """Main demo function"""
    print("🎯 TRANSFORMER ATTENTION ANALYSIS - COMPLETE DEMO")
    print("="*70)
    
    # Check dependencies
    try:
        import transformers
        import torch
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install required packages:")
        print("   pip install transformers torch matplotlib seaborn")
        return
    
    # Run main demo
    try:
        run_transformer_demo()
    except Exception as e:
        print(f"❌ Error in main demo: {e}")
        return
    
    # Ask user for additional demos
    while True:
        print("\n" + "="*50)
        print("🎮 ADDITIONAL DEMO OPTIONS:")
        print("1. Custom Text Analysis Demo")
        print("2. Attention Pattern Comparison Demo")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '1':
            try:
                demo_custom_text_analysis()
            except Exception as e:
                print(f"❌ Error in custom text demo: {e}")
        
        elif choice == '2':
            try:
                demo_attention_comparison()
            except Exception as e:
                print(f"❌ Error in comparison demo: {e}")
        
        elif choice == '3':
            print("👋 Demo completed!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()