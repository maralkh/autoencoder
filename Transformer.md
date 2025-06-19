# 🤖 Transformer Attention Analysis

Advanced tools for capturing and analyzing attention patterns in transformer models.

## 📋 Overview

This module provides comprehensive tools to:
- **Capture attention data** during transformer forward passes using hook functions
- **Analyze attention patterns** across layers and heads
- **Visualize attention flow** through the model
- **Generate detailed reports** with insights and recommendations

## 🗂️ Files

- `transformer_autoencoder.py` - Main transformer runner with attention hooks
- `attention_data_analyzer.py` - Analysis tools for captured attention data
- `transformer_demo_example.py` - Complete demo example
- `README.md` - This documentation

## 🚀 Quick Start

### Basic Usage

```python
from transformer_autoencoder import TransformerRunner

# Initialize transformer
runner = TransformerRunner(model_name="gpt2", device='cuda')

# Prepare your texts
texts = ["Your sample texts here..."]
input_ids, attention_mask = runner.prepare_data(texts, max_length=128)

# Run with attention capture
outputs = runner.run_with_hooks(input_ids, attention_mask, save_dir="attention_data")

# Analyze patterns
analysis = runner.analyze_attention_patterns("attention_data")

# Create visualizations
runner.visualize_attention_flow("attention_data")
runner.visualize_attention_weights("attention_data", layer_idx=0, head_idx=0)
```

### Advanced Analysis

```python
from attention_data_analyzer import AttentionDataAnalyzer

# Load saved attention data
analyzer = AttentionDataAnalyzer("attention_data")

# Generate comprehensive report
analyzer.create_comprehensive_report()
```

## 🔧 Installation

```bash
# Required packages
pip install transformers torch matplotlib seaborn scikit-learn

# Optional for better performance
pip install accelerate
```

## 📊 What Gets Captured

### Attention Inputs
- Hidden states entering attention layers
- Shape information and statistics
- Layer-wise metadata

### Attention Outputs  
- Processed hidden states from attention
- Transformation statistics
- Output shape information

### Attention Weights
- Multi-head attention weight matrices
- Head-specific attention patterns
- Token-to-token attention scores

## 🎨 Visualizations Generated

### 1. Attention Flow Analysis
- Mean activation flow through layers
- Standard deviation evolution
- Transformation magnitude tracking
- Input vs output correlations

### 2. Attention Weight Heatmaps
- Layer-specific attention patterns
- Head-specific visualizations
- Token-to-token attention maps

### 3. Comprehensive Dashboard
- Layer-wise statistics
- Activation distributions
- Dimensionality analysis
- Pattern evolution tracking

## 📋 Analysis Reports

### Automated Analysis Includes:
- **Layer Statistics**: Mean, std, transformation magnitude per layer
- **Attention Properties**: Entropy, sparsity, diversity across heads
- **Pattern Evolution**: How attention changes through depth
- **Sequence Analysis**: Position-dependent attention patterns
- **Head Comparison**: Diversity and specialization analysis
- **Recommendations**: Insights for model improvement

### Report Files Generated:
- `attention_analysis_report.txt` - Detailed text report
- `attention_data_summary.json` - Structured summary data
- `comprehensive_attention_analysis.png` - Visual dashboard
- `attention_flow_visualization.png` - Flow analysis
- `attention_weights_*.png` - Weight visualizations

## 🎯 Use Cases

### Research Applications
- **Model Interpretability**: Understand what the model focuses on
- **Architecture Analysis**: Compare different transformer variants
- **Training Dynamics**: Track attention evolution during training
- **Attention Patterns**: Study linguistic patterns in attention

### Debugging & Optimization
- **Attention Collapse**: Detect uniform attention patterns
- **Head Redundancy**: Identify similar attention heads
- **Layer Analysis**: Find optimal model depth
- **Pattern Anomalies**: Spot unusual attention behaviors

### Educational Purposes
- **Attention Visualization**: Teach transformer mechanics
- **Pattern Analysis**: Demonstrate attention concepts
- **Model Exploration**: Interactive attention analysis

## 📈 Example Analysis Workflow

```python
# 1. Initialize and run analysis
runner = TransformerRunner("gpt2")
texts = ["Example text for analysis"]
input_ids, mask = runner.prepare_data(texts)
outputs = runner.run_with_hooks(input_ids, mask, "results")

# 2. Generate visualizations
runner.visualize_attention_flow("results")
for layer in range(3):  # First 3 layers
    for head in range(2):  # First 2 heads
        runner.visualize_attention_weights("results", layer, head)

# 3. Comprehensive analysis
analyzer = AttentionDataAnalyzer("results")
analyzer.create_comprehensive_report()

# 4. Custom analysis
# Load saved data for further processing
with open("results/attention_data_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    
# Your custom analysis here...
```

## ⚙️ Configuration Options

### Model Selection
```python
# Different model sizes
runner = TransformerRunner("gpt2")           # 117M parameters
runner = TransformerRunner("gpt2-medium")    # 345M parameters  
runner = TransformerRunner("gpt2-large")     # 762M parameters

# Other transformer models
runner = TransformerRunner("distilgpt2")     # Smaller, faster
runner = TransformerRunner("gpt2-xl")        # 1.5B parameters
```

### Analysis Parameters
```python
# Adjust sequence length
input_ids, mask = runner.prepare_data(texts, max_length=512)

# Select specific layers for analysis
runner.visualize_attention_weights("results", layer_idx=5, head_idx=3)

# Customize analysis depth
analyzer.create_comprehensive_report()  # Full analysis
# Or load specific components only
```

## 🚨 Memory Considerations

### Large Models
- **GPU Memory**: Ensure sufficient VRAM for model + attention storage
- **Batch Size**: Reduce batch size if memory issues occur
- **Sequence Length**: Shorter sequences require less memory
- **Layer Selection**: Analyze subset of layers for large models

### Memory-Efficient Options
```python
# Use smaller model
runner = TransformerRunner("distilgpt2", device='cuda')

# Shorter sequences
input_ids, mask = runner.prepare_data(texts, max_length=64)

# Process fewer samples
texts = texts[:5]  # Limit number of texts
```

## 🔍 Troubleshooting

### Common Issues

**ImportError: No module named 'transformers'**
```bash
pip install transformers
```

**CUDA out of memory**
```python
# Use CPU instead
runner = TransformerRunner("gpt2", device='cpu')

# Or reduce sequence length
input_ids, mask = runner.prepare_data(texts, max_length=32)
```

**No attention weights captured**
- Ensure model has `output_attentions=True`
- Check that hooks are registered correctly
- Verify model architecture supports attention extraction

**Empty visualization**
- Check that forward pass completed successfully
- Ensure attention data was saved properly
- Verify file paths and permissions

## 📚 Additional Resources

### Papers & References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418)

### Related Tools
- [BertViz](https://github.com/jessevig/bertviz) - BERT attention visualization
- [Transformer Lens](https://github.com/neelnanda-io/TransformerLens) - Mechanistic interpretability

### Hugging Face Documentation
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Model Hub](https://huggingface.co/models)

## 🤝 Contributing

To extend this module:

1. **Add new models**: Extend `TransformerRunner` for different architectures
2. **New visualizations**: Add methods to `AttentionDataAnalyzer`
3. **Analysis metrics**: Implement new attention analysis techniques
4. **Export formats**: Add support for different output formats

## 📄 License

This module is part of the Autoencoder Repository and follows the same MIT license.

---

🎯 **Happy Analyzing!** Use these tools to gain deep insights into transformer attention patterns.