"""
Text Autoencoder Implementation
==============================

Autoencoder for text data using RNN/LSTM architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import string
import random
from collections import Counter
import pickle


class TextDataset(Dataset):
    """Dataset class for text data"""
    
    def __init__(self, texts, vocab, max_length=50):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        # Add special tokens to vocabulary if not present
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            if token not in self.word_to_idx:
                self.word_to_idx[token] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = token
        
        self.vocab_size = len(self.word_to_idx)
    
    def text_to_indices(self, text):
        """Convert text to indices"""
        words = text.lower().split()
        indices = [self.word_to_idx.get(word, self.word_to_idx[self.UNK_TOKEN]) for word in words]
        
        # Add SOS and EOS tokens
        indices = [self.word_to_idx[self.SOS_TOKEN]] + indices + [self.word_to_idx[self.EOS_TOKEN]]
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.word_to_idx[self.PAD_TOKEN]] * (self.max_length - len(indices)))
        
        return torch.LongTensor(indices)
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            word = self.idx_to_word[idx.item() if isinstance(idx, torch.Tensor) else idx]
            if word == self.EOS_TOKEN:
                break
            if word not in [self.PAD_TOKEN, self.SOS_TOKEN]:
                words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = self.text_to_indices(text)
        return indices, indices  # Input and target are the same for autoencoder


class TextAutoencoder(nn.Module):
    """
    Text Autoencoder using LSTM
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): Hidden dimension of LSTM
        latent_dim (int): Latent space dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, latent_dim=64, 
                 num_layers=2, dropout=0.3):
        super(TextAutoencoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Latent space projection
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)  # *2 for bidirectional
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, x, lengths=None):
        """
        Encode text sequence to latent representation
        
        Args:
            x: Input text indices [batch_size, seq_len]
            lengths: Actual lengths of sequences (for packing)
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # LSTM encoding
        if lengths is not None:
            # Pack sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.encoder_lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.encoder_lstm(embedded)
        
        # Use last hidden state from both directions
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to latent space
        latent = self.encoder_fc(combined_hidden)
        
        return latent
    
    def decode(self, latent, max_length, target_sequence=None):
        """
        Decode latent representation to text sequence
        
        Args:
            latent: Latent representation [batch_size, latent_dim]
            max_length: Maximum sequence length to generate
            target_sequence: Target sequence for teacher forcing during training
        """
        batch_size = latent.size(0)
        device = latent.device
        
        # Initialize decoder hidden state
        decoder_hidden = self.decoder_fc(latent)  # [batch_size, hidden_dim]
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        # Initialize with SOS token
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # Assuming SOS token is 1
        
        outputs = []
        
        for t in range(max_length):
            # Embedding
            embedded = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim]
            
            # LSTM step
            lstm_out, (decoder_hidden, decoder_cell) = self.decoder_lstm(embedded, (decoder_hidden, decoder_cell))
            
            # Output projection
            output = self.output_projection(lstm_out)  # [batch_size, 1, vocab_size]
            outputs.append(output)
            
            # Teacher forcing during training
            if target_sequence is not None and t < target_sequence.size(1) - 1:
                decoder_input = target_sequence[:, t+1:t+2]
            else:
                # Use predicted token
                decoder_input = output.argmax(dim=-1)
        
        return torch.cat(outputs, dim=1)  # [batch_size, max_length, vocab_size]
    
    def forward(self, x, lengths=None):
        """Complete forward pass"""
        # Encode
        latent = self.encode(x, lengths)
        
        # Decode with teacher forcing
        max_length = x.size(1)
        reconstructed = self.decode(latent, max_length, x)
        
        return reconstructed, latent
    
    def generate(self, latent, max_length=50):
        """Generate text from latent representation"""
        self.eval()
        with torch.no_grad():
            output = self.decode(latent, max_length)
            return output.argmax(dim=-1)


class TextAutoencoderTrainer:
    """Training class for Text Autoencoder"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Calculate actual lengths (for packing)
            lengths = (data != 0).sum(dim=1).cpu()
            
            # Forward pass
            reconstructed, latent = self.model(data, lengths)
            
            # Calculate loss
            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            reconstructed = reconstructed.view(-1, reconstructed.size(-1))
            target = target.view(-1)
            
            loss = self.criterion(reconstructed, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
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
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                lengths = (data != 0).sum(dim=1).cpu()
                reconstructed, latent = self.model(data, lengths)
                
                reconstructed = reconstructed.view(-1, reconstructed.size(-1))
                target = target.view(-1)
                
                loss = self.criterion(reconstructed, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training loop"""
        print(f"Training Text Autoencoder on {self.device}")
        
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
        plt.title('Text Autoencoder - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


def create_vocabulary(texts, min_freq=2, max_vocab_size=10000):
    """
    Create vocabulary from text corpus
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for word inclusion
        max_vocab_size: Maximum vocabulary size
    """
    # Tokenize and count words
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Filter by frequency and limit size
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    filtered_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)
    
    if len(filtered_words) > max_vocab_size - 4:  # Reserve space for special tokens
        filtered_words = filtered_words[:max_vocab_size - 4]
    
    # Add special tokens
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + filtered_words
    
    return vocab


def generate_sample_texts(num_samples=1000):
    """Generate sample texts for demonstration"""
    templates = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming the world",
        "machine learning models learn from data",
        "deep neural networks are powerful tools",
        "natural language processing enables text understanding",
        "computer vision helps machines see",
        "reinforcement learning trains agents through interaction",
        "data science combines statistics and programming",
        "python is a popular programming language",
        "open source software powers innovation"
    ]
    
    # Generate variations
    texts = []
    for _ in range(num_samples):
        template = random.choice(templates)
        words = template.split()
        
        # Random modifications
        if random.random() < 0.3:
            # Remove random word
            if len(words) > 3:
                words.pop(random.randint(0, len(words)-1))
        
        if random.random() < 0.3:
            # Add random word
            extra_words = ['very', 'quite', 'really', 'extremely', 'highly']
            words.insert(random.randint(0, len(words)), random.choice(extra_words))
        
        texts.append(' '.join(words))
    
    return texts


def visualize_text_reconstructions(model, dataset, data_loader, num_samples=5):
    """Visualize text reconstruction results"""
    model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data[:num_samples].to(next(model.parameters()).device)
        
        # Get reconstructions
        reconstructed, latent = model(data)
        reconstructed_indices = reconstructed.argmax(dim=-1)
        
        print("Text Reconstruction Results:")
        print("=" * 80)
        
        for i in range(num_samples):
            original_text = dataset.indices_to_text(data[i])
            reconstructed_text = dataset.indices_to_text(reconstructed_indices[i])
            
            print(f"Sample {i+1}:")
            print(f"Original:      {original_text}")
            print(f"Reconstructed: {reconstructed_text}")
            print("-" * 40)


def analyze_latent_space_text(model, data_loader, dataset, device, num_samples=500):
    """Analyze latent space of text autoencoder"""
    model.eval()
    
    latent_vectors = []
    original_texts = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i * data.size(0) >= num_samples:
                break
            
            data = data.to(device)
            lengths = (data != 0).sum(dim=1).cpu()
            latent = model.encode(data, lengths)
            
            latent_vectors.append(latent.cpu().numpy())
            
            # Store original texts
            for j in range(data.size(0)):
                text = dataset.indices_to_text(data[j])
                original_texts.append(text)
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    original_texts = original_texts[:num_samples]
    
    # Visualize latent space
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    axes[0, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0, 0].set_title('PCA of Text Latent Space')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Latent dimension statistics
    latent_means = np.mean(latent_vectors, axis=0)
    latent_stds = np.std(latent_vectors, axis=0)
    
    axes[0, 1].plot(latent_means, 'o-', label='Mean', alpha=0.7)
    axes[0, 1].plot(latent_stds, 'o-', label='Std', alpha=0.7)
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Latent Dimension Statistics')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of latent activations
    axes[1, 0].hist(latent_vectors.flatten(), bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Latent Activations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Text length vs latent norm
    text_lengths = [len(text.split()) for text in original_texts]
    latent_norms = np.linalg.norm(latent_vectors, axis=1)
    
    axes[1, 1].scatter(text_lengths, latent_norms, alpha=0.6)
    axes[1, 1].set_xlabel('Text Length (words)')
    axes[1, 1].set_ylabel('Latent Vector Norm')
    axes[1, 1].set_title('Text Length vs Latent Representation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return latent_vectors, original_texts


def text_interpolation(model, dataset, text1, text2, num_steps=5):
    """Interpolate between two texts in latent space"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert texts to indices
    indices1 = dataset.text_to_indices(text1).unsqueeze(0).to(device)
    indices2 = dataset.text_to_indices(text2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode texts
        latent1 = model.encode(indices1)
        latent2 = model.encode(indices2)
        
        # Create interpolations
        alphas = np.linspace(0, 1, num_steps)
        interpolated_texts = []
        
        for alpha in alphas:
            # Linear interpolation in latent space
            latent_interp = (1 - alpha) * latent1 + alpha * latent2
            
            # Generate text from interpolated latent
            generated_indices = model.generate(latent_interp, max_length=dataset.max_length)
            generated_text = dataset.indices_to_text(generated_indices[0])
            
            interpolated_texts.append(generated_text)
    
    # Display results
    print("Text Interpolation Results:")
    print("=" * 60)
    print(f"Text 1: {text1}")
    for i, (alpha, text) in enumerate(zip(alphas, interpolated_texts)):
        print(f"α={alpha:.2f}: {text}")
    print(f"Text 2: {text2}")
    
    return interpolated_texts


def semantic_similarity_analysis(model, dataset, data_loader, device):
    """Analyze semantic similarity in latent space"""
    model.eval()
    
    # Get sample texts and their latent representations
    texts = []
    latent_vectors = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            lengths = (data != 0).sum(dim=1).cpu()
            latent = model.encode(data, lengths)
            
            for i in range(data.size(0)):
                text = dataset.indices_to_text(data[i])
                texts.append(text)
                latent_vectors.append(latent[i].cpu().numpy())
                
                if len(texts) >= 100:  # Limit for analysis
                    break
            
            if len(texts) >= 100:
                break
    
    latent_vectors = np.array(latent_vectors)
    
    # Calculate pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(latent_vectors)
    
    # Find most similar pairs
    n = len(texts)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((similarity_matrix[i, j], texts[i], texts[j]))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    print("Most Similar Text Pairs in Latent Space:")
    print("=" * 80)
    for sim, text1, text2 in similarities[:10]:
        print(f"Similarity: {sim:.3f}")
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print("-" * 40)
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Pairwise Text Similarities in Latent Space')
    plt.xlabel('Text Index')
    plt.ylabel('Text Index')
    plt.show()


def get_text_data_loaders(texts=None, batch_size=32, max_length=20, train_ratio=0.8):
    """Create data loaders for text data"""
    
    if texts is None:
        texts = generate_sample_texts(1000)
    
    # Create vocabulary
    vocab = create_vocabulary(texts, min_freq=1, max_vocab_size=5000)
    
    # Create dataset
    dataset = TextDataset(texts, vocab, max_length)
    
    # Train/validation split
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset


def main():
    """Main execution example"""
    print("📝 Starting Text Autoencoder Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample data
    print("📚 Generating sample text data...")
    texts = generate_sample_texts(2000)
    
    # Create data loaders
    train_loader, val_loader, dataset = get_text_data_loaders(
        texts, batch_size=64, max_length=15
    )
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("🧠 Creating Text Autoencoder...")
    model = TextAutoencoder(
        vocab_size=dataset.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        latent_dim=32,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TextAutoencoderTrainer(model, device, lr=1e-3)
    
    # Train
    print("🏋️ Starting training...")
    trainer.train(train_loader, val_loader, epochs=20)
    
    # Plot losses
    trainer.plot_losses()
    
    # Visualize reconstructions
    print("🎨 Visualizing text reconstructions...")
    visualize_text_reconstructions(model, dataset, val_loader)
    
    # Analyze latent space
    print("🔍 Analyzing latent space...")
    latent_vectors, original_texts = analyze_latent_space_text(
        model, val_loader, dataset, device
    )
    
    # Text interpolation
    print("🌈 Text interpolation...")
    text1 = "artificial intelligence is powerful"
    text2 = "machine learning models are useful"
    text_interpolation(model, dataset, text1, text2)
    
    # Semantic similarity analysis
    print("📊 Semantic similarity analysis...")
    semantic_similarity_analysis(model, dataset, val_loader, device)
    
    # Save model and dataset
    torch.save(model.state_dict(), 'text_autoencoder.pth')
    with open('text_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    print("💾 Model and dataset saved!")
    
    print("\n📋 Summary:")
    print("Text Autoencoder learns to encode text sequences into a")
    print("continuous latent space and decode them back to text.")
    print("This enables text interpolation, similarity analysis,")
    print("and semantic representation learning.")


if __name__ == "__main__":
    main()