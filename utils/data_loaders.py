"""
Data Loaders and Utilities
==========================

Common utilities for loading datasets and preprocessing data
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
import os
import librosa
from PIL import Image


class CustomDataset(Dataset):
    """Custom dataset class for arbitrary data"""
    
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        else:
            return sample, 0  # Dummy label for unsupervised learning


def get_mnist_loaders(batch_size=128, train_size=None, test_size=None, normalize=True):
    """
    Load MNIST dataset
    
    Args:
        batch_size: Batch size for data loaders
        train_size: Limit training set size (for quick testing)
        test_size: Limit test set size (for quick testing)
        normalize: Whether to normalize pixel values
    """
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Limit dataset size if specified
    if train_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    if test_size:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_fashion_mnist_loaders(batch_size=128, train_size=None, test_size=None):
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    if train_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    if test_size:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, train_size=None, test_size=None):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    if train_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    if test_size:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_celeba_loaders(batch_size=128, image_size=64, data_path='./data/celeba'):
    """
    Load CelebA dataset (requires manual download)
    
    Args:
        batch_size: Batch size
        image_size: Size to resize images to
        data_path: Path to CelebA data directory
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        dataset = torchvision.datasets.CelebA(
            root='./data', split='train', download=False, transform=transform
        )
        
        # Split into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    except:
        print("CelebA dataset not found. Please download manually.")
        return None, None


def get_synthetic_2d_data(dataset_type='blobs', n_samples=1000, batch_size=128):
    """
    Generate synthetic 2D datasets for testing
    
    Args:
        dataset_type: Type of synthetic data ('blobs', 'circles', 'moons')
        n_samples: Number of samples
        batch_size: Batch size
    """
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, 
                         random_state=42, cluster_std=1.5)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.6, random_state=42)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Normalize data
    X = (X - X.mean(0)) / X.std(0)
    
    # Convert to tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Create dataset and loader
    dataset = CustomDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def get_text_data_loader(text_file_path, sequence_length=50, batch_size=32):
    """
    Load text data for sequence autoencoders
    
    Args:
        text_file_path: Path to text file
        sequence_length: Length of text sequences
        batch_size: Batch size
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character mappings
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Create sequences
        sequences = []
        for i in range(0, len(text) - sequence_length, sequence_length):
            seq = text[i:i + sequence_length]
            sequences.append([char_to_idx[ch] for ch in seq])
        
        # Convert to tensors
        sequences = torch.LongTensor(sequences)
        
        # Create dataset and loader
        dataset = CustomDataset(sequences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader, char_to_idx, idx_to_char, len(chars)
    
    except FileNotFoundError:
        print(f"Text file not found: {text_file_path}")
        return None, None, None, None


def get_audio_data_loader(audio_dir, sample_rate=22050, duration=3.0, batch_size=32):
    """
    Load audio data for audio autoencoders
    
    Args:
        audio_dir: Directory containing audio files
        sample_rate: Audio sample rate
        duration: Duration of audio clips in seconds
        batch_size: Batch size
    """
    try:
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        audio_data = []
        
        for file in audio_files[:100]:  # Limit for demo
            file_path = os.path.join(audio_dir, file)
            audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            # Pad or trim to fixed length
            target_length = int(sample_rate * duration)
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            audio_data.append(audio)
        
        # Convert to tensor
        audio_data = torch.FloatTensor(np.array(audio_data))
        
        # Create dataset and loader
        dataset = CustomDataset(audio_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
    
    except Exception as e:
        print(f"Error loading audio data: {e}")
        return None


def create_noisy_dataset(clean_loader, noise_type='gaussian', noise_factor=0.3):
    """
    Create a noisy version of a dataset for denoising tasks
    
    Args:
        clean_loader: DataLoader with clean data
        noise_type: Type of noise to add
        noise_factor: Noise intensity
    """
    noisy_data = []
    clean_data = []
    labels = []
    
    for data, label in clean_loader:
        # Add noise
        if noise_type == 'gaussian':
            noise = torch.randn_like(data) * noise_factor
            noisy = torch.clamp(data + noise, 0, 1)
        elif noise_type == 'salt_pepper':
            noise_mask = torch.rand_like(data)
            noisy = data.clone()
            noisy[noise_mask < noise_factor/2] = 0
            noisy[noise_mask > 1 - noise_factor/2] = 1
        elif noise_type == 'masking':
            mask = torch.rand_like(data) > noise_factor
            noisy = data * mask.float()
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noisy_data.append(noisy)
        clean_data.append(data)
        labels.append(label)
    
    # Concatenate all data
    noisy_data = torch.cat(noisy_data, dim=0)
    clean_data = torch.cat(clean_data, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Create paired dataset (noisy input, clean target)
    class NoisyDataset(Dataset):
        def __init__(self, noisy, clean, labels):
            self.noisy = noisy
            self.clean = clean
            self.labels = labels
        
        def __len__(self):
            return len(self.noisy)
        
        def __getitem__(self, idx):
            return self.noisy[idx], self.clean[idx], self.labels[idx]
    
    dataset = NoisyDataset(noisy_data, clean_data, labels)
    loader = DataLoader(dataset, batch_size=clean_loader.batch_size, shuffle=True)
    
    return loader


def create_multimodal_dataset(image_loader, text_data=None, audio_data=None):
    """
    Create a multimodal dataset combining different modalities
    
    Args:
        image_loader: DataLoader for images
        text_data: Optional text data
        audio_data: Optional audio data
    """
    multimodal_data = []
    
    for i, (images, labels) in enumerate(image_loader):
        batch_size = images.size(0)
        
        # Create multimodal samples
        for j in range(batch_size):
            sample = {'image': images[j], 'label': labels[j]}
            
            # Add text if available
            if text_data is not None and i * batch_size + j < len(text_data):
                sample['text'] = text_data[i * batch_size + j]
            
            # Add audio if available
            if audio_data is not None and i * batch_size + j < len(audio_data):
                sample['audio'] = audio_data[i * batch_size + j]
            
            multimodal_data.append(sample)
        
        if i >= 100:  # Limit for demo
            break
    
    class MultimodalDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = MultimodalDataset(multimodal_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return loader


def visualize_dataset_samples(loader, dataset_name="Dataset", num_samples=8, figsize=(12, 6)):
    """
    Visualize samples from a dataset
    
    Args:
        loader: DataLoader to visualize
        dataset_name: Name of the dataset for title
        num_samples: Number of samples to show
        figsize: Figure size
    """
    # Get a batch of data
    data_iter = iter(loader)
    
    try:
        if hasattr(loader.dataset, 'data') and len(loader.dataset.data.shape) == 2:
            # 2D data (like synthetic datasets)
            data, labels = next(data_iter)
            
            plt.figure(figsize=figsize)
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.title(f'{dataset_name} - 2D Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.colorbar()
            plt.show()
        
        else:
            # Image data
            data, labels = next(data_iter)
            
            if len(data.shape) == 4:  # Batch of images
                num_samples = min(num_samples, data.size(0))
                
                # Determine grid size
                cols = min(8, num_samples)
                rows = (num_samples + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=figsize)
                if rows == 1:
                    axes = [axes] if cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i in range(num_samples):
                    img = data[i]
                    
                    # Handle different image formats
                    if img.shape[0] == 1:  # Grayscale
                        img = img.squeeze(0)
                        cmap = 'gray'
                    elif img.shape[0] == 3:  # RGB
                        img = img.permute(1, 2, 0)
                        cmap = None
                    else:
                        img = img.squeeze()
                        cmap = 'gray'
                    
                    # Denormalize if needed
                    if img.min() < 0:
                        img = (img + 1) / 2  # Assuming [-1, 1] normalization
                    
                    axes[i].imshow(img.cpu().numpy(), cmap=cmap)
                    axes[i].set_title(f'Label: {labels[i].item()}')
                    axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(num_samples, len(axes)):
                    axes[i].axis('off')
                
                plt.suptitle(f'{dataset_name} - Sample Images')
                plt.tight_layout()
                plt.show()
    
    except Exception as e:
        print(f"Error visualizing dataset: {e}")


def get_dataset_stats(loader, dataset_name="Dataset"):
    """
    Get basic statistics about a dataset
    
    Args:
        loader: DataLoader to analyze
        dataset_name: Name of the dataset
    """
    print(f"\n📊 {dataset_name} Statistics:")
    print("-" * 50)
    
    # Basic info
    total_samples = len(loader.dataset)
    batch_size = loader.batch_size
    num_batches = len(loader)
    
    print(f"Total samples: {total_samples:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    
    # Get a sample to analyze data properties
    try:
        data_iter = iter(loader)
        sample_data, sample_labels = next(data_iter)
        
        print(f"Data shape: {sample_data.shape}")
        print(f"Data type: {sample_data.dtype}")
        print(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        print(f"Data mean: {sample_data.mean():.3f}")
        print(f"Data std: {sample_data.std():.3f}")
        
        # Label statistics
        if hasattr(sample_labels, 'unique'):
            unique_labels = sample_labels.unique()
            print(f"Number of unique labels: {len(unique_labels)}")
            print(f"Label range: [{sample_labels.min()}, {sample_labels.max()}]")
    
    except Exception as e:
        print(f"Error computing statistics: {e}")


def save_dataset_samples(loader, save_dir, dataset_name, num_samples=100):
    """
    Save dataset samples to disk for inspection
    
    Args:
        loader: DataLoader
        save_dir: Directory to save samples
        dataset_name: Name for saved files
        num_samples: Number of samples to save
    """
    import os
    from torchvision.utils import save_image
    
    os.makedirs(save_dir, exist_ok=True)
    
    saved_count = 0
    for batch_idx, (data, labels) in enumerate(loader):
        for i in range(data.size(0)):
            if saved_count >= num_samples:
                break
            
            # Save image
            filename = f"{dataset_name}_sample_{saved_count:04d}_label_{labels[i].item()}.png"
            filepath = os.path.join(save_dir, filename)
            
            # Denormalize if needed
            img = data[i]
            if img.min() < 0:
                img = (img + 1) / 2
            
            save_image(img, filepath)
            saved_count += 1
        
        if saved_count >= num_samples:
            break
    
    print(f"Saved {saved_count} samples to {save_dir}")


def create_train_val_split(dataset, val_ratio=0.2, random_seed=42):
    """
    Split dataset into training and validation sets
    
    Args:
        dataset: PyTorch dataset
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility
    """
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


class DataAugmentation:
    """Data augmentation utilities for autoencoders"""
    
    @staticmethod
    def get_image_transforms(augment=True, image_size=None, normalize=True):
        """Get image transformation pipeline"""
        transforms_list = []
        
        # Resize if specified
        if image_size:
            transforms_list.append(transforms.Resize(image_size))
        
        # Augmentations
        if augment:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        
        # Convert to tensor
        transforms_list.append(transforms.ToTensor())
        
        # Normalization
        if normalize:
            transforms_list.append(transforms.Normalize((0.5,), (0.5,)))  # Generic normalization
        
        return transforms.Compose(transforms_list)
    
    @staticmethod
    def add_noise_transform(noise_type='gaussian', noise_factor=0.1):
        """Create noise transformation"""
        class AddNoise:
            def __init__(self, noise_type, noise_factor):
                self.noise_type = noise_type
                self.noise_factor = noise_factor
            
            def __call__(self, tensor):
                if self.noise_type == 'gaussian':
                    noise = torch.randn_like(tensor) * self.noise_factor
                    return torch.clamp(tensor + noise, 0, 1)
                elif self.noise_type == 'salt_pepper':
                    noise_mask = torch.rand_like(tensor)
                    noisy = tensor.clone()
                    noisy[noise_mask < self.noise_factor/2] = 0
                    noisy[noise_mask > 1 - self.noise_factor/2] = 1
                    return noisy
                else:
                    return tensor
        
        return AddNoise(noise_type, noise_factor)


def benchmark_dataloader(loader, num_batches=10):
    """
    Benchmark dataloader performance
    
    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to time
    """
    import time
    
    print(f"🚀 Benchmarking DataLoader Performance")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of workers: {loader.num_workers}")
    
    # Warm up
    for i, batch in enumerate(loader):
        if i >= 2:
            break
    
    # Benchmark
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    
    print(f"Total time for {num_batches} batches: {total_time:.3f}s")
    print(f"Average time per batch: {avg_time_per_batch:.3f}s")
    print(f"Estimated samples per second: {loader.batch_size / avg_time_per_batch:.1f}")


def main():
    """Demo of data loading utilities"""
    print("📚 Data Loading Utilities Demo")
    
    # Test MNIST
    print("\n🔢 Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64, train_size=1000)
    get_dataset_stats(train_loader, "MNIST")
    visualize_dataset_samples(train_loader, "MNIST")
    
    # Test synthetic 2D data
    print("\n🎯 Loading Synthetic 2D Data...")
    synthetic_loader = get_synthetic_2d_data('blobs', n_samples=500, batch_size=64)
    get_dataset_stats(synthetic_loader, "Synthetic Blobs")
    visualize_dataset_samples(synthetic_loader, "Synthetic Blobs")
    
    # Test noisy dataset creation
    print("\n🔊 Creating Noisy Dataset...")
    noisy_loader = create_noisy_dataset(train_loader, 'gaussian', 0.3)
    
    # Benchmark performance
    print("\n⚡ Benchmarking Performance...")
    benchmark_dataloader(train_loader, num_batches=5)
    
    print("\n✅ Data loading utilities demo completed!")


if __name__ == "__main__":
    main()