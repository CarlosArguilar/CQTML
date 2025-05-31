# CQTML - Constant-Q Transform Machine Learning

A comprehensive framework for audio processing and machine learning using Constant-Q Transform (CQT) representations, featuring Vision Transformer models, preference learning, and advanced stochastic generation capabilities.

## 🎵 Overview

CQTML provides state-of-the-art tools for:
- **CQT-based audio processing** with optimized parameter handling
- **Vision Transformer models** adapted for CQT spectrograms
- **Preference learning** and comparison dataset generation
- **Stochastic generation** with temperature control and internal randomness
- **Reward model training** for audio quality assessment

## 📁 Project Structure

```
CQTML/
├── models/              # Neural network model definitions
│   ├── cqt_vit_model.py      # CQT-adapted Vision Transformer
│   ├── cqt_reward_model.py   # Reward model for preference learning
│   └── vanila_model.py       # Baseline models
├── data/                # Dataset classes and loaders
│   ├── freemusic.py          # Free Music Archive dataset
│   └── musicnet.py           # MusicNet dataset interface
├── preferences/         # Preference learning utilities
│   └── comparison_dataset_generator.py  # Generate comparison datasets
├── core/                # Core processing utilities
│   ├── cqtml.py              # CQT processor and utilities
│   └── main.py               # Main application entry
├── tests/               # Test scripts and demonstrations
│   ├── test_comparison.py         # Test stochastic generation
│   ├── test_determinism.py       # Test model determinism
│   └── demo_stochastic_generation.py  # Feature demonstrations
├── notebooks/           # Jupyter notebooks for experimentation
├── cqtml_interface/     # External API interfaces
└── utils/               # General utilities
```

## 🚀 Key Features

### 🎯 CQT Vision Transformer
- **Adaptive architecture** for CQT spectrograms with shape `[batch_size, 2, 84, T]`
- **Resource optimization** with 8-bit quantization, FP16, gradient checkpointing
- **Flexible output** mapping to `[batch_size, 4, T]` for multi-device generation
- **Stochastic inference** with temperature control and dropout-based randomness

### 🏆 Preference Learning
- **Comparison dataset generation** for preference learning and RLHF
- **Reward model architecture** with dual encoders and cross-attention fusion
- **Advanced stochastic sampling** ensuring meaningful output diversity
- **Temperature-controlled generation** for fine-tuned randomness

### 📊 Advanced Generation Modes
- **Deterministic generation**: Reproducible outputs for testing
- **Stochastic generation**: Temperature-controlled randomness
- **Multiple sampling**: Generate diverse outputs from same input
- **Internal randomness**: Uses dropout and Gumbel sampling techniques

## 🛠️ Installation

### Prerequisites
```bash
pip install torch torchvision
pip install timm transformers datasets
pip install librosa numpy tqdm
pip install bitsandbytes  # Optional: for 8-bit quantization
```

### Setup
```bash
git clone <repository-url>
cd CQTML
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## 💻 Usage Examples

### Basic CQT Model Usage
```python
from data.freemusic import FreeMusic
from models.cqt_vit_model import CQTViTModel

# Load dataset
dataset = FreeMusic(output_format='cqt', max_duration=5.0)
example_cqt = dataset[0]

# Create model with stochastic capabilities
model = CQTViTModel.create_model(
    cqt_shape=example_cqt.shape,
    num_devices=4,
    stochastic_inference=True,
    dropout_prob=0.1
)

# Generate outputs
input_tensor = example_cqt.unsqueeze(0)
deterministic_output = model.generate_deterministic(input_tensor)
stochastic_output = model.generate_stochastic(input_tensor, temperature=1.2)
```

### Preference Dataset Generation
```python
from preferences.comparison_dataset_generator import ComparisonDatasetGenerator

# Create comparison dataset for preference learning
generator = ComparisonDatasetGenerator(
    model=model,
    dataset=dataset,
    save_path="preference_dataset.pkl"
)

# Generate comparison pairs
generator.generate_comparison_dataset(
    num_samples=1000,
    temperature=1.3,
    use_stochastic=True
)
```

### Reward Model Training
```python
from models.cqt_reward_model import CQTRewardModel

# Create reward model for preference learning
reward_model = CQTRewardModel.create_model(
    cqt_shape=example_cqt.shape,
    output_shape=(4, example_cqt.shape[-1]),
    device='cuda'
)

# Evaluate input/output pairs
reward_score = reward_model(input_cqt, model_output)
```

## 🧪 Testing and Demos

### Run Tests
```bash
# Test stochastic generation capabilities
python tests/test_comparison.py

# Test model determinism
python tests/test_determinism.py

# Run full feature demonstration
python tests/demo_stochastic_generation.py
```

### Expected Results
- **Deterministic mode**: Identical outputs across runs
- **Stochastic mode**: Controlled variation with temperature
- **Comparison generation**: Meaningful differences for preference learning

## 🎛️ Configuration Options

### Model Optimization
```python
model = CQTViTModel.create_model(
    cqt_shape=cqt_shape,
    use_8bit=True,              # 8-bit quantization
    use_half_precision=True,    # FP16 precision
    gradient_checkpointing=True, # Memory optimization
    stochastic_inference=True,   # Enable stochastic generation
    dropout_prob=0.15           # Internal randomness level
)
```

### Dataset Configuration
```python
dataset = FreeMusic(
    sample_rate=22050,
    max_duration=30.0,
    output_format='cqt',        # 'audio' or 'cqt'
    cache_cqt=True,            # Cache CQT transforms
    cqt_params={               # Custom CQT parameters
        'hop_length': 64,
        'n_bins': 84,
        'bins_per_octave': 12
    }
)
```

## 📈 Performance Features

### Memory Optimization
- **Gradient checkpointing** for reduced memory usage
- **8-bit quantization** with bitsandbytes
- **CQT caching** for faster data loading
- **Batch processing** with configurable sizes

### Generation Quality
- **Temperature sampling** for controlled randomness
- **Dropout-based variation** during inference
- **Multiple sample generation** from single input
- **State-of-the-art techniques** following modern generative models

## 🔬 Technical Details

### CQT Processing
- Default parameters optimized for music: 84 bins, 12 bins/octave
- Hop length of 64 samples for temporal resolution
- Automatic resampling to target sample rate (22050 Hz)

### Model Architecture
- Based on `vit_tiny_patch16_224` with adaptations for CQT data
- Modified input projection for 2-channel CQT input
- Interpolated positional embeddings for variable-length sequences
- Custom output head for multi-device generation

### Stochastic Generation
- Temperature-controlled Gumbel sampling
- Inference-time dropout for internal randomness
- Multiple generation modes (deterministic/stochastic)
- Controllable diversity for preference learning
