# CQTML - Constant-Q Transform Machine Learning

A comprehensive framework for audio processing and machine learning using Constant-Q Transform (CQT) representations, featuring Vision Transformer models adapted for GRPO (Generalized Relative Policy Optimization) training, preference learning, and probabilistic generation capabilities.

## 🎵 Overview

CQTML provides state-of-the-art tools for:
- **CQT-based audio processing** with optimized parameter handling
- **GRPO-compatible Vision Transformer models** with categorical probability distributions
- **Policy gradient optimization** with log probability computation
- **Multi-action generation** from single states for reinforcement learning
- **Reward model training** for audio quality assessment

## 📁 Project Structure

```
CQTML/
├── models/              # Neural network model definitions
│   ├── cqt_vit_model.py      # GRPO-compatible CQT Vision Transformer
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
│   ├── test_grpo_interface.py     # Test GRPO PolicyModel interface
│   ├── test_log_probabilities.py # Test log probability computation
│   ├── test_action_generation.py # Test multi-action generation
│   └── demo_grpo_workflow.py     # GRPO training demonstration
├── notebooks/           # Jupyter notebooks for experimentation
├── cqtml_interface/     # External API interfaces
└── utils/               # General utilities
```

## 🚀 Key Features

### 🎯 GRPO-Compatible CQT Vision Transformer
- **PolicyModel protocol** implementation for GRPO training frameworks
- **3D output tensors** with log probability distributions `[batch_size, num_devices, T, distribution_size]`
- **Categorical sampling** with temperature control for action generation
- **Log probability computation** for policy gradient updates
- **Resource optimization** with 8-bit quantization, FP16, gradient checkpointing

### 🏆 Policy Gradient Training
- **Multi-action generation** for exploration and comparison
- **Differentiable log probability** calculation for gradient flow
- **Temperature-controlled sampling** for exploration vs exploitation
- **Normalized action values** in `[0, 1]` range for continuous control

### 📊 Advanced Generation Modes
- **Action generation**: Generate multiple diverse actions per state
- **Log probability evaluation**: Compute action probabilities for training
- **Temperature scaling**: Control exploration vs exploitation trade-off
- **Gradient-preserving computation**: Full backpropagation support

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

### Basic GRPO Model Usage
```python
from data.freemusic import FreeMusic
from models.cqt_vit_model import CQTViTModel

# Load dataset
dataset = FreeMusic(output_format='cqt', max_duration=5.0)
example_cqt = dataset[0]

# Create GRPO-compatible model
model = CQTViTModel.create_model(
    cqt_shape=example_cqt.shape,
    num_devices=4,
    distribution_size=32,  # Size of categorical distribution
    use_half_precision=True
)

# Generate multiple actions for exploration
states = [example_cqt, example_cqt]  # List of CQT inputs
actions = model.generate_actions(
    states=states,
    num_actions_per_state=4,
    temperature=1.2
)

# Compute log probabilities for policy gradient
selected_actions = [actions[0][0], actions[1][0]]  # First action per state
log_probs = model.get_log_probabilities(states, selected_actions)
```

### GRPO Training Integration
```python
# PolicyModel protocol methods
class GRPOTrainer:
    def __init__(self, policy_model):
        self.policy = policy_model
    
    def train_step(self, states, rewards):
        # Generate actions for each state
        actions = self.policy.generate_actions(
            states=states,
            num_actions_per_state=2,
            temperature=1.0
        )
        
        # Compute log probabilities
        log_probs = self.policy.get_log_probabilities(states, actions)
        
        # Policy gradient computation
        policy_loss = -(log_probs * rewards).mean()
        policy_loss.backward()
        
        return policy_loss
```

### Advanced Configuration
```python
# Model with custom distribution size and optimizations
model = CQTViTModel.create_model(
    cqt_shape=(2, 84, 1000),
    num_devices=4,
    distribution_size=64,        # Larger distribution for finer control
    use_8bit=True,              # Memory optimization
    use_half_precision=True,    # Speed optimization
    gradient_checkpointing=True # Memory optimization
)

# Multi-temperature sampling for diverse exploration
diverse_actions = []
for temp in [0.5, 1.0, 1.5, 2.0]:
    actions = model.generate_actions(
        states=[cqt_input],
        num_actions_per_state=1,
        temperature=temp
    )
    diverse_actions.extend(actions[0])
```

## 🧪 Testing and Demos

### Run Tests
```bash
# Test GRPO PolicyModel interface compliance
python tests/test_grpo_interface.py

# Test log probability computation and gradients
python tests/test_log_probabilities.py

# Test multi-action generation capabilities
python tests/test_action_generation.py

# Run complete GRPO workflow demonstration
python tests/demo_grpo_workflow.py
```

### Expected Results
- **PolicyModel compliance**: All required methods implemented correctly
- **Gradient flow**: Log probabilities maintain gradients for backpropagation
- **Action diversity**: Different actions generated with temperature scaling
- **Probability consistency**: Log probabilities correctly computed for given actions

## 🎛️ Configuration Options

### Model Architecture
```python
model = CQTViTModel.create_model(
    cqt_shape=(2, 84, T),
    num_devices=4,              # Number of output devices
    distribution_size=32,       # Categorical distribution size
    patch_size=16,              # ViT patch size
    device='auto'               # Auto-detect optimal device
)
```

### Resource Optimization
```python
model = CQTViTModel.create_model(
    cqt_shape=cqt_shape,
    use_8bit=True,              # 8-bit quantization
    use_half_precision=True,    # FP16 precision
    gradient_checkpointing=True # Memory optimization
)
```

### Generation Parameters
```python
actions = model.generate_actions(
    states=states,
    num_actions_per_state=4,    # Multiple actions per state
    temperature=1.2             # Exploration control
)
```

## 📈 Performance Features

### Memory Optimization
- **Gradient checkpointing** for reduced memory during training
- **8-bit quantization** with bitsandbytes for inference
- **Half precision** computation for speed improvements
- **Efficient categorical sampling** with torch.multinomial

### Training Efficiency
- **Preserved gradients** in log probability computation
- **Vectorized operations** for batch processing
- **Temperature scaling** for exploration control
- **PolicyModel protocol** for framework compatibility

## 🔬 Technical Details

### Output Architecture
- **3D probability tensors**: `[batch_size, num_devices, T, distribution_size]`
- **Categorical distributions**: Each time step and device has own distribution
- **Log-softmax normalization**: Proper probability distributions
- **Action sampling**: Multinomial sampling from categorical distributions

### Action Representation
- **Normalized values**: Actions in `[0, 1]` range for audio device activation
- **Index mapping**: `sampled_index / (distribution_size - 1)`
- **Continuous interpretation**: Suitable for continuous control tasks
- **Gradient compatibility**: Differentiable through sampling process

### GRPO Integration
- **PolicyModel protocol**: Duck-typed interface for GRPO frameworks
- **Multi-action generation**: Required for policy gradient algorithms
- **Log probability computation**: Maintains gradients for training
- **Parameter access**: Direct access to model parameters for optimization

### Model Architecture
- Based on `vit_tiny_patch16_224` with adaptations for CQT data
- Modified input projection for 2-channel CQT input
- Interpolated positional embeddings for variable-length sequences
- LogProbabilityHead for categorical distribution output

### Probability Distributions
- Log-softmax activation for proper probability distributions
- Temperature scaling for exploration vs exploitation control
- Categorical sampling with torch.multinomial
- Gradient-preserving log probability computation
