import torch
from torch.utils.data import DataLoader
from freemusic import FreeMusic
from cqt_vit_model import CQTViTModel
from cqt_reward_model import CQTRewardModel

cqt_dataset = FreeMusic(
    output_format='cqt',
    verbose=True,
    max_duration=5.0
)

# Get example data to extract dimensions
example_cqt = cqt_dataset[0]  # Get first sample
print(f"Example CQT shape: {example_cqt.shape}")

# Create CQT-adapted ViT model with resource efficiency options
model = CQTViTModel.create_model(
    cqt_shape=example_cqt.shape, 
    num_devices=4,
    use_8bit=False,              # Set to True for 8-bit quantization (requires bitsandbytes)
    use_half_precision=True,     # Use FP16 for faster inference (if on GPU)
    gradient_checkpointing=True, # Save memory during training
    device='auto'                # Auto-detect best device
)

# Create reward model to evaluate input/output pairs
reward_model = CQTRewardModel.create_model(
    cqt_shape=example_cqt.shape,
    output_shape=(4, example_cqt.shape[-1]),  # 4 devices, same time dimension
    embed_dim=128,               # Explicit embedding dimension
    num_heads=4,                 # Number of attention heads
    dropout=0.1,                 # Dropout rate
    use_8bit=False,              # 8-bit quantization (set to True if needed)
    use_half_precision=True,     # FP16 for efficiency
    gradient_checkpointing=True, # Memory optimization
    device='auto'                # Auto-detect device
)

# Create DataLoader for batch processing
dataloader = DataLoader(cqt_dataset, batch_size=2, shuffle=False, num_workers=0)  # Reduced batch size for efficiency

model.eval()
reward_model.eval()

with torch.no_grad():
    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to same device as model
        batch_data = batch_data.to(next(model.parameters()).device)
        
        # Get CQTViTModel output
        output = model(batch_data)
        
        # Get reward scores for input/output pairs
        rewards = reward_model(batch_data, output)
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {batch_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Input dtype: {batch_data.dtype}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Reward scores: {rewards}")
        print(f"  Mean reward: {rewards.mean().item():.3f}")
        print(f"  Reward std: {rewards.std().item():.3f}")
        
        # Process only first batch for demonstration
        if batch_idx == 0:
            break

