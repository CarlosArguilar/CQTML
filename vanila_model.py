import torch
from torch.utils.data import DataLoader
from freemusic import FreeMusic
from cqt_vit_model import CQTViTModel

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

# Create DataLoader for batch processing
dataloader = DataLoader(cqt_dataset, batch_size=2, shuffle=False, num_workers=0)  # Reduced batch size for efficiency

model.eval()
with torch.no_grad():
    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to same device as model
        batch_data = batch_data.to(next(model.parameters()).device)
        
        output = model(batch_data)
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {batch_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Input dtype: {batch_data.dtype}")
        print(f"  Output dtype: {output.dtype}")
        
        # Process only first batch for demonstration
        if batch_idx == 0:
            break

