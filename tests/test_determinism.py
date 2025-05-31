import torch
from data.freemusic import FreeMusic
from models.cqt_vit_model import CQTViTModel

print("Testing model determinism...")

# Load dataset and create model
cqt_dataset = FreeMusic(output_format='cqt', verbose=False, max_duration=5.0)
example_cqt = cqt_dataset[0]

model = CQTViTModel.create_model(
    cqt_shape=example_cqt.shape, 
    num_devices=4,
    use_8bit=False,
    use_half_precision=False,
    gradient_checkpointing=False,
    device='cpu'
)

# Prepare input
input_tensor = example_cqt.unsqueeze(0)  # Add batch dimension
print(f"Input shape: {input_tensor.shape}")

# Set model to eval mode (important for determinism)
model.eval()

# Run multiple inferences with the same input
print("Running multiple inferences...")
outputs = []

with torch.no_grad():
    for i in range(5):
        output = model(input_tensor)
        outputs.append(output.clone())
        print(f"Run {i+1}: Output shape {output.shape}, Mean {output.mean().item():.6f}")

# Check if all outputs are identical
print("\nChecking determinism...")
all_same = True
for i in range(1, len(outputs)):
    if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
        all_same = False
        diff = torch.abs(outputs[0] - outputs[i]).max().item()
        print(f"Output {i+1} differs from output 1, max diff: {diff}")

if all_same:
    print("✓ Model is DETERMINISTIC - all outputs are identical")
else:
    print("✗ Model is NON-DETERMINISTIC - outputs differ between runs")

# Test with different random seeds
print("\nTesting with different random seeds...")
torch.manual_seed(42)
output_seed42 = model(input_tensor).clone()

torch.manual_seed(123)  
output_seed123 = model(input_tensor).clone()

if torch.allclose(output_seed42, output_seed123, atol=1e-6):
    print("✓ Model output is consistent across different random seeds")
else:
    diff = torch.abs(output_seed42 - output_seed123).max().item()
    print(f"✗ Model output varies with random seed, max diff: {diff}")

print("Determinism test completed!") 