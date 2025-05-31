import torch
from data.freemusic import FreeMusic
from models.cqt_vit_model import CQTViTModel
from preferences.comparison_dataset_generator import ComparisonDatasetGenerator

print("Testing advanced stochastic generation for comparison dataset...")

# Load dataset
cqt_dataset = FreeMusic(output_format='cqt', verbose=False, max_duration=5.0)
example_cqt = cqt_dataset[0]

# Create model with stochastic inference enabled
model = CQTViTModel.create_model(
    cqt_shape=example_cqt.shape, 
    num_devices=4,
    device='cpu',
    stochastic_inference=True,
    dropout_prob=0.1
)

print(f"Dataset size: {len(cqt_dataset)}")
print(f"Model has stochastic inference: {hasattr(model, 'generate_stochastic')}")

# Test different generation modes
input_tensor = example_cqt.unsqueeze(0)
model.eval()

print("\n=== Testing Generation Modes ===")

# Test 1: Deterministic vs Deterministic (should be identical)
print("\n1. Deterministic vs Deterministic:")
with torch.no_grad():
    det_output_1 = model.generate_deterministic(input_tensor)
    det_output_2 = model.generate_deterministic(input_tensor)

max_diff = torch.abs(det_output_1 - det_output_2).max().item()
print(f"   Max difference: {max_diff:.8f}")
print(f"   ✓ Identical: {max_diff < 1e-6}")

# Test 2: Deterministic vs Stochastic (should be different)
print("\n2. Deterministic vs Stochastic (temp=1.2):")
with torch.no_grad():
    det_output = model.generate_deterministic(input_tensor)
    stoch_output = model.generate_stochastic(input_tensor, temperature=1.2)

max_diff = torch.abs(det_output - stoch_output).max().item()
print(f"   Max difference: {max_diff:.6f}")
print(f"   ✓ Different: {max_diff > 1e-6}")

# Test 3: Multiple stochastic samples (should all be different)
print("\n3. Multiple Stochastic Samples (temp=1.5):")
with torch.no_grad():
    stoch_samples = model.generate_stochastic(input_tensor, temperature=1.5, num_samples=3)

print(f"   Generated {stoch_samples.shape[0]} samples")
for i in range(stoch_samples.shape[0]):
    for j in range(i+1, stoch_samples.shape[0]):
        diff = torch.abs(stoch_samples[i] - stoch_samples[j]).max().item()
        print(f"   Sample {i} vs {j} diff: {diff:.6f}")

# Test 4: Temperature effects
print("\n4. Temperature Effects:")
temperatures = [0.5, 1.0, 1.5, 2.0]
with torch.no_grad():
    det_baseline = model.generate_deterministic(input_tensor)
    
    for temp in temperatures:
        stoch_output = model.generate_stochastic(input_tensor, temperature=temp)
        diff = torch.abs(det_baseline - stoch_output).max().item()
        variance = stoch_output.var().item()
        print(f"   Temp {temp}: Max diff = {diff:.6f}, Variance = {variance:.6f}")

print("\n=== Testing Comparison Dataset Generation ===")

# Test with different temperature settings
temperatures_to_test = [1.1, 1.5, 2.0]

for temp in temperatures_to_test:
    print(f"\nTesting with temperature = {temp}:")
    
    generator = ComparisonDatasetGenerator(
        model=model,
        dataset=cqt_dataset,
        save_path=f"test_stochastic_{temp:.1f}.pkl"
    )
    
    # Generate a small dataset to test 
    generator.generate_comparison_dataset(
        num_samples=2,
        batch_size=1,
        verbose=False,
        temperature=temp,
        use_stochastic=True
    )
    
    print(f"  Generated {len(generator.comparisons)} comparison pairs")
    if generator.comparisons:
        comp = generator.comparisons[0]
        diff = abs(comp['output_a'] - comp['output_b']).max()
        print(f"  Max output difference: {diff:.6f}")

print("\n✓ Advanced stochastic generation test completed!")
print("\nKey features demonstrated:")
print("  - Deterministic generation (reproducible)")
print("  - Stochastic generation with temperature control")
print("  - Multiple sample generation")
print("  - Temperature-controlled randomness")
print("  - Dropout-based internal randomness") 