import torch
from data.freemusic import FreeMusic
from models.cqt_vit_model import CQTViTModel
from preferences.comparison_dataset_generator import ComparisonDatasetGenerator

def demo_stochastic_generation():
    """
    Demonstration of state-of-the-art stochastic generation for CQT models
    """
    print("🎵 CQT Model Stochastic Generation Demo")
    print("=" * 50)
    
    # Load dataset and create model
    print("Loading dataset and model...")
    cqt_dataset = FreeMusic(output_format='cqt', verbose=False, max_duration=5.0)
    example_cqt = cqt_dataset[0]
    
    model = CQTViTModel.create_model(
        cqt_shape=example_cqt.shape, 
        num_devices=4,
        device='cpu',
        stochastic_inference=True,  # Enable stochastic generation
        dropout_prob=0.15  # Higher dropout for more variation
    )
    
    input_tensor = example_cqt.unsqueeze(0)
    model.eval()
    
    print(f"\n📊 Input CQT shape: {input_tensor.shape}")
    print(f"📱 Output devices: {4}")
    
    # Demonstrate different generation modes
    print("\n🎯 Generation Modes:")
    
    with torch.no_grad():
        # 1. Deterministic generation
        det_output = model.generate_deterministic(input_tensor)
        print(f"   Deterministic: Always produces the same output")
        print(f"   Shape: {det_output.shape}")
        
        # 2. Stochastic generation with different temperatures
        temperatures = [0.8, 1.2, 1.8]
        
        print(f"\n🌡️  Temperature Effects:")
        for temp in temperatures:
            stoch_output = model.generate_stochastic(input_tensor, temperature=temp)
            diff_from_det = torch.abs(det_output - stoch_output).max().item()
            variance = stoch_output.var().item()
            
            print(f"   Temp {temp:3.1f}: Diff={diff_from_det:6.3f}, Variance={variance:6.3f}")
        
        # 3. Multiple samples at same temperature
        print(f"\n🎲 Multiple Samples (temp=1.5):")
        samples = model.generate_stochastic(input_tensor, temperature=1.5, num_samples=3)
        
        for i in range(samples.shape[0]):
            sample_variance = samples[i].var().item()
            print(f"   Sample {i+1}: Variance={sample_variance:6.3f}")
    
    # Demonstrate comparison dataset generation
    print(f"\n📋 Comparison Dataset Generation:")
    print("   Creating comparison pairs for preference learning...")
    
    generator = ComparisonDatasetGenerator(
        model=model,
        dataset=cqt_dataset,
        save_path="demo_comparison_dataset.pkl"
    )
    
    # Generate a small demo dataset
    generator.generate_comparison_dataset(
        num_samples=5,
        batch_size=1,
        verbose=False,
        temperature=1.3,
        use_stochastic=True
    )
    
    print(f"   ✓ Generated {len(generator.comparisons)} comparison pairs")
    
    # Show statistics
    if generator.comparisons:
        diffs = []
        for comp in generator.comparisons:
            diff = abs(comp['output_a'] - comp['output_b']).max()
            diffs.append(diff)
        
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        min_diff = min(diffs)
        
        print(f"   Output differences: avg={avg_diff:.3f}, max={max_diff:.3f}, min={min_diff:.3f}")
    
    print(f"\n✨ Key Advantages of This Approach:")
    print(f"   • Deterministic mode: Reproducible outputs for testing")
    print(f"   • Temperature control: Adjustable randomness level")
    print(f"   • Internal randomness: Uses dropout and Gumbel sampling")
    print(f"   • Multiple samples: Generate diverse outputs from same input")
    print(f"   • State-of-the-art: Follows modern generative model practices")
    
    print(f"\n🎯 Perfect for:")
    print(f"   • Preference learning datasets")
    print(f"   • A/B testing of model outputs")
    print(f"   • Reward model training")
    print(f"   • Exploring model output diversity")
    
    print(f"\n✅ Demo completed!")

if __name__ == "__main__":
    demo_stochastic_generation() 