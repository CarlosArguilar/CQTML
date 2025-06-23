"""
Demo: Generate comparison dataset with GRPO-compatible CQT ViT model

This demo shows how to use the updated comparison dataset generator
that works with the new GRPO-compatible model interface.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel
from preferences.comparison_dataset_generator import ComparisonDatasetGenerator, generate_comparison_dataset


class SyntheticCQTDataset:
    """Synthetic CQT dataset for demonstration"""
    
    def __init__(self, num_samples=20, cqt_shape=(2, 84, 128)):
        self.num_samples = num_samples
        self.cqt_shape = cqt_shape
        
        # Create realistic chunk mapping
        self.chunk_to_file_map = {}
        chunks_per_file = 5
        for i in range(num_samples):
            file_idx = i // chunks_per_file
            local_chunk_idx = i % chunks_per_file
            self.chunk_to_file_map[i] = (file_idx, local_chunk_idx)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate more realistic CQT-like data
        # Lower frequencies have more energy, sparse high frequencies
        data = torch.zeros(self.cqt_shape)
        for i in range(self.cqt_shape[0]):  # channels
            for j in range(self.cqt_shape[1]):  # frequency bins
                # Create frequency-dependent energy
                energy_factor = np.exp(-j / 20.0)  # Lower frequencies stronger
                data[i, j, :] = torch.randn(self.cqt_shape[2]) * energy_factor
        
        return data
    
    def _get_audio_chunk(self, file_idx, local_chunk_idx):
        # Generate synthetic audio with some structure
        duration_samples = 22050  # 1 second at 22kHz
        t = np.linspace(0, 1, duration_samples)
        
        # Create harmonic content
        audio = np.zeros(duration_samples)
        for harmonic in [220, 440, 880]:  # Some musical notes
            audio += 0.1 * np.sin(2 * np.pi * harmonic * t) * np.exp(-t * 2)
        
        # Add some noise
        audio += 0.05 * np.random.randn(duration_samples)
        
        return torch.tensor(audio, dtype=torch.float32)


def mock_compare_signals(audio_tensor, signal_a, signal_b):
    """
    Mock comparison function for demonstration
    
    In real usage, this would be the cqtml_interface.api.compare_signals function
    that plays audio and gets human preferences.
    """
    # Convert tensors to numpy if needed
    if hasattr(signal_a, 'numpy'):
        signal_a = signal_a.numpy()
    if hasattr(signal_b, 'numpy'):
        signal_b = signal_b.numpy()
    
    # Mock preference based on signal characteristics
    # Prefer signals with more balanced activation across devices
    variance_a = np.var(signal_a, axis=0).mean()  # Variance across devices
    variance_b = np.var(signal_b, axis=0).mean()
    
    # Prefer more balanced (less variable) signals
    if variance_a < variance_b:
        return 0  # Prefer A
    elif variance_b < variance_a:
        return 1  # Prefer B
    else:
        # For ties, prefer signal with higher overall activation
        if np.mean(signal_a) > np.mean(signal_b):
            return 0
        else:
            return 1


def demo_dual_temperature_comparison():
    """Demo: Generate comparisons using dual temperature approach"""
    print("🔥 Demo: Dual Temperature Comparison Generation")
    print("=" * 50)
    
    # Create model and dataset
    cqt_shape = (2, 84, 128)  # Stereo, 84 frequency bins, 128 time steps
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=32,
        device='cpu'
    )
    
    dataset = SyntheticCQTDataset(num_samples=10, cqt_shape=cqt_shape)
    
    # Replace comparison function with mock for demo
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        # Create generator
        generator = ComparisonDatasetGenerator(
            model=model,
            dataset=dataset,
            save_path="demo_dual_temp_comparisons.pkl"
        )
        
        print(f"Generating comparisons from {len(dataset)} CQT samples...")
        print("Using dual temperature approach:")
        print("  - Temperature A: 0.3 (more deterministic/conservative)")
        print("  - Temperature B: 1.8 (more exploratory/diverse)")
        
        # Generate comparisons
        generator.generate_comparison_dataset(
            num_samples=8,
            batch_size=1,
            verbose=True,
            temperature_a=0.3,
            temperature_b=1.8,
            actions_per_sample=3
        )
        
        # Show results
        print(f"\n📊 Generated {len(generator.comparisons)} comparison pairs")
        
        # Analyze preferences
        preferences = [comp['preference'] for comp in generator.comparisons]
        pref_counts = {0: preferences.count(0), 1: preferences.count(1)}
        
        print(f"Preference distribution:")
        print(f"  - Conservative (A): {pref_counts[0]} ({pref_counts[0]/len(preferences)*100:.1f}%)")
        print(f"  - Exploratory (B): {pref_counts[1]} ({pref_counts[1]/len(preferences)*100:.1f}%)")
        
        # Save dataset
        saved_path = generator.save_dataset()
        print(f"💾 Saved comparison dataset to: {saved_path}")
        
        # Convert to training data
        training_data = generator.get_training_data()
        print(f"🎯 Converted to {len(training_data)} training samples for reward model")
        
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def demo_single_temperature_comparison():
    """Demo: Generate comparisons using single temperature approach"""
    print("\n🌡️ Demo: Single Temperature Comparison Generation")
    print("=" * 50)
    
    cqt_shape = (2, 84, 96)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=24,
        device='cpu'
    )
    
    dataset = SyntheticCQTDataset(num_samples=8, cqt_shape=cqt_shape)
    
    # Replace comparison function with mock for demo
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        generator = ComparisonDatasetGenerator(
            model=model,
            dataset=dataset,
            save_path="demo_single_temp_comparisons.pkl"
        )
        
        print(f"Generating comparisons from {len(dataset)} CQT samples...")
        print("Using single temperature approach:")
        print("  - Temperature: 1.2 (moderate exploration)")
        print("  - Comparing most vs least diverse actions")
        
        # Generate comparisons using single temperature method
        generator.generate_comparison_dataset_single_temp(
            num_samples=6,
            batch_size=1,
            verbose=True,
            temperature=1.2,
            actions_per_sample=5
        )
        
        # Show results
        print(f"\n📊 Generated {len(generator.comparisons)} comparison pairs")
        
        # Show example comparison
        if generator.comparisons:
            comp = generator.comparisons[0]
            print(f"\nExample comparison:")
            print(f"  - Input shape: {comp['input'].shape}")
            print(f"  - Output A shape: {comp['output_a'].shape} (less diverse)")
            print(f"  - Output B shape: {comp['output_b'].shape} (more diverse)")
            print(f"  - Preference: {'A (less diverse)' if comp['preference'] == 0 else 'B (more diverse)'}")
        
        # Save dataset
        saved_path = generator.save_dataset()
        print(f"💾 Saved comparison dataset to: {saved_path}")
        
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def demo_utility_functions():
    """Demo: Use utility functions for quick dataset generation"""
    print("\n🛠️ Demo: Utility Functions")
    print("=" * 30)
    
    cqt_shape = (2, 84, 64)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=16,
        device='cpu'
    )
    
    dataset = SyntheticCQTDataset(num_samples=5, cqt_shape=cqt_shape)
    
    # Replace comparison function with mock for demo
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        print("Using quick comparison dataset creation...")
        
        # Method 1: Quick creation with dual temperature
        generator, saved_path = ComparisonDatasetGenerator.create_quick_comparison_dataset(
            model=model,
            dataset=dataset,
            num_samples=3,
            save_path="demo_quick_dual.pkl",
            temperature_a=0.4,
            temperature_b=1.6,
            comparison_method='dual_temperature'
        )
        
        print(f"✅ Quick dual temperature: {len(generator.comparisons)} comparisons saved to {saved_path}")
        
        # Method 2: Simple function call
        saved_path = generate_comparison_dataset(
            model=model,
            dataset=dataset,
            num_samples=3,
            save_path="demo_simple_function.pkl",
            temperature_a=0.5,
            temperature_b=1.5,
            comparison_method='single_temperature'
        )
        
        print(f"✅ Simple function: comparison dataset saved to {saved_path}")
        
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def demonstrate_action_generation():
    """Show the action generation capabilities"""
    print("\n🎮 Demo: Model Action Generation")
    print("=" * 35)
    
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=32,
        device='cpu'
    )
    
    # Generate sample CQT data
    sample_cqt = torch.randn(cqt_shape)
    
    print(f"Input CQT shape: {sample_cqt.shape}")
    print(f"Model configuration:")
    print(f"  - Devices: {model.num_devices}")
    print(f"  - Distribution size: {model.distribution_size}")
    
    # Generate actions with different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        actions = model.generate_actions(
            states=[sample_cqt],
            num_actions_per_state=3,
            temperature=temp
        )
        
        action_array = np.array(actions[0])  # First state's actions
        action_variance = np.var(action_array, axis=0).mean()  # Variance across actions
        action_range = np.ptp(action_array)  # Peak-to-peak range
        
        print(f"\nTemperature {temp}:")
        print(f"  - Action shape: {action_array.shape}")
        print(f"  - Average variance: {action_variance:.4f}")
        print(f"  - Value range: {action_range:.4f}")
        print(f"  - Sample action mean: {np.mean(action_array[0]):.4f}")


if __name__ == "__main__":
    print("🚀 GRPO-Compatible Comparison Dataset Generator Demo")
    print("=" * 60)
    print()
    
    try:
        # Run all demos
        demonstrate_action_generation()
        demo_dual_temperature_comparison()
        demo_single_temperature_comparison()
        demo_utility_functions()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📝 Key Features Demonstrated:")
        print("✅ GRPO-compatible model action generation")
        print("✅ Dual temperature comparison generation")
        print("✅ Single temperature variance-based comparison")
        print("✅ Save/load comparison datasets")
        print("✅ Convert to training data format")
        print("✅ Utility functions for quick usage")
        
        print("\n🔗 Next Steps:")
        print("1. Replace mock_compare_signals with real cqtml_interface.api.compare_signals")
        print("2. Use real audio dataset instead of synthetic data")
        print("3. Train reward model on generated comparison data")
        print("4. Use reward model with GRPO for policy optimization")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 