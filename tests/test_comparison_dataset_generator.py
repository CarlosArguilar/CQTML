"""
Test the updated comparison dataset generator with GRPO-compatible model.

This test verifies that the comparison dataset generator correctly uses the
new generate_actions() interface and produces valid comparison datasets.
"""

import torch
import numpy as np
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel
from preferences.comparison_dataset_generator import ComparisonDatasetGenerator, generate_comparison_dataset


class MockDataset:
    """Mock dataset for testing comparison dataset generation"""
    
    def __init__(self, num_samples=10, cqt_shape=(2, 84, 100)):
        self.num_samples = num_samples
        self.cqt_shape = cqt_shape
        
        # Create mock chunk_to_file_map
        self.chunk_to_file_map = {}
        for i in range(num_samples):
            self.chunk_to_file_map[i] = (0, i)  # (file_idx, local_chunk_idx)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return synthetic CQT data
        return torch.randn(self.cqt_shape)
    
    def _get_audio_chunk(self, file_idx, local_chunk_idx):
        # Return synthetic audio data
        return torch.randn(22050)  # 1 second at 22050 Hz


def mock_compare_signals(audio_tensor, signal_a, signal_b):
    """Mock comparison function that returns random preference"""
    # Simple mock: prefer the signal with higher mean activation
    # Convert to numpy arrays if they're tensors
    if hasattr(signal_a, 'numpy'):
        signal_a = signal_a.numpy() if hasattr(signal_a, 'numpy') else signal_a
    if hasattr(signal_b, 'numpy'):
        signal_b = signal_b.numpy() if hasattr(signal_b, 'numpy') else signal_b
    
    mean_a = np.mean(signal_a)
    mean_b = np.mean(signal_b)
    
    if mean_a > mean_b:
        return 0  # Prefer A
    elif mean_b > mean_a:
        return 1  # Prefer B
    else:
        return 0  # Tie goes to A


def test_dual_temperature_generation():
    """Test comparison dataset generation with dual temperatures"""
    print("🔥 Testing dual temperature comparison generation...")
    
    # Create test model and dataset
    cqt_shape = (2, 84, 50)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=16,
        device='cpu'
    )
    
    dataset = MockDataset(num_samples=5, cqt_shape=cqt_shape)
    
    # Temporarily replace compare_signals
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        # Create generator
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            generator = ComparisonDatasetGenerator(
                model=model,
                dataset=dataset,
                save_path=tmp_file.name
            )
            
            # Generate comparisons
            generator.generate_comparison_dataset(
                num_samples=3,
                batch_size=1,
                verbose=True,
                temperature_a=0.3,  # Lower temperature
                temperature_b=1.8,  # Higher temperature
                actions_per_sample=3
            )
            
            # Verify results
            assert len(generator.comparisons) == 3, f"Expected 3 comparisons, got {len(generator.comparisons)}"
            
            # Check comparison structure
            comp = generator.comparisons[0]
            required_keys = ['input', 'audio', 'output_a', 'output_b', 'preference', 
                           'temperature_a', 'temperature_b', 'actions_per_sample']
            
            for key in required_keys:
                assert key in comp, f"Missing key: {key}"
            
            # Check data types and shapes
            assert isinstance(comp['input'], np.ndarray), "Input should be numpy array"
            assert isinstance(comp['audio'], np.ndarray), "Audio should be numpy array"
            assert isinstance(comp['output_a'], np.ndarray), "Output A should be numpy array"
            assert isinstance(comp['output_b'], np.ndarray), "Output B should be numpy array"
            assert comp['input'].shape == cqt_shape, f"Input shape mismatch: {comp['input'].shape}"
            assert comp['output_a'].shape == (3, 50), f"Output A shape mismatch: {comp['output_a'].shape}"
            assert comp['output_b'].shape == (3, 50), f"Output B shape mismatch: {comp['output_b'].shape}"
            
            # Check temperature values
            assert comp['temperature_a'] == 0.3, f"Temperature A mismatch: {comp['temperature_a']}"
            assert comp['temperature_b'] == 1.8, f"Temperature B mismatch: {comp['temperature_b']}"
            
            # Check preference values
            assert comp['preference'] in [0, 1], f"Invalid preference: {comp['preference']}"
            
            print("✅ Dual temperature generation works correctly")
            print(f"   - Generated {len(generator.comparisons)} comparisons")
            print(f"   - Temperature A: {comp['temperature_a']} (deterministic)")
            print(f"   - Temperature B: {comp['temperature_b']} (exploratory)")
            
        # Clean up
        os.unlink(tmp_file.name)
            
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def test_single_temperature_generation():
    """Test comparison dataset generation with single temperature"""
    print("\n🌡️ Testing single temperature comparison generation...")
    
    cqt_shape = (2, 84, 40)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=12,
        device='cpu'
    )
    
    dataset = MockDataset(num_samples=4, cqt_shape=cqt_shape)
    
    # Temporarily replace compare_signals
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            generator = ComparisonDatasetGenerator(
                model=model,
                dataset=dataset,
                save_path=tmp_file.name
            )
            
            # Generate comparisons using single temperature method
            generator.generate_comparison_dataset_single_temp(
                num_samples=2,
                batch_size=1,
                verbose=True,
                temperature=1.2,
                actions_per_sample=4
            )
            
            # Verify results
            assert len(generator.comparisons) == 2, f"Expected 2 comparisons, got {len(generator.comparisons)}"
            
            comp = generator.comparisons[0]
            
            # Check single temperature specific keys
            assert 'temperature' in comp, "Missing temperature key"
            assert 'comparison_type' in comp, "Missing comparison_type key"
            assert comp['comparison_type'] == 'single_temperature_variance', "Wrong comparison type"
            assert comp['temperature'] == 1.2, f"Temperature mismatch: {comp['temperature']}"
            
            print("✅ Single temperature generation works correctly")
            print(f"   - Generated {len(generator.comparisons)} comparisons")
            print(f"   - Temperature: {comp['temperature']}")
            print(f"   - Comparison type: {comp['comparison_type']}")
            
        # Clean up
        os.unlink(tmp_file.name)
            
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def test_save_and_load():
    """Test saving and loading comparison datasets"""
    print("\n💾 Testing save and load functionality...")
    
    cqt_shape = (2, 84, 30)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=8,
        device='cpu'
    )
    
    dataset = MockDataset(num_samples=3, cqt_shape=cqt_shape)
    
    # Temporarily replace compare_signals
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            generator = ComparisonDatasetGenerator(
                model=model,
                dataset=dataset,
                save_path=tmp_file.name
            )
            
            # Generate and save
            generator.generate_comparison_dataset(
                num_samples=2,
                temperature_a=0.5,
                temperature_b=1.5,
                verbose=False
            )
            
            original_comparisons = len(generator.comparisons)
            saved_path = generator.save_dataset()
            
            # Create new generator and load
            new_generator = ComparisonDatasetGenerator(
                model=model,
                dataset=dataset,
                save_path=tmp_file.name
            )
            
            dataset_info = new_generator.load_dataset()
            
            # Verify loaded data
            assert len(new_generator.comparisons) == original_comparisons, "Comparison count mismatch after load"
            assert 'metadata' in dataset_info, "Missing metadata"
            assert dataset_info['metadata']['generation_method'] == 'grpo_compatible', "Wrong generation method"
            
            print("✅ Save and load functionality works correctly")
            print(f"   - Saved {original_comparisons} comparisons")
            print(f"   - Loaded {len(new_generator.comparisons)} comparisons")
            
        # Clean up
        os.unlink(tmp_file.name)
            
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def test_training_data_conversion():
    """Test conversion to training data format"""
    print("\n🎯 Testing training data conversion...")
    
    cqt_shape = (2, 84, 25)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=6,
        device='cpu'
    )
    
    dataset = MockDataset(num_samples=3, cqt_shape=cqt_shape)
    
    # Temporarily replace compare_signals with predictable results
    def predictable_compare_signals(audio_tensor, signal_a, signal_b):
        # Alternate between preferences for testing
        return 0 if np.random.random() > 0.5 else 1
    
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = predictable_compare_signals
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            generator = ComparisonDatasetGenerator(
                model=model,
                dataset=dataset,
                save_path=tmp_file.name
            )
            
            generator.generate_comparison_dataset(
                num_samples=3,
                temperature_a=0.5,
                temperature_b=1.5,
                verbose=False
            )
            
            # Convert to training data
            training_data = generator.get_training_data()
            
            # Verify training data format
            assert len(training_data) <= len(generator.comparisons), "Too much training data"
            
            if training_data:  # If we have any training data
                sample = training_data[0]
                required_keys = ['input', 'preferred', 'rejected']
                
                for key in required_keys:
                    assert key in sample, f"Missing key in training data: {key}"
                
                # Check shapes
                assert sample['input'].shape == cqt_shape, "Input shape mismatch in training data"
                assert sample['preferred'].shape == (2, 25), "Preferred shape mismatch in training data"
                assert sample['rejected'].shape == (2, 25), "Rejected shape mismatch in training data"
                
                print("✅ Training data conversion works correctly")
                print(f"   - Generated {len(training_data)} training samples from {len(generator.comparisons)} comparisons")
            else:
                print("✅ Training data conversion works (no valid preferences generated)")
            
        # Clean up
        os.unlink(tmp_file.name)
            
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


def test_utility_functions():
    """Test utility functions"""
    print("\n🛠️ Testing utility functions...")
    
    cqt_shape = (2, 84, 20)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=4,
        device='cpu'
    )
    
    dataset = MockDataset(num_samples=2, cqt_shape=cqt_shape)
    
    # Temporarily replace compare_signals
    import preferences.comparison_dataset_generator as cdg
    original_compare_signals = cdg.compare_signals
    cdg.compare_signals = mock_compare_signals
    
    try:
        # Test quick comparison dataset
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            generator, saved_path = ComparisonDatasetGenerator.create_quick_comparison_dataset(
                model=model,
                dataset=dataset,
                num_samples=2,
                save_path=tmp_file.name,
                temperature_a=0.5,
                temperature_b=1.5,
                comparison_method='dual_temperature'
            )
            
            assert len(generator.comparisons) == 2, "Quick dataset generation failed"
            assert saved_path.exists(), "Dataset file not saved"
            
            print("✅ Quick comparison dataset creation works")
            
        # Clean up
        os.unlink(tmp_file.name)
        
        # Test generate_comparison_dataset function
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            saved_path = generate_comparison_dataset(
                model=model,
                dataset=dataset,
                num_samples=2,
                save_path=tmp_file.name,
                temperature_a=0.3,
                temperature_b=1.7,
                comparison_method='single_temperature'
            )
            
            assert saved_path.exists(), "Utility function dataset file not saved"
            
            print("✅ Utility function works correctly")
            
        # Clean up
        os.unlink(tmp_file.name)
            
    finally:
        # Restore original function
        cdg.compare_signals = original_compare_signals


if __name__ == "__main__":
    print("🚀 Running Comparison Dataset Generator Tests\n")
    
    try:
        test_dual_temperature_generation()
        test_single_temperature_generation()
        test_save_and_load()
        test_training_data_conversion()
        test_utility_functions()
        
        print("\n🎉 All comparison dataset generator tests passed!")
        print("✅ Updated generator is ready for GRPO-compatible models")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 