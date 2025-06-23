"""
Test multi-action generation capabilities for GRPO training.

This test focuses on the action generation functionality, including
temperature scaling, diversity, and consistency checks.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel


def test_multi_action_generation():
    """Test generation of multiple actions per state"""
    print("🎯 Testing multi-action generation...")
    
    cqt_shape = (2, 84, 50)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    # Create test states
    states = [torch.randn(2, 84, 50) for _ in range(2)]
    
    # Generate multiple actions per state
    num_actions_per_state = 5
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=num_actions_per_state,
        temperature=1.0
    )
    
    # Verify structure
    assert len(actions) == len(states), f"Expected {len(states)} state results"
    
    for i, state_actions in enumerate(actions):
        assert len(state_actions) == num_actions_per_state, \
            f"State {i}: expected {num_actions_per_state} actions"
        
        for j, action in enumerate(state_actions):
            assert isinstance(action, np.ndarray), f"Action {i},{j} should be numpy array"
            assert action.shape == (4, 50), f"Action {i},{j} shape should be (4, 50)"
            assert 0.0 <= action.min() and action.max() <= 1.0, \
                f"Action {i},{j} values should be in [0,1]"
    
    print("✅ Multi-action generation works correctly")
    print(f"   - Generated {len(states)} state results")
    print(f"   - Each with {num_actions_per_state} actions")
    print(f"   - Actions shape: (4, 50)")


def test_action_diversity():
    """Test that generated actions are diverse"""
    print("\n🌈 Testing action diversity...")
    
    cqt_shape = (2, 84, 40)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=24,
        device='cpu'
    )
    
    # Generate multiple actions from same state
    state = torch.randn(2, 84, 40)
    actions = model.generate_actions(
        states=[state],
        num_actions_per_state=10,
        temperature=1.5
    )
    
    state_actions = np.array(actions[0])  # [10, 3, 40]
    
    # Calculate pairwise differences
    differences = []
    for i in range(len(state_actions)):
        for j in range(i + 1, len(state_actions)):
            diff = np.mean(np.abs(state_actions[i] - state_actions[j]))
            differences.append(diff)
    
    mean_difference = np.mean(differences)
    std_difference = np.std(differences)
    
    # Actions should be meaningfully different
    assert mean_difference > 0.05, f"Actions too similar: mean diff = {mean_difference:.4f}"
    
    print("✅ Generated actions are appropriately diverse")
    print(f"   - Mean pairwise difference: {mean_difference:.4f}")
    print(f"   - Std of differences: {std_difference:.4f}")
    print(f"   - Min difference: {min(differences):.4f}")
    print(f"   - Max difference: {max(differences):.4f}")


def test_temperature_scaling_effects():
    """Test temperature effects on action generation"""
    print("\n🌡️ Testing temperature scaling effects...")
    
    cqt_shape = (2, 84, 30)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=20,
        device='cpu'
    )
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    state = torch.randn(2, 84, 30)
    
    temp_stats = {}
    
    for temp in temperatures:
        # Generate actions with this temperature
        actions = model.generate_actions(
            states=[state],
            num_actions_per_state=8,
            temperature=temp
        )
        
        actions_array = np.array(actions[0])  # [8, 3, 30]
        
        # Calculate statistics
        variance = np.var(actions_array, axis=0).mean()
        mean_val = np.mean(actions_array)
        
        temp_stats[temp] = {
            'variance': variance,
            'mean': mean_val,
            'actions': actions_array
        }
        
        print(f"   - Temperature {temp}: variance = {variance:.4f}, mean = {mean_val:.3f}")
    
    # Verify temperature trends
    variances = [temp_stats[t]['variance'] for t in temperatures]
    
    # Higher temperature should generally lead to higher variance
    assert variances[-1] > variances[0], \
        f"High temp variance ({variances[-1]:.4f}) should be > low temp variance ({variances[0]:.4f})"
    
    print("✅ Temperature scaling works as expected")


def test_deterministic_behavior():
    """Test deterministic behavior with temperature=0 (approximately)"""
    print("\n🔒 Testing low-temperature deterministic behavior...")
    
    cqt_shape = (2, 84, 35)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=16,
        device='cpu'
    )
    
    state = torch.randn(2, 84, 35)
    
    # Generate actions with very low temperature multiple times
    low_temp_actions = []
    for _ in range(3):
        actions = model.generate_actions(
            states=[state],
            num_actions_per_state=1,
            temperature=0.01  # Very low temperature
        )
        low_temp_actions.append(actions[0][0])
    
    # Calculate variance across runs
    actions_array = np.array(low_temp_actions)
    variance = np.var(actions_array, axis=0).mean()
    
    print(f"   - Low temperature variance: {variance:.6f}")
    
    # Should have very low variance (nearly deterministic)
    assert variance < 0.01, f"Low temperature should produce low variance: {variance:.6f}"
    
    print("✅ Low temperature produces nearly deterministic behavior")


def test_action_value_distribution():
    """Test distribution of action values"""
    print("\n📊 Testing action value distribution...")
    
    cqt_shape = (2, 84, 45)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=32,
        device='cpu'
    )
    
    # Generate many actions to analyze distribution
    states = [torch.randn(2, 84, 45) for _ in range(5)]
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=4,
        temperature=1.0
    )
    
    # Flatten all actions
    all_actions = []
    for state_actions in actions:
        all_actions.extend(state_actions)
    
    all_values = np.concatenate([action.flatten() for action in all_actions])
    
    # Analyze distribution
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    # Check distribution properties
    assert 0.0 <= min_val, f"Minimum value should be >= 0.0: {min_val}"
    assert max_val <= 1.0, f"Maximum value should be <= 1.0: {max_val}"
    assert 0.2 <= mean_val <= 0.8, f"Mean should be reasonable: {mean_val}"
    assert std_val > 0.1, f"Standard deviation should show variation: {std_val}"
    
    print("✅ Action value distribution is reasonable")
    print(f"   - Mean: {mean_val:.3f}")
    print(f"   - Std: {std_val:.3f}")
    print(f"   - Range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"   - Total values analyzed: {len(all_values)}")


def test_different_input_formats():
    """Test action generation with different input formats"""
    print("\n🔄 Testing different input formats...")
    
    cqt_shape = (2, 84, 25)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=12,
        device='cpu'
    )
    
    # Test with different input types
    base_tensor = torch.randn(2, 84, 25)
    
    test_cases = [
        # Case 1: Pure torch tensor
        base_tensor,
        # Case 2: Numpy array
        base_tensor.numpy(),
        # Case 3: Python list (will be converted)
        base_tensor.tolist(),
    ]
    
    for i, test_input in enumerate(test_cases):
        print(f"   - Testing input format {i + 1}: {type(test_input)}")
        
        try:
            actions = model.generate_actions(
                states=[test_input],
                num_actions_per_state=2,
                temperature=1.0
            )
            
            # Verify output
            assert len(actions) == 1, "Should have one state result"
            assert len(actions[0]) == 2, "Should have 2 actions per state"
            assert isinstance(actions[0][0], np.ndarray), "Actions should be numpy arrays"
            
            print(f"     ✅ Format {i + 1} works correctly")
            
        except Exception as e:
            print(f"     ❌ Format {i + 1} failed: {e}")
            raise
    
    print("✅ All input formats handled correctly")


def test_state_independence():
    """Test that different states produce different action distributions"""
    print("\n🎭 Testing state independence...")
    
    cqt_shape = (2, 84, 30)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=16,
        device='cpu'
    )
    
    # Create very different states
    state1 = torch.randn(2, 84, 30)
    state2 = torch.randn(2, 84, 30) * 3  # Different scale
    state3 = torch.zeros(2, 84, 30)      # All zeros
    
    states = [state1, state2, state3]
    
    # Generate actions for each state
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=5,
        temperature=1.0
    )
    
    # Calculate statistics for each state's actions
    state_stats = []
    for i, state_actions in enumerate(actions):
        actions_array = np.array(state_actions)
        stats = {
            'mean': np.mean(actions_array),
            'std': np.std(actions_array),
            'actions': actions_array
        }
        state_stats.append(stats)
        print(f"   - State {i + 1}: mean = {stats['mean']:.3f}, std = {stats['std']:.3f}")
    
    # States should produce sufficiently different action distributions
    means = [stats['mean'] for stats in state_stats]
    mean_diff = max(means) - min(means)
    
    assert mean_diff > 0.05, f"Different states should produce different action distributions: {mean_diff:.4f}"
    
    print("✅ Different states produce appropriately different actions")


if __name__ == "__main__":
    print("🚀 Running Action Generation Tests\n")
    
    try:
        test_multi_action_generation()
        test_action_diversity()
        test_temperature_scaling_effects()
        test_deterministic_behavior()
        test_action_value_distribution()
        test_different_input_formats()
        test_state_independence()
        
        print("\n🎉 All action generation tests passed!")
        print("✅ Action generation is ready for GRPO training")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 