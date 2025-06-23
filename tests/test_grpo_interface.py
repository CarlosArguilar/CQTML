"""
Test GRPO PolicyModel interface implementation for CQT ViT model.

This test verifies that the model correctly implements all required methods
for GRPO training and that the interface behaves as expected.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel


def test_policy_model_interface():
    """Test that the model implements all PolicyModel protocol methods"""
    print("🧪 Testing PolicyModel interface implementation...")
    
    # Create test model
    cqt_shape = (2, 84, 100)  # Small size for testing
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,  # Small for testing
        device='cpu'
    )
    
    # Test required methods exist
    required_methods = ['generate_actions', 'get_log_probabilities', 'get_parameters']
    for method_name in required_methods:
        assert hasattr(model, method_name), f"Model missing required method: {method_name}"
        assert callable(getattr(model, method_name)), f"Method {method_name} is not callable"
    
    print("✅ All required methods present and callable")


def test_generate_actions():
    """Test action generation functionality"""
    print("\n🎲 Testing action generation...")
    
    # Create test model and data
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    # Create test states
    states = [
        torch.randn(2, 84, 100),
        torch.randn(2, 84, 100),
        torch.randn(2, 84, 100)
    ]
    
    # Test action generation
    num_actions_per_state = 3
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=num_actions_per_state,
        temperature=1.0
    )
    
    # Verify output structure
    assert len(actions) == len(states), f"Expected {len(states)} state results, got {len(actions)}"
    
    for i, state_actions in enumerate(actions):
        assert len(state_actions) == num_actions_per_state, \
            f"State {i}: expected {num_actions_per_state} actions, got {len(state_actions)}"
        
        for j, action in enumerate(state_actions):
            assert isinstance(action, np.ndarray), f"Action {i},{j} should be numpy array"
            assert action.shape == (4, 100), f"Action {i},{j} shape should be (4, 100), got {action.shape}"
            assert action.min() >= 0.0 and action.max() <= 1.0, \
                f"Action {i},{j} values should be in [0,1], got range [{action.min():.3f}, {action.max():.3f}]"
    
    print("✅ Action generation produces correct output structure")
    print(f"   - Generated {len(actions)} state results")
    print(f"   - Each state has {num_actions_per_state} actions")
    print(f"   - Actions have shape (4, 100) with values in [0, 1]")


def test_temperature_effects():
    """Test that temperature affects action diversity"""
    print("\n🌡️ Testing temperature effects on action diversity...")
    
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=32,
        device='cpu'
    )
    
    # Use same state for comparison
    state = torch.randn(2, 84, 100)
    states = [state]
    
    # Generate actions with different temperatures
    temps = [0.1, 1.0, 2.0]
    temp_actions = {}
    
    for temp in temps:
        actions = model.generate_actions(
            states=states,
            num_actions_per_state=5,
            temperature=temp
        )
        temp_actions[temp] = actions[0]  # Get actions for first (only) state
    
    # Calculate diversity for each temperature
    for temp in temps:
        actions = np.array(temp_actions[temp])  # [num_actions, num_devices, T]
        
        # Calculate variance across actions
        action_variance = np.var(actions, axis=0).mean()
        print(f"   - Temperature {temp}: variance = {action_variance:.4f}")
    
    # Low temperature should have less variance than high temperature
    low_temp_var = np.var(np.array(temp_actions[0.1]), axis=0).mean()
    high_temp_var = np.var(np.array(temp_actions[2.0]), axis=0).mean()
    
    assert high_temp_var > low_temp_var, \
        f"High temperature should have more variance: {high_temp_var:.4f} > {low_temp_var:.4f}"
    
    print("✅ Temperature correctly affects action diversity")


def test_log_probabilities():
    """Test log probability computation"""
    print("\n📊 Testing log probability computation...")
    
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    # Create test data
    states = [torch.randn(2, 84, 100), torch.randn(2, 84, 100)]
    
    # Generate some actions
    actions_list = model.generate_actions(
        states=states,
        num_actions_per_state=1,
        temperature=1.0
    )
    
    # Get first action for each state
    actions = [torch.tensor(actions_list[i][0], dtype=torch.float32) for i in range(len(states))]
    
    # Compute log probabilities
    log_probs = model.get_log_probabilities(states, actions)
    
    # Verify output
    assert isinstance(log_probs, torch.Tensor), "Log probabilities should be a tensor"
    assert log_probs.shape == (len(states),), f"Expected shape ({len(states)},), got {log_probs.shape}"
    assert torch.all(log_probs <= 0), "Log probabilities should be non-positive"
    
    print("✅ Log probability computation works correctly")
    print(f"   - Output shape: {log_probs.shape}")
    print(f"   - Value range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")


def test_gradient_flow():
    """Test that gradients flow through log probability computation"""
    print("\n🔄 Testing gradient flow...")
    
    cqt_shape = (2, 84, 50)  # Smaller for faster computation
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=8,
        device='cpu'
    )
    
    # Create test data
    states = [torch.randn(2, 84, 50)]
    actions = [torch.rand(4, 50)]  # Random action in [0, 1]
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Compute log probabilities
    log_probs = model.get_log_probabilities(states, actions)
    
    # Compute a simple loss
    loss = -log_probs.sum()
    
    # Backpropagate
    loss.backward()
    
    # Check that gradients were computed
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    assert grad_count > 0, "No gradients computed - gradient flow is broken"
    
    print("✅ Gradients flow correctly through log probability computation")
    print(f"   - {grad_count} parameters received gradients")


def test_get_parameters():
    """Test parameter access method"""
    print("\n⚙️ Testing parameter access...")
    
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    # Get parameters
    params = model.get_parameters()
    
    # Verify output
    assert isinstance(params, dict), "get_parameters should return a dictionary"
    assert len(params) > 0, "Parameter dictionary should not be empty"
    
    # Check that all values are tensors
    for name, param in params.items():
        assert isinstance(param, torch.Tensor), f"Parameter {name} should be a tensor"
        assert isinstance(name, str), f"Parameter name should be string, got {type(name)}"
    
    print("✅ Parameter access works correctly")
    print(f"   - Retrieved {len(params)} parameters")


def test_action_consistency():
    """Test that same state produces different actions but consistent probabilities"""
    print("\n🔄 Testing action consistency...")
    
    cqt_shape = (2, 84, 50)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    # Same state, generate multiple times
    state = torch.randn(2, 84, 50)
    states = [state]
    
    # Generate actions multiple times
    all_actions = []
    for _ in range(3):
        actions = model.generate_actions(
            states=states,
            num_actions_per_state=2,
            temperature=1.5
        )
        all_actions.extend(actions[0])
    
    # Check that actions are different (stochastic)
    actions_array = np.array(all_actions)
    action_differences = []
    for i in range(len(all_actions)):
        for j in range(i + 1, len(all_actions)):
            diff = np.mean(np.abs(all_actions[i] - all_actions[j]))
            action_differences.append(diff)
    
    mean_diff = np.mean(action_differences)
    assert mean_diff > 0.01, f"Actions should be different, mean difference: {mean_diff:.4f}"
    
    print("✅ Action generation is appropriately stochastic")
    print(f"   - Mean difference between actions: {mean_diff:.4f}")


if __name__ == "__main__":
    print("🚀 Running GRPO PolicyModel Interface Tests\n")
    
    try:
        test_policy_model_interface()
        test_generate_actions()
        test_temperature_effects()
        test_log_probabilities()
        test_gradient_flow()
        test_get_parameters()
        test_action_consistency()
        
        print("\n🎉 All GRPO interface tests passed!")
        print("✅ Model is ready for GRPO training")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 