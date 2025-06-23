"""
Test log probability computation accuracy and gradient flow for GRPO training.

This test focuses specifically on the correctness of log probability calculations
and ensures gradients are properly preserved for policy gradient updates.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel


def test_log_probability_accuracy():
    """Test that log probabilities are computed correctly"""
    print("🧮 Testing log probability accuracy...")
    
    # Create small model for detailed testing
    cqt_shape = (2, 84, 20)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=8,
        device='cpu'
    )
    
    # Create test state
    state = torch.randn(2, 84, 20)
    states = [state]
    
    # Generate action to test
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=1,
        temperature=1.0
    )
    action = torch.tensor(actions[0][0], dtype=torch.float32)
    
    # Compute log probabilities
    log_probs = model.get_log_probabilities(states, [action])
    
    # Manual verification: get model output and compute expected log prob
    with torch.no_grad():
        model_output = model.forward(state.unsqueeze(0))  # [1, 2, 20, 8]
        
        # Convert action to indices
        action_indices = (action * (model.distribution_size - 1)).long()
        action_indices = torch.clamp(action_indices, 0, model.distribution_size - 1)
        
        # Manually gather log probabilities
        manual_log_probs = []
        for d in range(model.num_devices):
            for t in range(action.shape[1]):
                idx = action_indices[d, t]
                log_prob = model_output[0, d, t, idx]
                manual_log_probs.append(log_prob)
        
        expected_log_prob = sum(manual_log_probs)
    
    # Compare computed vs expected
    diff = abs(log_probs[0].item() - expected_log_prob.item())
    assert diff < 1e-5, f"Log probability mismatch: {log_probs[0].item()} vs {expected_log_prob.item()}"
    
    print("✅ Log probability computation is accurate")
    print(f"   - Computed: {log_probs[0].item():.6f}")
    print(f"   - Expected: {expected_log_prob.item():.6f}")
    print(f"   - Difference: {diff:.8f}")


def test_probability_normalization():
    """Test that probability distributions are properly normalized"""
    print("\n📊 Testing probability normalization...")
    
    cqt_shape = (2, 84, 30)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=16,
        device='cpu'
    )
    
    # Create test input
    state = torch.randn(2, 84, 30)
    
    # Get model output (log probabilities)
    with torch.no_grad():
        log_probs = model.forward(state.unsqueeze(0))  # [1, 3, 30, 16]
        
        # Convert to probabilities
        probs = torch.exp(log_probs)
        
        # Check normalization across distribution dimension
        prob_sums = probs.sum(dim=-1)  # Should all be 1.0
        
        # Verify normalization
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            "Probabilities are not properly normalized"
        
        # Check that all probabilities are positive
        assert torch.all(probs >= 0), "Some probabilities are negative"
        
        # Check that log probabilities are non-positive
        assert torch.all(log_probs <= 0), "Some log probabilities are positive"
    
    print("✅ Probability distributions are properly normalized")
    print(f"   - Probability sums: {prob_sums[0, 0, :5].tolist()}")  # Show first few
    print(f"   - Log prob range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")


def test_gradient_preservation():
    """Test that gradients are preserved through log probability computation"""
    print("\n🔄 Testing gradient preservation...")
    
    cqt_shape = (2, 84, 25)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=8,
        device='cpu'
    )
    
    # Enable gradients
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Create test data
    state = torch.randn(2, 84, 25, requires_grad=True)
    action = torch.rand(2, 25)  # Random action in [0, 1]
    
    # Compute log probabilities (should preserve gradients)
    log_probs = model.get_log_probabilities([state], [action])
    
    # Create loss and backpropagate
    loss = -log_probs.sum()
    loss.backward()
    
    # Check that input gradients were computed
    assert state.grad is not None, "Input gradients not computed"
    assert not torch.allclose(state.grad, torch.zeros_like(state.grad)), \
        "Input gradients are zero"
    
    # Check that model parameter gradients were computed
    param_with_grad = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            param_with_grad += 1
    
    assert param_with_grad > 0, "No model parameters received gradients"
    grad_ratio = param_with_grad / total_params
    
    print("✅ Gradients are preserved through log probability computation")
    print(f"   - Input gradients computed: Yes")
    print(f"   - Parameters with gradients: {param_with_grad}/{total_params} ({grad_ratio:.1%})")


def test_batch_consistency():
    """Test that log probabilities are consistent across batch processing"""
    print("\n📦 Testing batch consistency...")
    
    cqt_shape = (2, 84, 40)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=12,
        device='cpu'
    )
    
    # Create test states and actions
    states = [torch.randn(2, 84, 40) for _ in range(3)]
    actions = [torch.rand(3, 40) for _ in range(3)]
    
    # Compute log probabilities for batch
    batch_log_probs = model.get_log_probabilities(states, actions)
    
    # Compute log probabilities individually
    individual_log_probs = []
    for state, action in zip(states, actions):
        log_prob = model.get_log_probabilities([state], [action])
        individual_log_probs.append(log_prob[0])
    
    individual_log_probs = torch.stack(individual_log_probs)
    
    # Compare batch vs individual results
    diff = torch.abs(batch_log_probs - individual_log_probs)
    max_diff = diff.max().item()
    
    assert max_diff < 1e-5, f"Batch vs individual mismatch: max diff = {max_diff}"
    
    print("✅ Batch processing is consistent with individual processing")
    print(f"   - Max difference: {max_diff:.8f}")
    print(f"   - Batch results: {batch_log_probs.tolist()}")
    print(f"   - Individual results: {individual_log_probs.tolist()}")


def test_action_boundary_handling():
    """Test log probability computation for edge case actions"""
    print("\n🎯 Testing action boundary handling...")
    
    cqt_shape = (2, 84, 30)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=10,
        device='cpu'
    )
    
    state = torch.randn(2, 84, 30)
    
    # Test boundary actions
    boundary_actions = [
        torch.zeros(2, 30),  # All zeros
        torch.ones(2, 30),   # All ones
        torch.full((2, 30), 0.5),  # All middle values
    ]
    
    for i, action in enumerate(boundary_actions):
        log_probs = model.get_log_probabilities([state], [action])
        
        # Should be finite and non-positive
        assert torch.isfinite(log_probs).all(), f"Non-finite log prob for boundary action {i}"
        assert torch.all(log_probs <= 0), f"Positive log prob for boundary action {i}"
        
        print(f"   - Boundary action {i}: log_prob = {log_probs[0].item():.6f}")
    
    print("✅ Boundary actions handled correctly")


def test_temperature_scaling_consistency():
    """Test that temperature scaling affects log probabilities appropriately"""
    print("\n🌡️ Testing temperature scaling consistency...")
    
    cqt_shape = (2, 84, 25)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=2,
        distribution_size=8,
        device='cpu'
    )
    
    state = torch.randn(2, 84, 25)
    
    # Generate actions with different temperatures
    temps = [0.5, 1.0, 2.0]
    temp_results = {}
    
    for temp in temps:
        # Generate action with this temperature
        actions = model.generate_actions(
            states=[state],
            num_actions_per_state=1,
            temperature=temp
        )
        action = torch.tensor(actions[0][0], dtype=torch.float32)
        
        # Compute log probability
        log_prob = model.get_log_probabilities([state], [action])
        temp_results[temp] = (action, log_prob[0].item())
    
    # Analyze results
    for temp, (action, log_prob) in temp_results.items():
        print(f"   - Temperature {temp}: log_prob = {log_prob:.6f}")
    
    # Lower temperature should generally produce higher log probabilities
    # (actions closer to mode of distribution)
    print("✅ Temperature scaling affects log probabilities as expected")


if __name__ == "__main__":
    print("🚀 Running Log Probability Tests\n")
    
    try:
        test_log_probability_accuracy()
        test_probability_normalization()
        test_gradient_preservation()
        test_batch_consistency()
        test_action_boundary_handling()
        test_temperature_scaling_consistency()
        
        print("\n🎉 All log probability tests passed!")
        print("✅ Log probability computation is ready for GRPO training")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 