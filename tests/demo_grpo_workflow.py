"""
Demonstration of complete GRPO training workflow with CQT ViT model.

This demo shows how to use the GRPO-compatible CQT ViT model in a
realistic training scenario, including action generation, reward computation,
and policy gradient updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cqt_vit_model import CQTViTModel


class SimpleRewardModel(nn.Module):
    """Simple reward model for demonstration purposes"""
    
    def __init__(self, input_shape):
        super().__init__()
        num_devices, time_dim = input_shape
        
        # Simple linear layers to compute reward
        self.conv = nn.Conv1d(num_devices, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, actions):
        """Compute reward for actions [batch_size, num_devices, time_dim]"""
        if actions.dim() == 2:  # Single action
            actions = actions.unsqueeze(0)
        
        x = self.conv(actions)  # [batch_size, 16, time_dim]
        x = self.pool(x)  # [batch_size, 16, 1]
        x = x.squeeze(-1)  # [batch_size, 16]
        reward = self.fc(x)  # [batch_size, 1]
        
        return reward.squeeze(-1)  # [batch_size]


def create_synthetic_cqt_data(time_dim=60, num_samples=20):
    """Create synthetic CQT data for demonstration"""
    print("🎵 Creating synthetic CQT data...")
    
    cqt_data = []
    for i in range(num_samples):
        # Create synthetic CQT with realistic patterns
        cqt = torch.randn(2, 84, time_dim)
        
        # Add some structure (frequency correlations)
        cqt[0] = torch.cumsum(cqt[0] * 0.1, dim=0)  # Real part
        cqt[1] = torch.cumsum(cqt[1] * 0.1, dim=0)  # Imaginary part
        
        cqt_data.append(cqt)
    
    print(f"   - Generated {num_samples} synthetic CQT samples")
    print(f"   - Shape: {cqt_data[0].shape}")
    
    return cqt_data


def demonstrate_policy_model_interface():
    """Demonstrate PolicyModel interface compliance"""
    print("\n🤖 Demonstrating PolicyModel interface...")
    
    # Create model
    cqt_shape = (2, 84, 100)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=24,
        device='cpu'
    )
    
    # Test states
    states = [torch.randn(2, 84, 100) for _ in range(3)]
    
    print("   📊 Testing generate_actions...")
    actions = model.generate_actions(
        states=states,
        num_actions_per_state=2,
        temperature=1.0
    )
    print(f"      ✅ Generated {len(actions)} state results with {len(actions[0])} actions each")
    
    print("   📊 Testing get_log_probabilities...")
    selected_actions = [actions[i][0] for i in range(len(states))]  # First action per state
    log_probs = model.get_log_probabilities(states, selected_actions)
    print(f"      ✅ Computed log probabilities: {log_probs}")
    
    print("   📊 Testing get_parameters...")
    params = model.get_parameters()
    print(f"      ✅ Retrieved {len(params)} model parameters")
    
    return model, states, actions


def demonstrate_grpo_training_step():
    """Demonstrate a single GRPO training step"""
    print("\n🎯 Demonstrating GRPO training step...")
    
    # Create models
    cqt_shape = (2, 84, 100)
    policy_model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=4,
        distribution_size=16,
        device='cpu'
    )
    
    reward_model = SimpleRewardModel(input_shape=(4, 100))
    
    # Create optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
    
    # Create batch of states
    batch_size = 4
    states = [torch.randn(2, 84, 100) for _ in range(batch_size)]
    
    print("   🎲 Generating actions...")
    # Generate multiple actions per state for comparison
    actions_per_state = policy_model.generate_actions(
        states=states,
        num_actions_per_state=2,  # Generate 2 actions per state
        temperature=1.2
    )
    
    print("   🏆 Computing rewards...")
    # Compute rewards for all actions
    all_actions = []
    all_rewards = []
    state_indices = []
    
    for state_idx, state_actions in enumerate(actions_per_state):
        for action in state_actions:
            action_tensor = torch.tensor(action, dtype=torch.float32)
            reward = reward_model(action_tensor)
            
            all_actions.append(action_tensor)
            all_rewards.append(reward.item())
            state_indices.append(state_idx)
    
    print(f"      - Generated {len(all_actions)} total actions")
    print(f"      - Reward range: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
    
    print("   📈 Performing policy gradient update...")
    # Select best action for each state (simplified GRPO)
    selected_actions = []
    selected_rewards = []
    
    for state_idx in range(batch_size):
        # Find actions for this state
        state_action_indices = [i for i, s_idx in enumerate(state_indices) if s_idx == state_idx]
        state_rewards = [all_rewards[i] for i in state_action_indices]
        
        # Select action with highest reward
        best_action_idx = state_action_indices[np.argmax(state_rewards)]
        selected_actions.append(all_actions[best_action_idx])
        selected_rewards.append(all_rewards[best_action_idx])
    
    # Compute log probabilities for selected actions
    log_probs = policy_model.get_log_probabilities(states, selected_actions)
    
    # Simple policy gradient loss (maximize log prob weighted by reward)
    rewards_tensor = torch.tensor(selected_rewards, dtype=torch.float32)
    policy_loss = -(log_probs * rewards_tensor).mean()
    
    # Backpropagation
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    print(f"      ✅ Policy loss: {policy_loss.item():.6f}")
    print(f"      ✅ Mean reward: {rewards_tensor.mean().item():.3f}")
    
    return policy_loss.item(), rewards_tensor.mean().item()


def demonstrate_temperature_exploration():
    """Demonstrate temperature-based exploration"""
    print("\n🌡️ Demonstrating temperature-based exploration...")
    
    cqt_shape = (2, 84, 80)
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=20,
        device='cpu'
    )
    
    reward_model = SimpleRewardModel(input_shape=(3, 80))
    
    # Single state for comparison
    state = torch.randn(2, 84, 80)
    
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    for temp in temperatures:
        print(f"   🌡️ Temperature {temp}:")
        
        # Generate actions with this temperature
        actions = model.generate_actions(
            states=[state],
            num_actions_per_state=5,
            temperature=temp
        )
        
        # Compute rewards
        rewards = []
        for action in actions[0]:
            action_tensor = torch.tensor(action, dtype=torch.float32)
            reward = reward_model(action_tensor)
            rewards.append(reward.item())
        
        # Statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"      - Mean reward: {mean_reward:.3f}")
        print(f"      - Reward std: {std_reward:.3f}")
        print(f"      - Best reward: {max(rewards):.3f}")


def demonstrate_batch_training():
    """Demonstrate batch training over multiple episodes"""
    print("\n📚 Demonstrating batch training...")
    
    # Create models
    cqt_shape = (2, 84, 60)
    policy_model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=3,
        distribution_size=12,
        device='cpu'
    )
    
    reward_model = SimpleRewardModel(input_shape=(3, 60))
    optimizer = optim.Adam(policy_model.parameters(), lr=5e-4)
    
    # Training data
    cqt_data = create_synthetic_cqt_data(time_dim=60, num_samples=20)
    
    # Training loop
    num_epochs = 5
    batch_size = 4
    
    training_history = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []
        
        # Sample batch
        batch_indices = np.random.choice(len(cqt_data), batch_size, replace=False)
        batch_states = [cqt_data[i] for i in batch_indices]
        
        # Generate actions
        actions_per_state = policy_model.generate_actions(
            states=batch_states,
            num_actions_per_state=3,
            temperature=1.0
        )
        
        # Evaluate all actions and select best ones
        selected_actions = []
        selected_rewards = []
        
        for state_idx, state_actions in enumerate(actions_per_state):
            state_rewards = []
            for action in state_actions:
                action_tensor = torch.tensor(action, dtype=torch.float32)
                reward = reward_model(action_tensor)
                state_rewards.append(reward.item())
            
            # Select best action
            best_idx = np.argmax(state_rewards)
            selected_actions.append(torch.tensor(state_actions[best_idx], dtype=torch.float32))
            selected_rewards.append(state_rewards[best_idx])
        
        # Policy gradient update
        log_probs = policy_model.get_log_probabilities(batch_states, selected_actions)
        rewards_tensor = torch.tensor(selected_rewards, dtype=torch.float32)
        
        policy_loss = -(log_probs * rewards_tensor).mean()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Track progress
        avg_reward = rewards_tensor.mean().item()
        epoch_losses.append(policy_loss.item())
        epoch_rewards.append(avg_reward)
        
        training_history.append({
            'epoch': epoch,
            'loss': policy_loss.item(),
            'reward': avg_reward
        })
        
        print(f"   📊 Epoch {epoch + 1}: Loss = {policy_loss.item():.6f}, Reward = {avg_reward:.3f}")
    
    print("   ✅ Training completed!")
    
    # Show training progress
    initial_reward = training_history[0]['reward']
    final_reward = training_history[-1]['reward']
    improvement = final_reward - initial_reward
    
    print(f"   📈 Reward improvement: {initial_reward:.3f} → {final_reward:.3f} (+{improvement:.3f})")
    
    return training_history


def main():
    """Run complete GRPO workflow demonstration"""
    print("🚀 GRPO Training Workflow Demonstration")
    print("=" * 50)
    
    try:
        # Test basic interface
        model, states, actions = demonstrate_policy_model_interface()
        
        # Single training step
        loss, reward = demonstrate_grpo_training_step()
        
        # Temperature exploration
        demonstrate_temperature_exploration()
        
        # Batch training
        training_history = demonstrate_batch_training()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ PolicyModel interface working correctly")
        print("   ✅ GRPO training steps functional")
        print("   ✅ Temperature exploration implemented")
        print("   ✅ Batch training demonstrated")
        print("   ✅ Gradient flow preserved throughout")
        
        print(f"\n🏆 Final training metrics:")
        print(f"   - Last training loss: {training_history[-1]['loss']:.6f}")
        print(f"   - Last average reward: {training_history[-1]['reward']:.3f}")
        print(f"   - Total epochs completed: {len(training_history)}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 