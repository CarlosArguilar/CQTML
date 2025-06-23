import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import types
from typing import List, Any, Dict
from torch import Tensor


class CQTViTModel:
    """ViT model adapted for CQT data with shape [batch_size, channels, height, time]"""
    
    @staticmethod
    def create_model(cqt_shape, num_devices=4, patch_size=16, 
                    distribution_size=32, use_8bit=False, use_half_precision=False, 
                    gradient_checkpointing=False, device='auto'):
        """
        Create a ViT model adapted for CQT data with log probability distributions for GRPO
        
        Args:
            cqt_shape: Tuple (channels, height, time_dim) from CQT data
            num_devices: Number of output devices (default: 4)
            patch_size: ViT patch size (default: 16)
            distribution_size: Size of the probability distribution for each output value (default: 32)
            use_8bit: Enable 8-bit quantization for memory efficiency (default: False)
            use_half_precision: Use FP16 for faster inference (default: False)
            gradient_checkpointing: Enable gradient checkpointing to save memory (default: False)
            device: Device placement ('auto', 'cpu', 'cuda', 'mps') (default: 'auto')
            
        Returns:
            Modified ViT model ready for CQT inference with GRPO compatibility
        """
        channels, height, time_dim = cqt_shape
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Load base ViT model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        
        # Modify input projection for CQT channels
        model.patch_embed.proj = nn.Conv2d(
            channels, 
            model.patch_embed.proj.out_channels,
            kernel_size=model.patch_embed.proj.kernel_size,
            stride=model.patch_embed.proj.stride
        )
        
        # Update patch embedding dimensions
        model.patch_embed.img_size = (height, time_dim)
        model.patch_embed.grid_size = (height // patch_size, time_dim // patch_size)
        model.patch_embed.num_patches = model.patch_embed.grid_size[0] * model.patch_embed.grid_size[1]
        
        # Resize positional embeddings
        CQTViTModel._resize_pos_embeddings(model)
        
        # Replace classification head with log probability distribution head
        model.head = CQTViTModel.LogProbabilityHead(model.head.in_features, num_devices, distribution_size)
        
        # Add model parameters
        model.num_devices = num_devices
        model.distribution_size = distribution_size
        
        # Replace forward method
        model.forward = types.MethodType(CQTViTModel._forward_cqt, model)
        
        # Add PolicyModel protocol methods
        model.generate_actions = types.MethodType(CQTViTModel._generate_actions, model)
        model.get_log_probabilities = types.MethodType(CQTViTModel._get_log_probabilities, model)
        model.get_parameters = types.MethodType(CQTViTModel._get_parameters, model)
        
        # Apply resource efficiency optimizations
        model = CQTViTModel._apply_optimizations(
            model, use_8bit, use_half_precision, gradient_checkpointing, device
        )
        
        print(f"Model loaded on {device} with optimizations:")
        print(f"  - 8-bit quantization: {use_8bit}")
        print(f"  - Half precision: {use_half_precision}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        print(f"  - Distribution size: {distribution_size}")
        print(f"  - Model size: {CQTViTModel.get_model_memory_usage(model):.1f} MB")
        
        return model
    
    class LogProbabilityHead(nn.Module):
        """Head that outputs log probabilities for each output value"""
        
        def __init__(self, in_features, num_devices, distribution_size):
            super().__init__()
            self.num_devices = num_devices
            self.distribution_size = distribution_size
            
            # Linear layer to produce logits for all devices and distribution bins
            self.linear = nn.Linear(in_features, num_devices * distribution_size)
            
        def forward(self, x):
            # x shape: [batch_size, num_patches, in_features]
            batch_size, num_patches, _ = x.shape
            
            # Get logits
            logits = self.linear(x)  # [batch_size, num_patches, num_devices * distribution_size]
            
            # Reshape to separate devices and distribution dimensions
            logits = logits.view(batch_size, num_patches, self.num_devices, self.distribution_size)
            
            # Apply log_softmax to get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            return log_probs  # [batch_size, num_patches, num_devices, distribution_size]

    @staticmethod
    def _resize_pos_embeddings(model):
        """Resize positional embeddings to match new patch count"""
        old_pos_embed = model.pos_embed
        old_num_patches = old_pos_embed.shape[1] - 1  # Subtract class token
        new_num_patches = model.patch_embed.num_patches
        
        if old_num_patches != new_num_patches:
            # Extract class token and patch embeddings
            class_token = old_pos_embed[:, 0:1, :]
            patch_embeddings = old_pos_embed[:, 1:, :]
            
            # Reshape to 2D grid and interpolate
            old_grid_size = int(old_num_patches ** 0.5)
            patch_embeddings = patch_embeddings.reshape(1, old_grid_size, old_grid_size, -1)
            patch_embeddings = patch_embeddings.permute(0, 3, 1, 2)
            
            # Interpolate to new grid size
            new_patch_embeddings = nn.functional.interpolate(
                patch_embeddings,
                size=model.patch_embed.grid_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to sequence format
            new_patch_embeddings = new_patch_embeddings.permute(0, 2, 3, 1)
            new_patch_embeddings = new_patch_embeddings.reshape(1, new_num_patches, -1)
            
            # Update positional embeddings
            model.pos_embed = nn.Parameter(torch.cat([class_token, new_patch_embeddings], dim=1))
    
    @staticmethod
    def _forward_cqt(self, x):
        """Custom forward method for CQT data returning log probabilities"""
        T = x.shape[-1]  # Get time dimension
        
        # Convert to same dtype as model if using half precision
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
        
        # Standard ViT forward pass
        x = self.forward_features(x)  # [batch_size, num_patches, embed_dim]
        
        # Pass through log probability head
        log_probs = self.head(x)  # [batch_size, num_patches, num_devices, distribution_size]
        
        # Interpolate to match target time dimension
        batch_size, num_patches, num_devices, distribution_size = log_probs.shape
        
        # Reshape for interpolation: [batch_size * num_devices * distribution_size, 1, num_patches]
        log_probs_reshaped = log_probs.permute(0, 2, 3, 1).reshape(
            batch_size * num_devices * distribution_size, 1, num_patches
        )
        
        # Interpolate to target time dimension
        log_probs_interpolated = F.interpolate(
            log_probs_reshaped, size=T, mode='linear', align_corners=False
        )
        
        # Reshape back to target format: [batch_size, num_devices, T, distribution_size]
        log_probs_final = log_probs_interpolated.reshape(
            batch_size, num_devices, distribution_size, T
        ).permute(0, 1, 3, 2)
        
        # Re-normalize after interpolation to ensure proper probability distributions
        log_probs_final = F.log_softmax(log_probs_final, dim=-1)
        
        return log_probs_final
    
    @staticmethod
    def _generate_actions(self, states: List[Any], num_actions_per_state: int, **generation_kwargs: Any) -> List[List[Any]]:
        """
        Generate multiple actions for each state/input.
        
        Args:
            states: List of input CQT tensors
            num_actions_per_state: Number of actions to generate per state
            **generation_kwargs: Additional generation parameters (e.g., temperature)
            
        Returns:
            List of lists, where each inner list contains actions for one state
        """
        temperature = generation_kwargs.get('temperature', 1.0)
        all_actions = []
        
        # Process each state
        for state in states:
            state_actions = []
            
            # Convert state to tensor if needed
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            # Ensure state has batch dimension
            if state.dim() == 3:  # [channels, height, time]
                state = state.unsqueeze(0)  # [1, channels, height, time]
            
            # Move to same device as model
            state = state.to(next(self.parameters()).device)
            
            # Generate multiple actions for this state
            for _ in range(num_actions_per_state):
                with torch.no_grad():
                    # Get log probabilities
                    log_probs = self.forward(state)  # [1, num_devices, T, distribution_size]
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        log_probs = log_probs / temperature
                    
                    # Sample from the distribution with numerical stability
                    # Subtract max for numerical stability before exp
                    log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]
                    probs = torch.exp(log_probs_stable)
                    
                    # Ensure probabilities are positive (numerical stability)
                    probs = torch.clamp(probs, min=1e-8)
                    
                    # Sample indices from the categorical distribution
                    sampled_indices = torch.multinomial(
                        probs.reshape(-1, self.distribution_size), 
                        num_samples=1
                    ).reshape(1, self.num_devices, -1)
                    
                    # Convert indices to continuous values [0, 1]
                    action = sampled_indices.float() / (self.distribution_size - 1)
                    
                    # Remove batch dimension and convert to numpy
                    action = action.squeeze(0).cpu().numpy()
                    state_actions.append(action)
            
            all_actions.append(state_actions)
        
        return all_actions
    
    @staticmethod
    def _get_log_probabilities(self, states: List[Any], actions: List[Any]) -> Tensor:
        """
        Calculate log probabilities of actions given states.
        
        Args:
            states: List of input CQT tensors
            actions: List of corresponding actions (2D tensors with shape [num_devices, T])
            
        Returns:
            Tensor of log probabilities for each (state, action) pair
        """
        log_probs_list = []
        
        for state, action in zip(states, actions):
            # Convert to tensors if needed
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            
            # Ensure state has batch dimension
            if state.dim() == 3:
                state = state.unsqueeze(0)
            
            # Move to same device as model
            state = state.to(next(self.parameters()).device)
            action = action.to(next(self.parameters()).device)
            
            # Get log probabilities from model (WITH gradients for training)
            model_log_probs = self.forward(state)  # [1, num_devices, T, distribution_size]
            
            # Convert continuous action values [0, 1] to discrete indices
            action_indices = (action * (self.distribution_size - 1)).long()
            action_indices = torch.clamp(action_indices, 0, self.distribution_size - 1)
            
            # Gather log probabilities for the specific actions
            batch_size, num_devices, T, _ = model_log_probs.shape
            
            # Expand action_indices to match model_log_probs dimensions
            action_indices = action_indices.unsqueeze(0)  # [1, num_devices, T]
            action_indices = action_indices.unsqueeze(-1)  # [1, num_devices, T, 1]
            
            # Gather the log probabilities
            action_log_probs = torch.gather(model_log_probs, dim=-1, index=action_indices)
            action_log_probs = action_log_probs.squeeze(-1)  # [1, num_devices, T]
            
            # Sum over devices and time to get single log probability for this (state, action) pair
            total_log_prob = action_log_probs.sum()
            log_probs_list.append(total_log_prob)
        
        return torch.stack(log_probs_list)
    
    @staticmethod
    def _get_parameters(self) -> Dict[str, Tensor]:
        """Get model parameters for optimization."""
        return {name: param for name, param in self.named_parameters()}
    
    @staticmethod
    def _apply_optimizations(model, use_8bit, use_half_precision, gradient_checkpointing, device):
        """Apply resource efficiency optimizations to the model"""
        
        # Move to device first
        model = model.to(device)
        
        # Apply 8-bit quantization
        if use_8bit:
            try:
                import bitsandbytes as bnb
                # Convert linear layers to 8-bit
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Replace with 8-bit linear layer
                        new_module = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False
                        )
                        # Copy weights
                        new_module.weight.data = module.weight.data
                        if module.bias is not None:
                            new_module.bias.data = module.bias.data
                        
                        # Replace the module
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        if parent_name:
                            parent = model.get_submodule(parent_name)
                            setattr(parent, child_name, new_module)
                        else:
                            setattr(model, child_name, new_module)
                            
                print("  ✓ Applied 8-bit quantization")
            except ImportError:
                print("  ✗ bitsandbytes not available, skipping 8-bit quantization")
                print("    Install with: pip install bitsandbytes")
        
        # Apply half precision
        if use_half_precision:
            if device != 'cpu':  # Half precision not recommended on CPU
                model = model.half()
                print("  ✓ Applied half precision (FP16)")
            else:
                print("  ✗ Half precision skipped (not recommended on CPU)")
        
        # Apply gradient checkpointing
        if gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("  ✓ Applied gradient checkpointing")
            else:
                # Manual gradient checkpointing for transformer blocks
                try:
                    for block in model.blocks:
                        block = torch.utils.checkpoint.checkpoint_wrapper(block)
                    print("  ✓ Applied manual gradient checkpointing")
                except:
                    print("  ✗ Gradient checkpointing not available for this model")
        
        return model
    
    @staticmethod
    def get_model_memory_usage(model):
        """Get estimated model memory usage in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = (param_size + buffer_size) / 1024 / 1024  # Convert to MB
        return model_size 