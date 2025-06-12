import torch
import torch.nn as nn
import torch.nn.functional as F


class CQTRewardModel(nn.Module):
    """
    Reward model for evaluating CQTViTModel input/output pairs.
    
    Input: 
        - cqt_input: [batch_size, 2, 84, T] (original CQT data)
        - model_output: [batch_size, 4, T] (CQTViTModel output)
    Output:
        - reward: [batch_size] (scalar reward for each sample)
    """
    
    def __init__(self, input_channels=2, output_channels=4, freq_bins=84, 
                 embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Input encoder (for CQT data) - captures spectral patterns
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Pool freq dimension, keep time
        )
        
        # Output encoder (for processed data) - captures temporal patterns
        self.output_encoder = nn.Sequential(
            nn.Conv1d(output_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Projection layers to match embed_dim
        self.input_projection = nn.Linear(256, embed_dim)
        self.output_projection = nn.Linear(256, embed_dim)
        
        # Cross-attention: how well does output relate to input?
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion and scoring
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, cqt_input, model_output):
        """
        Args:
            cqt_input: [batch_size, 2, 84, T] - Original CQT data
            model_output: [batch_size, 4, T] - CQTViTModel output
            
        Returns:
            reward: [batch_size] - Scalar reward for each sample
        """
        # Convert to same dtype as model if using half precision
        if next(self.parameters()).dtype == torch.float16:
            cqt_input = cqt_input.half()
            model_output = model_output.half()
        
        # Encode input CQT
        input_features = self.input_encoder(cqt_input)  # [B, 256, 1, T_reduced]
        input_features = input_features.squeeze(2).transpose(1, 2)  # [B, T_reduced, 256]
        input_features = self.input_projection(input_features)  # [B, T_reduced, embed_dim]
        
        # Encode output
        output_features = self.output_encoder(model_output).transpose(1, 2)  # [B, T_reduced, 256]
        output_features = self.output_projection(output_features)  # [B, T_reduced, embed_dim]
        
        # Align temporal dimensions
        min_time = min(input_features.size(1), output_features.size(1))
        input_features = input_features[:, :min_time, :]
        output_features = output_features[:, :min_time, :]
        
        # Cross-attention: evaluate output quality based on input
        cross_attended, _ = self.cross_attention(
            query=output_features,
            key=input_features, 
            value=input_features
        )
        
        # Fusion of original output features and cross-attended features
        fused = torch.cat([output_features, cross_attended], dim=-1)
        fused = self.fusion(fused)  # [B, T, embed_dim]
        
        # Temporal pooling and scoring
        pooled = self.temporal_pool(fused.transpose(1, 2)).squeeze(-1)  # [B, embed_dim]
        reward = self.scorer(pooled).squeeze(-1)  # [B]
        
        return reward
    
    @staticmethod
    def create_model(cqt_shape, output_shape, embed_dim=128, num_heads=4, dropout=0.1,
                    use_8bit=False, use_half_precision=False, 
                    gradient_checkpointing=False, device='auto'):
        """
        Factory method to create CQTRewardModel with explicit configuration
        
        Args:
            cqt_shape: Tuple (channels, freq_bins, time) for CQT input
            output_shape: Tuple (channels, time) for model output
            embed_dim: Embedding dimension (default: 128)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.1)
            use_8bit: Enable 8-bit quantization for memory efficiency (default: False)
            use_half_precision: Use FP16 for faster inference (default: False)
            gradient_checkpointing: Enable gradient checkpointing to save memory (default: False)
            device: Device placement ('auto', 'cpu', 'cuda', 'mps') (default: 'auto')
        """
        input_channels, freq_bins, _ = cqt_shape
        output_channels, _ = output_shape
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Create model
        model = CQTRewardModel(
            input_channels=input_channels,
            output_channels=output_channels,
            freq_bins=freq_bins,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Apply resource efficiency optimizations
        model = CQTRewardModel._apply_optimizations(
            model, use_8bit, use_half_precision, gradient_checkpointing, device
        )
        
        print(f"Reward model loaded on {device} with:")
        print(f"  - Embed dim: {embed_dim}, Heads: {num_heads}")
        print(f"  - 8-bit quantization: {use_8bit}")
        print(f"  - Half precision: {use_half_precision}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        print(f"  - Model size: {model.get_model_size():.1f} MB")
        
        return model
    
    @staticmethod
    def _apply_optimizations(model, use_8bit, use_half_precision, gradient_checkpointing, device):
        """Apply resource efficiency optimizations to the reward model"""
        
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
            # Apply to attention layers and CNN blocks
            try:
                # Wrap attention layer
                model.cross_attention = torch.utils.checkpoint.checkpoint_wrapper(model.cross_attention)
                
                # Wrap encoder blocks if possible
                if hasattr(model.input_encoder, '__iter__'):
                    for i, layer in enumerate(model.input_encoder):
                        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                            model.input_encoder[i] = torch.utils.checkpoint.checkpoint_wrapper(layer)
                
                if hasattr(model.output_encoder, '__iter__'):
                    for i, layer in enumerate(model.output_encoder):
                        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                            model.output_encoder[i] = torch.utils.checkpoint.checkpoint_wrapper(layer)
                            
                print("  ✓ Applied gradient checkpointing")
            except:
                print("  ✗ Gradient checkpointing not fully available")
        
        return model
    
    def get_model_size(self):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024
