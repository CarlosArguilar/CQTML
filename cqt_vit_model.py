import torch
import torch.nn as nn
import timm
import types


class CQTViTModel:
    """ViT model adapted for CQT data with shape [batch_size, channels, height, time]"""
    
    @staticmethod
    def create_model(cqt_shape, num_devices=4, patch_size=16, 
                    use_8bit=False, use_half_precision=False, 
                    gradient_checkpointing=False, device='auto'):
        """
        Create a ViT model adapted for CQT data with resource efficiency options
        
        Args:
            cqt_shape: Tuple (channels, height, time_dim) from CQT data
            num_devices: Number of output devices (default: 4)
            patch_size: ViT patch size (default: 16)
            use_8bit: Enable 8-bit quantization for memory efficiency (default: False)
            use_half_precision: Use FP16 for faster inference (default: False)
            gradient_checkpointing: Enable gradient checkpointing to save memory (default: False)
            device: Device placement ('auto', 'cpu', 'cuda', 'mps') (default: 'auto')
            
        Returns:
            Modified ViT model ready for CQT inference
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
        
        # Replace classification head
        model.head = nn.Linear(model.head.in_features, num_devices)
        
        # Replace forward method
        model.forward = types.MethodType(CQTViTModel._forward_cqt, model)
        
        # Apply resource efficiency optimizations
        model = CQTViTModel._apply_optimizations(
            model, use_8bit, use_half_precision, gradient_checkpointing, device
        )
        
        print(f"Model loaded on {device} with optimizations:")
        print(f"  - 8-bit quantization: {use_8bit}")
        print(f"  - Half precision: {use_half_precision}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        print(f"  - Model size: {CQTViTModel.get_model_memory_usage(model):.1f} MB")
        
        return model
    
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
        """Custom forward method for CQT data"""
        T = x.shape[-1]  # Get time dimension
        
        # Convert to same dtype as model if using half precision
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
        
        # Standard ViT forward pass
        x = self.forward_features(x)  # [batch_size, num_patches, embed_dim]
        x = self.head(x)  # [batch_size, num_patches, num_devices]
        
        # Reshape to target format [batch_size, num_devices, T]
        x = x.transpose(1, 2)  # [batch_size, num_devices, num_patches]
        x = nn.functional.interpolate(x, size=T, mode='linear', align_corners=False)
        
        return x 
    
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