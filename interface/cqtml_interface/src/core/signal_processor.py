import torch
import numpy as np
import warnings


class SignalProcessor:
    """
    Processes signal tensors for visualization and comparison.
    Handles signal data manipulation, normalization, and time synchronization.
    """
    
    def __init__(self, signal_tensor, duration=None, max_value=1.0):
        """
        Initialize the signal processor.
        
        Args:
            signal_tensor (torch.Tensor): Signal tensor with shape [devices, timesteps] or [devices, timesteps, channels]
            duration (float, optional): Duration of the signal in seconds
            max_value (float, optional): Maximum signal value for normalization
        """
        if not isinstance(signal_tensor, torch.Tensor):
            raise TypeError("Signal tensor must be a PyTorch tensor")
            
        self.signal_tensor = signal_tensor
        self.max_value = max_value
        self.duration = duration
        
        # Check tensor dimensions
        if len(signal_tensor.shape) not in [2, 3]:
            raise ValueError("Signal tensor must have 2 or 3 dimensions")
            
        self.n_devices = signal_tensor.shape[0]
        self.n_timesteps = signal_tensor.shape[1]
        
        # Check if we have color channels
        self.has_color = len(signal_tensor.shape) == 3
        if self.has_color:
            self.n_channels = signal_tensor.shape[2]
    
    def get_value_at_time(self, device_idx, time_sec):
        """
        Get the signal value for a specific device at a specific time.
        
        Args:
            device_idx (int): The device index
            time_sec (float): The time in seconds
            
        Returns:
            tuple: RGB and alpha values (RGB only if has_color=True)
        """
        if self.duration is None:
            raise ValueError("Duration must be set to get value at specific time")
            
        # Convert time to index with interpolation
        time_ratio = time_sec / self.duration
        timestep_float = time_ratio * (self.n_timesteps - 1)
        
        # Use linear interpolation between timesteps
        timestep_low = int(np.floor(timestep_float))
        timestep_high = int(np.ceil(timestep_float))
        
        # Handle edge case at the end of the timeline
        if timestep_high >= self.n_timesteps:
            timestep_high = self.n_timesteps - 1
            
        # Calculate interpolation weight
        if timestep_low == timestep_high:
            weight = 0
        else:
            weight = timestep_float - timestep_low
            
        if self.has_color:
            # RGB values with alpha
            low_value = self.signal_tensor[device_idx, timestep_low].cpu().numpy()
            high_value = self.signal_tensor[device_idx, timestep_high].cpu().numpy()
            
            # Interpolate
            value = (1 - weight) * low_value + weight * high_value
            
            # Split into RGB and alpha
            if value.shape[0] >= 3:
                rgb = value[:3] / self.max_value
            else:
                rgb = np.array([0, 0, 1])  # Default blue
                
            if value.shape[0] >= 4:
                alpha = float(value[3] / self.max_value)
            else:
                alpha = float(value[0] / self.max_value)
                
            return tuple(rgb), alpha
        else:
            # Monochrome value (only alpha)
            low_value = self.signal_tensor[device_idx, timestep_low].item()
            high_value = self.signal_tensor[device_idx, timestep_high].item()
            
            # Interpolate and normalize
            value = (1 - weight) * low_value + weight * high_value
            alpha = float(max(0, min(1, value / self.max_value)))
            
            return (0, 0, 1), alpha  # Default blue with varying alpha
            
    def synchronize_with_audio(self, audio_tensor, audio_sample_rate):
        """
        Synchronize signal data with audio data.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor
            audio_sample_rate (int): Audio sample rate in Hz
            
        Returns:
            bool: True if synchronization was successful, False otherwise
        """
        audio_duration = len(audio_tensor) / audio_sample_rate
        
        if self.duration is None:
            # If signal duration was not set, use audio duration
            self.duration = audio_duration
            return True
            
        if abs(self.duration - audio_duration) > 0.001:  # Small tolerance
            # If durations don't match, warn and adjust
            warnings.warn(
                f"Signal duration ({self.duration}s) doesn't match audio duration "
                f"({audio_duration}s). Signal will be interpolated to match audio."
            )
            self.duration = audio_duration
            return True
            
        return True 