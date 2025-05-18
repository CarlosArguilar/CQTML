import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class SignalObj:
    def __init__(self, signals, duration=10, max_value=1):
        """
        Initialize the SignalVisualizer class.
        
        Args:
            signals: PyTorch tensor with signals (devices × timesteps) or (devices × timesteps × channels)
            duration: Duration in seconds that the signal represents (default: 10s)
            max_value: Maximum value for signal normalization (default: 1)
        """
        self.signals = signals
        self.duration = duration
        self.max_value = max_value
        
        # Check tensor dimensions
        if not isinstance(signals, torch.Tensor):
            raise TypeError("Signals must be a PyTorch tensor")
        
        if len(signals.shape) not in [2, 3]:
            raise ValueError("Signal tensor must have 2 or 3 dimensions")
            
        self.n_devices = signals.shape[0]
        self.n_timesteps = signals.shape[1]
        
        # Check if we have color channels
        self.has_color = len(signals.shape) == 3
        if self.has_color:
            self.n_channels = signals.shape[2]
        
        # Time axis
        self.time = torch.linspace(0, duration, self.n_timesteps)
    
    def visualization(self, figsize=(12, 8)):
        """
        Visualize signals for all devices over time using discrete blocks for each timestep.
        
        Each row represents a device, with color intensity based on signal strength.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate width of each time block
        time_block_width = self.duration / self.n_timesteps
        
        # For each device
        for i in range(self.n_devices):
            device_pos = self.n_devices - i  # Reverse order (first device at top)
            device_height = 0.8  # Height of each device row (less than 1 to create gaps)
            
            # Process signals for each timestep
            for t in range(self.n_timesteps):
                # Calculate position of this time block
                time_start = t * time_block_width
                
                if self.has_color:
                    # With RGBA channels
                    signal_data = self.signals[i, t].cpu().numpy()
                    
                    # Normalize RGB values to [0, 1]
                    if signal_data.shape[0] >= 3:
                        rgb = signal_data[:3] / self.max_value
                    else:
                        rgb = np.array([0, 0, 1])  # Default blue
                    
                    # Get alpha value
                    if signal_data.shape[0] >= 4:
                        alpha = float(signal_data[3] / self.max_value)
                    else:
                        alpha = float(signal_data[0] / self.max_value)
                    
                    # Create color with alpha
                    color = (rgb[0], rgb[1], rgb[2], alpha)
                else:
                    # Monochrome with varying alpha
                    signal_value = float(self.signals[i, t].item())
                    alpha = max(0, min(1, signal_value / self.max_value))
                    color = (0, 0, 1, alpha)  # Blue with varying alpha
                
                # Create rectangle for this timestep (only if alpha > 0)
                if alpha > 0:
                    rect = patches.Rectangle(
                        (time_start, device_pos - device_height/2),  # (x,y) bottom left corner
                        time_block_width,  # width
                        device_height,  # height
                        facecolor=color,
                        edgecolor=None
                    )
                    ax.add_patch(rect)
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Device')
        ax.set_title('Device Signal Visualization')
        
        # Set y-ticks for devices
        ax.set_yticks(range(1, self.n_devices + 1))
        ax.set_yticklabels([f'Device {self.n_devices - i}' for i in range(self.n_devices)])
        
        # Set axis limits
        ax.set_xlim(0, self.duration)
        ax.set_ylim(0.5, self.n_devices + 0.5)
        
        plt.tight_layout()
        plt.show()