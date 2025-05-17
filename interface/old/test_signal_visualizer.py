import torch
import numpy as np
from signal_visualizer import SignalVisualizer
import matplotlib.pyplot as plt

def create_example_signals(n_devices=3, n_timesteps=200, has_color=False, n_channels=4, seed=42):
    """
    Create example signals for testing.
    
    Args:
        n_devices: Number of devices/signals
        n_timesteps: Number of timesteps
        has_color: Whether to include color channels
        n_channels: Number of color channels (if has_color is True)
        seed: Random seed for reproducibility
        
    Returns:
        PyTorch tensor with example signals
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if has_color:
        # Create 3D tensor: devices × timesteps × channels
        signals = torch.zeros((n_devices, n_timesteps, n_channels))
        
        for i in range(n_devices):
            # Generate some interesting patterns
            t = torch.linspace(0, 8 * np.pi, n_timesteps)
            
            # RGB values - create different wave patterns
            signals[i, :, 0] = 0.5 + 0.5 * torch.sin(t + i * np.pi / 3)  # Red
            signals[i, :, 1] = 0.5 + 0.5 * torch.sin(t + i * np.pi / 2)  # Green
            signals[i, :, 2] = 0.5 + 0.5 * torch.sin(t + i * np.pi)      # Blue
            
            # Alpha/intensity values
            if n_channels >= 4:
                # Make some interesting envelope patterns for alpha
                alpha = torch.zeros(n_timesteps)
                
                # Create pulses
                for j in range(5):
                    center = n_timesteps * (j + 0.5) / 5
                    width = n_timesteps / 15
                    pulse = torch.exp(-((torch.arange(n_timesteps) - center) / width) ** 2)
                    alpha += pulse
                
                # Normalize to [0, 1]
                alpha = alpha / alpha.max()
                signals[i, :, 3] = alpha
            
    else:
        # Create 2D tensor: devices × timesteps
        signals = torch.zeros((n_devices, n_timesteps))
        
        for i in range(n_devices):
            # Generate some interesting patterns
            t = torch.linspace(0, 8 * np.pi, n_timesteps)
            
            # Base wave
            base_wave = 0.5 + 0.5 * torch.sin(t + i * np.pi / 2)
            
            # Add modulation to make it more interesting
            mod_wave = 0.5 + 0.5 * torch.sin(0.2 * t + i * np.pi / 3)
            
            signals[i, :] = base_wave * mod_wave
    
    return signals

def create_simple_hardcoded_signals():
    """
    Create a simple, hard-coded 2D tensor with few timesteps for clear visualization.
    
    Returns:
        PyTorch tensor with shape [3, 10] (3 devices, 10 timesteps)
    """
    # Create a 2D tensor: 3 devices × 10 timesteps
    signals = torch.zeros((3, 10))
    
    # Device 1: Increasing signal
    signals[0] = torch.tensor([0, 0, 0.3, 0.5, 0.5, 1, 1, 1, 0.4, 0.2])
    
    # Device 2: Alternating signal
    signals[1] = torch.tensor([0.0, 0.8, 0.1, 0.9, 0.2, 1.0, 0.3, 0.7, 0.0, 0.5])
    
    # Device 3: Pulse signal
    signals[2] = torch.tensor([0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1, 0.5, 0.8, 0.2])
    
    return signals

def main():
    # Test with hard-coded 2D tensor (10 timesteps)
    print("Testing with hard-coded 2D tensor (10 timesteps)")
    signals_simple = create_simple_hardcoded_signals()
    visualizer_simple = SignalVisualizer(signals_simple, duration=5)
    fig_simple, _ = visualizer_simple.visualization()
    
    # Save the figure
    fig_simple.savefig("signals_simple_visualization.png")
    print(f"Simple signal visualization saved to signals_simple_visualization.png")
    
    # Test with 2D tensor (no color channels)
    print("\nTesting with 2D tensor (grayscale signals)")
    signals_2d = create_example_signals(n_devices=4, n_timesteps=500, has_color=False)
    visualizer_2d = SignalVisualizer(signals_2d, duration=15)
    fig_2d, _ = visualizer_2d.visualization()
    
    # Save the figure
    fig_2d.savefig("signals_2d_visualization.png")
    print(f"2D signal visualization saved to signals_2d_visualization.png")
    
    # Test with 3D tensor (with RGBA channels)
    print("\nTesting with 3D tensor (RGBA signals)")
    signals_3d = create_example_signals(n_devices=3, n_timesteps=500, has_color=True, n_channels=4)
    visualizer_3d = SignalVisualizer(signals_3d, duration=15, max_value=1)
    fig_3d, _ = visualizer_3d.visualization()
    
    # Save the figure
    fig_3d.savefig("signals_3d_visualization.png")
    print(f"3D signal visualization saved to signals_3d_visualization.png")
    
    # Show all figures
    plt.show()

if __name__ == "__main__":
    main() 