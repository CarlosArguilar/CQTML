import torch
import numpy as np
import sys
import os

# Add parent directory to path to import from main package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import SignalVisualizerApp
from examples.test_visualization import create_custom_signal, create_colored_signal


def create_alternative_signal(base_signal):
    """
    Create an alternative signal based on the base signal for comparison.
    
    Args:
        base_signal (torch.Tensor): Base signal tensor
        
    Returns:
        torch.Tensor: Modified signal tensor for comparison
    """
    # Clone the base signal
    alt_signal = base_signal.clone()
    
    # Modify the signal
    n_devices, n_timesteps = base_signal.shape[:2]
    
    # Check if we have color channels
    has_color = len(base_signal.shape) == 3
    
    if has_color:
        # For colored signals
        n_channels = base_signal.shape[2]
        
        # Device 1: Modify intensity pattern
        t = torch.linspace(0, 4 * np.pi, n_timesteps)
        alt_signal[0, :, 3] = 0.5 + 0.5 * torch.sin(1.5 * t)  # Different frequency
        
        # Device 2: Different progression
        alt_signal[1, :, 3] = torch.pow(torch.linspace(0, 1, n_timesteps), 2)  # Quadratic increase
        
        # Device 3: Phase shifted square wave
        square_wave = torch.zeros(n_timesteps)
        period = n_timesteps // 6  # Different period
        for i in range(0, n_timesteps, period):
            end = min(i + period // 2, n_timesteps)
            square_wave[i:end] = 1.0
        alt_signal[2, :, 3] = square_wave
        
        # Device 4: Different color pattern
        t = torch.linspace(0, 3 * np.pi, n_timesteps)  # Different periodicity
        alt_signal[3, :, 0] = 0.5 + 0.5 * torch.sin(1.2 * t)
        alt_signal[3, :, 1] = 0.5 + 0.5 * torch.sin(1.2 * t + 2*np.pi/3)
        alt_signal[3, :, 2] = 0.5 + 0.5 * torch.sin(1.2 * t + 4*np.pi/3)
    else:
        # For grayscale signals
        # Device 1: Modified sine wave
        t = torch.linspace(0, 4 * np.pi, n_timesteps)
        alt_signal[0] = 0.6 + 0.4 * torch.sin(1.5 * t)  # Different amplitude and frequency
        
        # Device 2: Different progression
        alt_signal[1] = torch.pow(torch.linspace(0, 1, n_timesteps), 2)  # Quadratic increase
        
        # Device 3: Phase shifted square wave
        square_wave = torch.zeros(n_timesteps)
        period = n_timesteps // 6  # Different period
        for i in range(0, n_timesteps, period):
            end = min(i + period // 2, n_timesteps)
            square_wave[i:end] = 1.0
        alt_signal[2] = square_wave
        
        # Device 4: Modified random pattern
        random_values = torch.rand(n_timesteps // 15)  # Different resolution
        # Interpolate to get smooth random pattern
        indices = torch.linspace(0, n_timesteps // 15 - 1, n_timesteps)
        indices_floor = indices.floor().long()
        indices_ceil = indices.ceil().long().clamp(max=n_timesteps // 15 - 1)
        weights_ceil = indices - indices_floor
        weights_floor = 1.0 - weights_ceil
        
        alt_signal[3] = weights_floor * random_values[indices_floor] + weights_ceil * random_values[indices_ceil]
    
    return alt_signal


def on_selection_callback(selection):
    """
    Handle selection of the best signal.
    
    Args:
        selection (str): Selected signal ('left' or 'right')
    """
    print(f"\nUser selected the {selection} signal as the best.")
    print("You can implement your own logic here to save or process the selection.")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Signal Comparison")
    parser.add_argument("--color", action="store_true", help="Use colored signals (RGBA)")
    args = parser.parse_args()
    
    # Create and run application
    app = SignalVisualizerApp()
    
    # Create signal data based on arguments
    if args.color:
        # Colored signal
        signal_tensor1, audio_tensor, sample_rate = create_colored_signal()
    else:
        # Grayscale signal
        signal_tensor1, audio_tensor, sample_rate = create_custom_signal()
    
    # Create an alternative signal for comparison
    signal_tensor2 = create_alternative_signal(signal_tensor1)
    
    # Set up comparison mode
    app.set_comparison_mode(
        signal_tensor1, 
        signal_tensor2, 
        audio_tensor, 
        sample_rate=sample_rate,
        selection_callback=on_selection_callback
    )
    
    # Run the application
    app.run() 