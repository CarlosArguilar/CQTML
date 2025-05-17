import torch
import numpy as np
import sys
import os

# Add parent directory to path to import from main package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import SignalVisualizerApp


def create_custom_signal():
    """
    Create a more complex example signal for testing visualization mode.
    
    Returns:
        tuple: (signal_tensor, audio_tensor, sample_rate)
    """
    # Audio signal parameters
    sample_rate = 44100
    duration = 8.0  # 8 seconds
    n_samples = int(sample_rate * duration)
    
    # Create a more interesting audio signal (dual tone with fade-in/fade-out)
    t = torch.linspace(0, duration, n_samples)
    
    # Create two frequency components
    audio_440 = torch.sin(2 * np.pi * 440 * t)  # 440 Hz (A4)
    audio_880 = 0.5 * torch.sin(2 * np.pi * 880 * t)  # 880 Hz (A5)
    
    # Create envelope (fade in and fade out)
    fade_time = 1.0  # 1 second fade in/out
    fade_samples = int(fade_time * sample_rate)
    
    # Linear fade in and fade out
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    
    # Create full envelope (fade in, sustain, fade out)
    envelope = torch.ones(n_samples)
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out
    
    # Combine components with envelope
    audio_tensor = (audio_440 + audio_880) * envelope
    
    # Signal tensor (4 devices with different patterns)
    n_devices = 4
    n_timesteps = 400  # More timesteps for smoother visualization
    signal_tensor = torch.zeros((n_devices, n_timesteps))
    
    # Time vector for signals
    t_signal = torch.linspace(0, duration, n_timesteps)
    
    # Device 1: Pulsating pattern
    signal_tensor[0] = 0.7 + 0.3 * torch.sin(2 * np.pi * 0.5 * t_signal)
    
    # Device 2: Gradual increase then decrease
    midpoint = n_timesteps // 2
    signal_tensor[1, :midpoint] = torch.linspace(0.1, 1.0, midpoint)
    signal_tensor[1, midpoint:] = torch.linspace(1.0, 0.1, n_timesteps - midpoint)
    
    # Device 3: Square wave pattern
    square_wave = torch.zeros(n_timesteps)
    period = n_timesteps // 8
    for i in range(0, n_timesteps, period):
        end = min(i + period // 2, n_timesteps)
        square_wave[i:end] = 1.0
    signal_tensor[2] = square_wave
    
    # Device 4: Random pattern with smoothing
    random_values = torch.rand(n_timesteps // 20)
    # Interpolate to get smooth random pattern
    indices = torch.linspace(0, n_timesteps // 20 - 1, n_timesteps)
    indices_floor = indices.floor().long()
    indices_ceil = indices.ceil().long().clamp(max=n_timesteps // 20 - 1)
    weights_ceil = indices - indices_floor
    weights_floor = 1.0 - weights_ceil
    
    signal_tensor[3] = weights_floor * random_values[indices_floor] + weights_ceil * random_values[indices_ceil]
    
    return signal_tensor, audio_tensor, sample_rate


def create_colored_signal():
    """
    Create a signal with RGB color channels for testing.
    
    Returns:
        tuple: (signal_tensor, audio_tensor, sample_rate)
    """
    # Get base signal
    grayscale_signal, audio_tensor, sample_rate = create_custom_signal()
    
    # Extract dimensions
    n_devices, n_timesteps = grayscale_signal.shape
    
    # Create colored signal with RGBA channels
    colored_signal = torch.zeros((n_devices, n_timesteps, 4))
    
    # Device 1: Red with pulsating alpha
    colored_signal[0, :, 0] = 1.0  # Full red
    colored_signal[0, :, 3] = grayscale_signal[0]  # Alpha from grayscale
    
    # Device 2: Green with increasing/decreasing alpha
    colored_signal[1, :, 1] = 1.0  # Full green
    colored_signal[1, :, 3] = grayscale_signal[1]  # Alpha from grayscale
    
    # Device 3: Blue with square pattern alpha
    colored_signal[2, :, 2] = 1.0  # Full blue
    colored_signal[2, :, 3] = grayscale_signal[2]  # Alpha from grayscale
    
    # Device 4: RGB color changing over time with fixed alpha
    t = torch.linspace(0, 2 * np.pi, n_timesteps)
    colored_signal[3, :, 0] = 0.5 + 0.5 * torch.sin(t)  # Red varies with sine
    colored_signal[3, :, 1] = 0.5 + 0.5 * torch.sin(t + 2*np.pi/3)  # Green varies with sine (offset)
    colored_signal[3, :, 2] = 0.5 + 0.5 * torch.sin(t + 4*np.pi/3)  # Blue varies with sine (offset)
    colored_signal[3, :, 3] = grayscale_signal[3]  # Alpha from grayscale
    
    return colored_signal, audio_tensor, sample_rate


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Signal Visualization")
    parser.add_argument("--color", action="store_true", help="Use colored signals (RGBA)")
    args = parser.parse_args()
    
    # Create and run application
    app = SignalVisualizerApp()
    
    # Create signal data based on arguments
    if args.color:
        signal_tensor, audio_tensor, sample_rate = create_colored_signal()
    else:
        signal_tensor, audio_tensor, sample_rate = create_custom_signal()
    
    # Set up visualization mode
    app.set_visualization_mode(signal_tensor, audio_tensor, sample_rate=sample_rate)
    
    # Run the application
    app.run()