import torch
import pygame
import sys
import os
import threading
import time
from typing import Optional, Union, Tuple

# Import the main application
from cqtml_interface.main import SignalVisualizerApp


def visualize_signal(
    audio_tensor: torch.Tensor,
    signal_tensor: torch.Tensor,
    sample_rate: int = 44100,
    duration: Optional[float] = None,
    max_value: float = 1.0
) -> None:
    """
    Visualize a signal with synchronized audio using the Signal Visualizer interface.
    
    This function opens an interactive PyGame window displaying a visual representation
    of the signal(s) with synchronized audio playback. The visualization shows devices 
    as circles that light up based on the signal values at each timestep.
    
    Args:
        audio_tensor (torch.Tensor): 
            Audio tensor (mono) with shape [samples]. This should be a 1D tensor containing
            audio samples with values typically in the range [-1.0, 1.0].
        
        signal_tensor (torch.Tensor): 
            Signal tensor that can have one of these formats:
            - [devices, timesteps]: For monochrome signals (2D tensor).
              Each device will use a default color with varying brightness.
            - [devices, timesteps, channels]: For colored signals (3D tensor).
              The channels can be:
              * 1 channel: Grayscale intensity
              * 3 channels: RGB values (each in range [0,1])
              * 4 channels: RGBA values (RGB + alpha/intensity)
        
        sample_rate (int, optional): 
            Audio sample rate in Hz. Defaults to 44100.
        
        duration (float, optional): 
            Signal duration in seconds. If None (default), it will be calculated from 
            the audio length and sample rate. If provided, the signal will be 
            synchronized to this duration.
        
        max_value (float, optional): 
            Maximum signal value for normalization. Defaults to 1.0.
            Signal values will be divided by this value to normalize them to [0,1].
    
    Returns:
        None: The function blocks until the visualization window is closed.
    
    Notes:
        - The signal visualization uses discrete rendering - for values below 0.5,
          the device appears off (black), and for values >= 0.5, the device appears
          on with its assigned color.
        - The interface includes a timeline for seeking and a play/pause button.
        - Press ESC or close the window to exit.
        - The audio will loop back to the beginning when it reaches the end.
        
    Example:
        ```python
        import torch
        from cqtml_interface import visualize_signal
        
        # Create a simple audio sine wave
        sample_rate = 44100
        duration = 5.0  # 5 seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio_tensor = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone
        
        # Create a simple binary signal for 4 devices
        # Format: [devices, timesteps]
        signal_tensor = torch.zeros((4, 100))
        for i in range(4):
            for j in range(100):
                if (j + i) % 4 == 0:
                    signal_tensor[i, j] = 1.0  # Turn device on
        
        # Visualize the signal
        visualize_signal(audio_tensor, signal_tensor, sample_rate)
        ```
    """
    # Initialize pygame if not already done
    if not pygame.get_init():
        pygame.init()
        
    # Create the visualizer app
    app = SignalVisualizerApp()
    
    # Set visualization mode with the provided tensors
    app.set_visualization_mode(
        signal_tensor=signal_tensor,
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        duration=duration,
        max_value=max_value
    )
    
    # Run the application (will block until closed)
    app.run()
    
    # Clean up pygame
    pygame.quit()


def compare_signals(
    audio_tensor: torch.Tensor,
    signal_tensor_a: torch.Tensor,
    signal_tensor_b: torch.Tensor,
    sample_rate: int = 44100,
    duration: Optional[float] = None,
    max_value: float = 1.0,
    timeout: Optional[float] = None
) -> Optional[int]:
    """
    Compare two signals with synchronized audio and return the preferred option.
    
    This function opens an interactive PyGame window displaying two signal visualizations
    side by side (vertically stacked), both synchronized with the same audio. The user
    can select which signal they prefer, and the function returns their selection.
    
    Args:
        audio_tensor (torch.Tensor): 
            Audio tensor (mono) with shape [samples]. This should be a 1D tensor containing
            audio samples with values typically in the range [-1.0, 1.0].
        
        signal_tensor_a (torch.Tensor): 
            First signal tensor (Signal A) that can have one of these formats:
            - [devices, timesteps]: For monochrome signals (2D tensor).
              Each device will use a default color with varying brightness.
            - [devices, timesteps, channels]: For colored signals (3D tensor).
              The channels can be:
              * 1 channel: Grayscale intensity
              * 3 channels: RGB values (each in range [0,1])
              * 4 channels: RGBA values (RGB + alpha/intensity)
        
        signal_tensor_b (torch.Tensor): 
            Second signal tensor (Signal B) with the same format as signal_tensor_a.
            Can have a different number of devices from signal_tensor_a.
        
        sample_rate (int, optional): 
            Audio sample rate in Hz. Defaults to 44100.
        
        duration (float, optional): 
            Signal duration in seconds. If None (default), it will be calculated from 
            the audio length and sample rate. If provided, the signals will be 
            synchronized to this duration.
        
        max_value (float, optional): 
            Maximum signal value for normalization. Defaults to 1.0.
            Signal values will be divided by this value to normalize them to [0,1].
        
        timeout (float, optional): 
            Maximum time in seconds to wait for the user to make a selection.
            If None (default), the function will wait indefinitely until a selection
            is made or the window is closed.
    
    Returns:
        Optional[int]: 
            - 0 if Signal A (top) was selected
            - 1 if Signal B (bottom) was selected
            - None if no selection was made (window closed without selection)
            or if the timeout was reached without a selection
    
    Notes:
        - The signal visualization uses discrete rendering - for values below 0.5,
          the device appears off (black), and for values >= 0.5, the device appears
          on with its assigned color.
        - The interface includes buttons to select either Signal A or Signal B,
          a shared timeline for seeking, and a play/pause button.
        - Press ESC or close the window to exit without making a selection.
        - The audio will loop back to the beginning when it reaches the end.
        - This function is particularly useful for A/B testing of signal generation
          algorithms or for collecting user preferences in experiments.
        
    Example:
        ```python
        import torch
        from cqtml_interface import compare_signals
        
        # Create a simple audio sine wave
        sample_rate = 44100
        duration = 5.0  # 5 seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio_tensor = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone
        
        # Create two different signals for comparison
        signal_a = torch.zeros((4, 100))
        signal_b = torch.zeros((4, 100))
        
        # Pattern A: Every 4th timestep
        for i in range(4):
            for j in range(100):
                if (j + i) % 4 == 0:
                    signal_a[i, j] = 1.0
        
        # Pattern B: Every 3rd timestep
        for i in range(4):
            for j in range(100):
                if (j + i) % 3 == 0:
                    signal_b[i, j] = 1.0
        
        # Compare signals and get user preference
        result = compare_signals(audio_tensor, signal_a, signal_b, sample_rate)
        
        if result == 0:
            print("User preferred Signal A")
        elif result == 1:
            print("User preferred Signal B")
        else:
            print("No selection was made")
        ```
    """
    # Initialize pygame if not already done
    if not pygame.get_init():
        pygame.init()
    
    # Variable to store the selection
    selection_result = {"selected": None, "done": False}
    
    # Callback function to capture the selection
    def selection_callback(selection):
        if selection == "top":
            selection_result["selected"] = 0  # Signal A
        elif selection == "bottom":
            selection_result["selected"] = 1  # Signal B
        selection_result["done"] = True
        
    # Create the visualizer app
    app = SignalVisualizerApp()
    
    # Set comparison mode with the provided tensors
    app.set_comparison_mode(
        signal_tensor_left=signal_tensor_a,
        signal_tensor_right=signal_tensor_b,
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        duration=duration,
        max_value=max_value,
        selection_callback=selection_callback
    )
    
    # Run in a separate thread if a timeout is specified
    if timeout is not None:
        def run_app():
            app.run(exit_on_callback=True)
            
        app_thread = threading.Thread(target=run_app)
        app_thread.daemon = True
        app_thread.start()
        
        # Wait for selection or timeout
        start_time = time.time()
        while not selection_result["done"] and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        # Force exit if timeout is reached
        if not selection_result["done"]:
            app.force_exit = True
    else:
        # Run normally, will exit when selection is made or window is closed
        app.run(exit_on_callback=True)
    
    # Clean up pygame
    pygame.quit()
    
    return selection_result["selected"] 