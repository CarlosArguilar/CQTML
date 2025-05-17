import pygame
import sys
import argparse
import torch
import numpy as np

from src.ui.theme import COLORS, SIZES
from src.ui.screens.visualization import VisualizationScreen
from src.ui.screens.comparison import ComparisonScreen


class SignalVisualizerApp:
    """
    Main application for signal visualization and comparison.
    """
    
    def __init__(self):
        """Initialize the application"""
        pygame.init()
        
        # Initialize screen
        self.screen_width = SIZES['window_width']
        self.screen_height = SIZES['window_height']
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Signal Visualizer")
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Initialize screens
        self.visualization_screen = VisualizationScreen(self.screen_width, self.screen_height)
        self.comparison_screen = ComparisonScreen(self.screen_width, self.screen_height)
        
        # Current mode
        self.current_mode = "visualization"  # or "comparison"
        
    def set_visualization_mode(self, signal_tensor, audio_tensor, duration=None, 
                              sample_rate=44100, max_value=1.0):
        """
        Set up visualization mode with data.
        
        Args:
            signal_tensor (torch.Tensor): Signal tensor
            audio_tensor (torch.Tensor): Audio tensor
            duration (float, optional): Signal duration in seconds
            sample_rate (int, optional): Audio sample rate
            max_value (float, optional): Maximum signal value
        """
        self.current_mode = "visualization"
        self.visualization_screen.set_data(
            signal_tensor, audio_tensor, duration, sample_rate, max_value
        )
        
    def set_comparison_mode(self, signal_tensor_left, signal_tensor_right, audio_tensor,
                           duration=None, sample_rate=44100, max_value=1.0, 
                           selection_callback=None):
        """
        Set up comparison mode with data.
        
        Args:
            signal_tensor_left (torch.Tensor): Left signal tensor
            signal_tensor_right (torch.Tensor): Right signal tensor
            audio_tensor (torch.Tensor): Audio tensor
            duration (float, optional): Signal duration in seconds
            sample_rate (int, optional): Audio sample rate
            max_value (float, optional): Maximum signal value
            selection_callback (function, optional): Callback when a signal is selected
        """
        self.current_mode = "comparison"
        self.comparison_screen.set_data(
            signal_tensor_left, signal_tensor_right, audio_tensor,
            duration, sample_rate, max_value, selection_callback
        )
        
    def run(self):
        """Run the main application loop"""
        running = True
        
        while running:
            # Calculate delta time
            dt = self.clock.tick(self.fps) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                
                # Send event to current screen
                if self.current_mode == "visualization":
                    self.visualization_screen.handle_event(event)
                else:
                    self.comparison_screen.handle_event(event)
            
            # Update current screen
            if self.current_mode == "visualization":
                self.visualization_screen.update(dt)
            else:
                self.comparison_screen.update(dt)
                
            # Draw current screen
            if self.current_mode == "visualization":
                self.visualization_screen.draw(self.screen)
            else:
                self.comparison_screen.draw(self.screen)
                
            # Update the display
            pygame.display.flip()
            
        # Clean up
        pygame.quit()
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Signal Visualizer Application")
    parser.add_argument("--mode", choices=["visualization", "comparison"], default="visualization",
                      help="Application mode: visualization or comparison")
    args = parser.parse_args()
    
    # Create and run application
    app = SignalVisualizerApp()
    
    # For demonstration, create some example data
    def create_example_data():
        # Create a simple sine wave audio signal
        sample_rate = 44100
        duration = 5.0  # 5 seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio_tensor = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create signal tensor with 4 devices and varying patterns
        n_devices = 4
        n_timesteps = 200
        signal_tensor = torch.zeros((n_devices, n_timesteps))
        
        # Create different patterns for each device
        for i in range(n_devices):
            t_signal = torch.linspace(0, 8 * np.pi, n_timesteps)
            signal_tensor[i] = 0.5 + 0.5 * torch.sin(t_signal + i * np.pi / 2)
        
        return signal_tensor, audio_tensor, sample_rate
    
    # Set up the requested mode with example data
    if args.mode == "visualization":
        signal_tensor, audio_tensor, sample_rate = create_example_data()
        app.set_visualization_mode(signal_tensor, audio_tensor, sample_rate=sample_rate)
    else:
        signal_tensor1, audio_tensor, sample_rate = create_example_data()
        
        # Create a slightly different pattern for the second signal
        n_devices = 4
        n_timesteps = 200
        signal_tensor2 = torch.zeros((n_devices, n_timesteps))
        for i in range(n_devices):
            t_signal = torch.linspace(0, 8 * np.pi, n_timesteps)
            # Different phase and frequency
            signal_tensor2[i] = 0.5 + 0.5 * torch.sin(1.2 * t_signal + i * np.pi / 3)
        
        # Set up comparison mode
        def on_selection(selection):
            print(f"Selected '{selection}' as the best signal")
            
        app.set_comparison_mode(signal_tensor1, signal_tensor2, audio_tensor, 
                               sample_rate=sample_rate, selection_callback=on_selection)
    
    # Run the application
    app.run() 