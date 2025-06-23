#!/usr/bin/env python3
"""
Test script for volume slider functionality.
Demonstrates the volume slider in both comparison and visualization screens.
"""

import os
import sys
import torch
import numpy as np
import pygame

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cqtml_interface.src.ui.screens.visualization import VisualizationScreen
from cqtml_interface.src.ui.screens.comparison import ComparisonScreen
from cqtml_interface.src.ui.theme import COLORS, SIZES


def create_test_audio_signal(duration=5.0, sample_rate=22050, frequency=440.0):
    """
    Create a test audio signal and corresponding signal tensor.
    
    Args:
        duration (float): Duration in seconds
        sample_rate (int): Sample rate in Hz
        frequency (float): Sine wave frequency in Hz
        
    Returns:
        tuple: (signal_tensor, audio_tensor)
    """
    # Create time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create audio signal (sine wave with envelope)
    envelope = np.exp(-t / 2.0)  # Exponential decay
    audio = envelope * np.sin(2 * np.pi * frequency * t)
    audio_tensor = torch.from_numpy(audio.astype(np.float32))
    
    # Create signal tensor (4 devices with different patterns)
    n_devices = 4
    n_samples = len(t)
    signal_tensor = torch.zeros(n_devices, n_samples)
    
    for i in range(n_devices):
        # Each device has a different frequency and phase
        device_freq = frequency * (i + 1) / n_devices
        phase = i * np.pi / 4
        device_signal = envelope * np.sin(2 * np.pi * device_freq * t + phase)
        signal_tensor[i] = torch.from_numpy(device_signal.astype(np.float32))
    
    return signal_tensor, audio_tensor


def test_visualization_screen():
    """Test the volume slider in visualization screen."""
    print("🎵 Testing Volume Slider in Visualization Screen")
    print("=" * 50)
    
    # Initialize pygame
    pygame.init()
    
    # Create window
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Volume Slider Test - Visualization")
    
    # Create test data
    signal_tensor, audio_tensor = create_test_audio_signal(duration=5.0)
    
    # Create visualization screen
    viz_screen = VisualizationScreen(screen_width, screen_height)
    viz_screen.set_data(signal_tensor, audio_tensor, duration=5.0, sample_rate=22050)
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    print("📋 Volume Slider Controls:")
    print("  • Click and drag the volume slider to adjust volume")
    print("  • Volume range: 0.0x (mute) to 10.0x (maximum volume)")
    print("  • Volume slider is located in bottom right corner")
    print("  • Visual indicators show volume level")
    print("  • Press SPACE to play/pause")
    print("  • Press ESC to exit")
    print()
    
    while running:
        delta_time = clock.tick(60) / 1000.0  # 60 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    viz_screen._toggle_playback()
            else:
                viz_screen.handle_event(event)
        
        # Update and draw
        viz_screen.update(delta_time)
        viz_screen.draw(screen)
        
        # Display current volume
        if hasattr(viz_screen, 'volume_slider'):
            current_volume = viz_screen.volume_slider.get_volume()
            volume_text = f"Current Volume: {current_volume:.1f}x"
            font = pygame.font.Font(None, 24)
            volume_surface = font.render(volume_text, True, COLORS['text'])
            screen.blit(volume_surface, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    print("✅ Visualization screen test completed!")


def test_comparison_screen():
    """Test the volume slider in comparison screen."""
    print("🎵 Testing Volume Slider in Comparison Screen")
    print("=" * 50)
    
    # Initialize pygame
    pygame.init()
    
    # Create window
    screen_width, screen_height = 1024, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Volume Slider Test - Comparison")
    
    # Create test data
    signal_tensor_a, audio_tensor = create_test_audio_signal(duration=5.0, frequency=440.0)
    signal_tensor_b, _ = create_test_audio_signal(duration=5.0, frequency=523.25)  # C note
    
    # Create comparison screen
    comp_screen = ComparisonScreen(screen_width, screen_height)
    comp_screen.set_data(
        signal_tensor_a, signal_tensor_b, audio_tensor,
        duration=5.0, sample_rate=22050
    )
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    print("📋 Volume Slider Controls:")
    print("  • Click and drag the volume slider to adjust volume")
    print("  • Volume range: 0.0x (mute) to 10.0x (maximum volume)")
    print("  • Volume slider is located in bottom right corner")
    print("  • Use selection buttons to choose preferred signal")
    print("  • Press SPACE to play/pause")
    print("  • Press ESC to exit")
    print()
    
    while running:
        delta_time = clock.tick(60) / 1000.0  # 60 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    comp_screen._toggle_playback()
            else:
                comp_screen.handle_event(event)
        
        # Update and draw
        comp_screen.update(delta_time)
        comp_screen.draw(screen)
        
        # Display current volume
        if hasattr(comp_screen, 'volume_slider'):
            current_volume = comp_screen.volume_slider.get_volume()
            volume_text = f"Current Volume: {current_volume:.1f}x"
            font = pygame.font.Font(None, 24)
            volume_surface = font.render(volume_text, True, COLORS['text'])
            screen.blit(volume_surface, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    print("✅ Comparison screen test completed!")


def main():
    """Main test function."""
    print("🎮 Volume Slider Test Suite")
    print("=" * 50)
    print()
    
    try:
        # Test visualization screen
        test_visualization_screen()
        print()
        
        # Test comparison screen  
        test_comparison_screen()
        print()
        
        print("🎉 All volume slider tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 