import pygame
import pygame.gfxdraw
from ...ui.theme import COLORS, SIZES, FONTS
from ...ui.components.button import Button
from ...ui.components.timeline import Timeline
from ...ui.components.device_renderer import DeviceRenderer
from ...core.signal_processor import SignalProcessor
from ...core.audio_processor import AudioProcessor


class VisualizationScreen:
    """
    Visualization mode screen for displaying signal data with audio playback.
    """
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the visualization screen.
        
        Args:
            screen_width (int): Screen width
            screen_height (int): Screen height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize components
        self._init_components()
        
        # State
        self.signal_processor = None
        self.audio_processor = None
        self.playing = False
        
    def _init_components(self):
        """Initialize UI components"""
        # Play/pause button - positioned at the bottom center
        button_x = (self.screen_width // 2) - (SIZES['button_width'] // 2)
        button_y = self.screen_height - SIZES['button_height'] - SIZES['controls_padding_bottom']
        self.play_button = Button(
            button_x, 
            button_y, 
            text="Play", 
            callback=self._toggle_playback,
            toggle=True,
            style="primary"
        )
        
        # Timeline - positioned above the play button
        timeline_y = button_y - SIZES['timeline_height'] - SIZES['padding'] * 2
        timeline_width = int(self.screen_width * 0.7)  # 70% of screen width
        timeline_x = (self.screen_width - timeline_width) // 2
        self.timeline = Timeline(
            timeline_x,
            timeline_y,
            timeline_width,
            10.0,  # Default duration
            on_seek=self._on_seek
        )
        
        # Device renderer - centered horizontally with proper vertical position
        # Calculate starting x position to center all devices
        n_default_devices = 4
        total_width = (n_default_devices - 1) * SIZES['device_circle_spacing']
        start_x = (self.screen_width - total_width) // 2
        
        self.device_renderer = DeviceRenderer(
            start_x,
            SIZES['content_padding_top'] + SIZES['device_circle_y'],
            n_default_devices,
            horizontal=True
        )
        
        # Title
        self.title = "Signal Visualization"
        
    def set_data(self, signal_tensor, audio_tensor, duration=None, sample_rate=44100, max_value=1.0):
        """
        Set the signal and audio data for visualization.
        
        Args:
            signal_tensor (torch.Tensor): Signal tensor
            audio_tensor (torch.Tensor): Audio tensor
            duration (float, optional): Signal duration in seconds
            sample_rate (int, optional): Audio sample rate
            max_value (float, optional): Maximum signal value
        """
        # Create signal processor
        self.signal_processor = SignalProcessor(signal_tensor, duration, max_value)
        
        # Create audio processor
        self.audio_processor = AudioProcessor(audio_tensor, sample_rate)
        
        # Synchronize durations
        self.signal_processor.synchronize_with_audio(audio_tensor, sample_rate)
        
        # Update timeline duration
        self.timeline.set_duration(self.audio_processor.duration)
        
        # Update device renderer
        self.device_renderer.set_signal_processor(self.signal_processor)
        
        # Update device renderer position to center all devices
        n_devices = self.signal_processor.n_devices
        total_width = (n_devices - 1) * SIZES['device_circle_spacing']
        start_x = (self.screen_width - total_width) // 2
        self.device_renderer.set_position(start_x, SIZES['content_padding_top'] + SIZES['device_circle_y'])
        
        # Pause playback if active
        if self.playing:
            self._toggle_playback()
            
    def update(self, delta_time):
        """
        Update the screen state.
        
        Args:
            delta_time (float): Time since last update in seconds
        """
        if not self.audio_processor:
            return
            
        # Check if audio was playing but has stopped (reached the end)
        was_playing = self.playing
        
        # Update playback state
        self.playing = self.audio_processor.playing
        
        # Get current progress to check if we're near the end
        current_time = self.audio_processor.get_current_time()
        duration = self.audio_processor.duration
        
        # Check if the audio has just finished (was playing but now stopped AND we're near the end)
        # Only reset if we were near the end when playback stopped (natural finish)
        if was_playing and not self.playing and current_time > (duration * 0.98):
            # Reset to beginning only if we actually reached the end
            self.audio_processor.set_time(0)
        
        # Update button text based on playback state
        self.play_button.set_text("Pause" if self.playing else "Play")
        self.play_button.set_toggled(self.playing)
        
        # Update timeline
        progress = self.audio_processor.get_progress()
        self.timeline.update_progress(progress, current_time, duration)
        
        # Update device values
        self.device_renderer.update_values(current_time)
        
    def draw(self, surface):
        """
        Draw the screen on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # Clear the screen
        surface.fill(COLORS['background'])
        
        # Draw title
        font_name, font_size, bold = FONTS['title']
        title_font = pygame.font.SysFont(font_name, font_size, bold)
        title_surface = title_font.render(self.title, True, COLORS['text'])
        title_x = (self.screen_width - title_surface.get_width()) // 2
        title_y = SIZES['title_padding_top']
        surface.blit(title_surface, (title_x, title_y))
        
        # Draw device renderer
        self.device_renderer.draw(surface)
        
        # Draw timeline
        self.timeline.draw(surface)
        
        # Draw play/pause button
        self.play_button.draw(surface)
        
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event (pygame.event.Event): Pygame event to handle
            
        Returns:
            bool: True if the event was handled
        """
        # Handle play/pause button
        if self.play_button.handle_event(event):
            return True
            
        # Handle timeline
        if self.timeline.handle_event(event):
            return True
            
        return False
        
    def _toggle_playback(self):
        """Toggle audio playback"""
        if not self.audio_processor:
            return
            
        if self.audio_processor.playing:
            self.audio_processor.pause()
        else:
            self.audio_processor.play()
            
    def _on_seek(self, time_sec):
        """
        Handle seeking in the timeline.
        
        Args:
            time_sec (float): Time to seek to in seconds
        """
        if not self.audio_processor:
            return
            
        self.audio_processor.set_time(time_sec) 