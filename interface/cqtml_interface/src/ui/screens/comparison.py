import pygame
import pygame.gfxdraw
from ...ui.theme import COLORS, SIZES, FONTS
from ...ui.components.button import Button
from ...ui.components.timeline import Timeline
from ...ui.components.device_renderer import DeviceRenderer
from ...core.signal_processor import SignalProcessor
from ...core.audio_processor import AudioProcessor


class ComparisonScreen:
    """
    Comparison mode screen for comparing two signal visualizations with shared audio.
    """
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the comparison screen.
        
        Args:
            screen_width (int): Screen width
            screen_height (int): Screen height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize components
        self._init_components()
        
        # State
        self.signal_processor_left = None
        self.signal_processor_right = None
        self.audio_processor = None
        self.playing = False
        self.selection_callback = None
        
    def _init_components(self):
        """Initialize UI components"""
        # Play/pause button
        button_x = (self.screen_width // 2) - (SIZES['button_width'] // 2)
        button_y = self.screen_height - SIZES['button_height'] - SIZES['padding'] * 3
        self.play_button = Button(
            button_x, 
            button_y, 
            text="Play", 
            callback=self._toggle_playback,
            toggle=True,
            style="primary"
        )
        
        # Timeline (shared between both visualizations)
        timeline_y = button_y - SIZES['timeline_height'] - SIZES['padding'] * 2
        timeline_width = self.screen_width - SIZES['padding'] * 4
        timeline_x = (self.screen_width - timeline_width) // 2
        self.timeline = Timeline(
            timeline_x,
            timeline_y,
            timeline_width,
            10.0,  # Default duration
            on_seek=self._on_seek
        )
        
        # Calculate panel dimensions
        panel_width = (self.screen_width // 2) - SIZES['padding'] * 2
        panel_height = timeline_y - SIZES['padding'] * 4
        
        # Left panel
        left_panel_x = SIZES['padding']
        left_panel_y = SIZES['padding'] * 2
        self.left_panel = pygame.Rect(left_panel_x, left_panel_y, panel_width, panel_height)
        
        # Right panel
        right_panel_x = self.screen_width // 2 + SIZES['padding']
        right_panel_y = SIZES['padding'] * 2
        self.right_panel = pygame.Rect(right_panel_x, right_panel_y, panel_width, panel_height)
        
        # Device renderers
        devices_y = left_panel_y + panel_height // 2
        
        # Left devices
        left_devices_x = left_panel_x + panel_width // 2
        self.device_renderer_left = DeviceRenderer(
            left_devices_x,
            devices_y,
            4,  # Default number of devices
            horizontal=True
        )
        
        # Right devices
        right_devices_x = right_panel_x + panel_width // 2
        self.device_renderer_right = DeviceRenderer(
            right_devices_x,
            devices_y,
            4,  # Default number of devices
            horizontal=True
        )
        
        # Selection buttons
        selection_button_width = 160
        selection_button_height = 40
        
        # Left selection button
        left_selection_x = left_panel_x + (panel_width - selection_button_width) // 2
        selection_y = timeline_y + SIZES['timeline_height'] + SIZES['padding'] * 2
        self.left_selection_button = Button(
            left_selection_x,
            selection_y,
            selection_button_width,
            selection_button_height,
            text="Select Left",
            callback=self._select_left,
            style="success"
        )
        
        # Right selection button
        right_selection_x = right_panel_x + (panel_width - selection_button_width) // 2
        self.right_selection_button = Button(
            right_selection_x,
            selection_y,
            selection_button_width,
            selection_button_height,
            text="Select Right",
            callback=self._select_right,
            style="success"
        )
        
        # Titles
        self.left_title = "Signal A"
        self.right_title = "Signal B"
        self.main_title = "Signal Comparison"
        
    def set_data(self, signal_tensor_left, signal_tensor_right, audio_tensor, 
                duration=None, sample_rate=44100, max_value=1.0, selection_callback=None):
        """
        Set the signal and audio data for comparison.
        
        Args:
            signal_tensor_left (torch.Tensor): Left signal tensor
            signal_tensor_right (torch.Tensor): Right signal tensor
            audio_tensor (torch.Tensor): Audio tensor
            duration (float, optional): Signal duration in seconds
            sample_rate (int, optional): Audio sample rate
            max_value (float, optional): Maximum signal value
            selection_callback (function, optional): Callback when a signal is selected
        """
        # Create signal processors
        self.signal_processor_left = SignalProcessor(signal_tensor_left, duration, max_value)
        self.signal_processor_right = SignalProcessor(signal_tensor_right, duration, max_value)
        
        # Create audio processor
        self.audio_processor = AudioProcessor(audio_tensor, sample_rate)
        
        # Synchronize durations
        self.signal_processor_left.synchronize_with_audio(audio_tensor, sample_rate)
        self.signal_processor_right.synchronize_with_audio(audio_tensor, sample_rate)
        
        # Update timeline duration
        self.timeline.set_duration(self.audio_processor.duration)
        
        # Update device renderers
        self.device_renderer_left.set_signal_processor(self.signal_processor_left)
        self.device_renderer_right.set_signal_processor(self.signal_processor_right)
        
        # Set selection callback
        self.selection_callback = selection_callback
        
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
            
        # Update playback state
        self.playing = self.audio_processor.playing
        
        # Update button text based on playback state
        self.play_button.set_text("Pause" if self.playing else "Play")
        self.play_button.set_toggled(self.playing)
        
        # Update timeline
        current_time = self.audio_processor.get_current_time()
        progress = self.audio_processor.get_progress()
        self.timeline.update_progress(progress, current_time, self.audio_processor.duration)
        
        # Update device values
        if self.signal_processor_left:
            self.device_renderer_left.update_values(current_time)
        if self.signal_processor_right:
            self.device_renderer_right.update_values(current_time)
        
    def draw(self, surface):
        """
        Draw the screen on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # Clear the screen
        surface.fill(COLORS['background'])
        
        # Draw panels
        pygame.draw.rect(surface, COLORS['dark_panel'], self.left_panel)
        pygame.draw.rect(surface, COLORS['dark_panel'], self.right_panel)
        
        # Draw main title
        font_name, font_size, bold = FONTS['title']
        title_font = pygame.font.SysFont(font_name, font_size, bold)
        title_surface = title_font.render(self.main_title, True, COLORS['text'])
        title_x = (self.screen_width - title_surface.get_width()) // 2
        title_y = SIZES['padding'] // 2
        surface.blit(title_surface, (title_x, title_y))
        
        # Draw panel titles
        font_name, font_size = FONTS['large']
        subtitle_font = pygame.font.SysFont(font_name, font_size)
        
        # Left title
        left_title_surface = subtitle_font.render(self.left_title, True, COLORS['text'])
        left_title_x = self.left_panel.centerx - left_title_surface.get_width() // 2
        left_title_y = self.left_panel.top + SIZES['padding']
        surface.blit(left_title_surface, (left_title_x, left_title_y))
        
        # Right title
        right_title_surface = subtitle_font.render(self.right_title, True, COLORS['text'])
        right_title_x = self.right_panel.centerx - right_title_surface.get_width() // 2
        right_title_y = self.right_panel.top + SIZES['padding']
        surface.blit(right_title_surface, (right_title_x, right_title_y))
        
        # Draw device renderers
        self.device_renderer_left.draw(surface)
        self.device_renderer_right.draw(surface)
        
        # Draw timeline
        self.timeline.draw(surface)
        
        # Draw play/pause button
        self.play_button.draw(surface)
        
        # Draw selection buttons
        self.left_selection_button.draw(surface)
        self.right_selection_button.draw(surface)
        
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
            
        # Handle selection buttons
        if self.left_selection_button.handle_event(event):
            return True
            
        if self.right_selection_button.handle_event(event):
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
        
    def _select_left(self):
        """Select the left signal as the best"""
        if self.selection_callback:
            self.selection_callback("left")
        else:
            print("Selected the left signal as best")
            
    def _select_right(self):
        """Select the right signal as the best"""
        if self.selection_callback:
            self.selection_callback("right")
        else:
            print("Selected the right signal as best") 