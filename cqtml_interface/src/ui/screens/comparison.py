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
        self.signal_processor_top = None
        self.signal_processor_bottom = None
        self.audio_processor = None
        self.playing = False
        self.selection_callback = None
        
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
        
        # Timeline (shared between both visualizations)
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
        
        # Calculate panel dimensions - vertically stacked panels
        panel_width = int(self.screen_width * 0.8)  # 80% of screen width
        available_height = timeline_y - SIZES['content_padding_top'] - SIZES['padding'] * 3
        panel_height = available_height // 2  # Split available height in half
        
        # Top panel (Signal A)
        top_panel_x = (self.screen_width - panel_width) // 2  # Centered horizontally
        top_panel_y = SIZES['content_padding_top']
        self.top_panel = pygame.Rect(top_panel_x, top_panel_y, panel_width, panel_height)
        
        # Bottom panel (Signal B)
        bottom_panel_x = top_panel_x  # Aligned with top panel
        bottom_panel_y = top_panel_y + panel_height + SIZES['padding']
        self.bottom_panel = pygame.Rect(bottom_panel_x, bottom_panel_y, panel_width, panel_height)
        
        # Default number of devices
        n_default_devices = 4
        
        # Calculate device width for proper centering
        devices_width = (n_default_devices - 1) * SIZES['device_circle_spacing']
        
        # Top devices - center in the top panel
        top_devices_x = top_panel_x + (panel_width - devices_width) // 2
        top_devices_y = top_panel_y + panel_height // 2
        self.device_renderer_top = DeviceRenderer(
            top_devices_x,
            top_devices_y,
            n_default_devices,
            horizontal=True
        )
        
        # Bottom devices - center in the bottom panel
        bottom_devices_x = bottom_panel_x + (panel_width - devices_width) // 2
        bottom_devices_y = bottom_panel_y + panel_height // 2
        self.device_renderer_bottom = DeviceRenderer(
            bottom_devices_x,
            bottom_devices_y,
            n_default_devices,
            horizontal=True
        )
        
        # Selection buttons - make them smaller
        selection_button_width = 120
        selection_button_height = 30
        
        # Position buttons in the upper right of each panel
        top_selection_x = top_panel_x + panel_width - selection_button_width - SIZES['padding']
        top_selection_y = top_panel_y + SIZES['padding']
        
        bottom_selection_x = bottom_panel_x + panel_width - selection_button_width - SIZES['padding']
        bottom_selection_y = bottom_panel_y + SIZES['padding']
        
        # Selection buttons with more general names
        self.top_selection_button = Button(
            top_selection_x,
            top_selection_y,
            selection_button_width,
            selection_button_height,
            text="Select Signal A",
            callback=self._select_top,
            style="success"
        )
        
        self.bottom_selection_button = Button(
            bottom_selection_x,
            bottom_selection_y,
            selection_button_width,
            selection_button_height,
            text="Select Signal B",
            callback=self._select_bottom,
            style="success"
        )
        
        # Titles
        self.top_title = "Signal A"
        self.bottom_title = "Signal B"
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
        self.signal_processor_top = SignalProcessor(signal_tensor_left, duration, max_value)
        self.signal_processor_bottom = SignalProcessor(signal_tensor_right, duration, max_value)
        
        # Create audio processor
        self.audio_processor = AudioProcessor(audio_tensor, sample_rate)
        
        # Synchronize durations
        self.signal_processor_top.synchronize_with_audio(audio_tensor, sample_rate)
        self.signal_processor_bottom.synchronize_with_audio(audio_tensor, sample_rate)
        
        # Update timeline duration
        self.timeline.set_duration(self.audio_processor.duration)
        
        # Update device renderers
        self.device_renderer_top.set_signal_processor(self.signal_processor_top)
        self.device_renderer_bottom.set_signal_processor(self.signal_processor_bottom)
        
        # Recalculate positions for device renderers to ensure they're centered
        # Top panel
        n_devices_top = self.signal_processor_top.n_devices
        devices_width_top = (n_devices_top - 1) * SIZES['device_circle_spacing']
        top_devices_x = self.top_panel.x + (self.top_panel.width - devices_width_top) // 2
        self.device_renderer_top.set_position(top_devices_x, self.top_panel.y + self.top_panel.height // 2)
        
        # Bottom panel
        n_devices_bottom = self.signal_processor_bottom.n_devices
        devices_width_bottom = (n_devices_bottom - 1) * SIZES['device_circle_spacing']
        bottom_devices_x = self.bottom_panel.x + (self.bottom_panel.width - devices_width_bottom) // 2
        self.device_renderer_bottom.set_position(bottom_devices_x, self.bottom_panel.y + self.bottom_panel.height // 2)
        
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
        if self.signal_processor_top:
            self.device_renderer_top.update_values(current_time)
        if self.signal_processor_bottom:
            self.device_renderer_bottom.update_values(current_time)
        
    def draw(self, surface):
        """
        Draw the screen on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # Clear the screen
        surface.fill(COLORS['background'])
        
        # Draw panels with rounded corners
        pygame.draw.rect(surface, COLORS['dark_panel'], self.top_panel, border_radius=10)
        pygame.draw.rect(surface, COLORS['dark_panel'], self.bottom_panel, border_radius=10)
        
        # Draw main title
        font_name, font_size, bold = FONTS['title']
        title_font = pygame.font.SysFont(font_name, font_size, bold)
        title_surface = title_font.render(self.main_title, True, COLORS['text'])
        title_x = (self.screen_width - title_surface.get_width()) // 2
        title_y = SIZES['title_padding_top']
        surface.blit(title_surface, (title_x, title_y))
        
        # Draw panel titles
        font_name, font_size = FONTS['large']
        subtitle_font = pygame.font.SysFont(font_name, font_size)
        
        # Top title
        top_title_surface = subtitle_font.render(self.top_title, True, COLORS['text'])
        top_title_x = self.top_panel.centerx - top_title_surface.get_width() // 2
        top_title_y = self.top_panel.top + SIZES['padding']
        surface.blit(top_title_surface, (top_title_x, top_title_y))
        
        # Bottom title
        bottom_title_surface = subtitle_font.render(self.bottom_title, True, COLORS['text'])
        bottom_title_x = self.bottom_panel.centerx - bottom_title_surface.get_width() // 2
        bottom_title_y = self.bottom_panel.top + SIZES['padding']
        surface.blit(bottom_title_surface, (bottom_title_x, bottom_title_y))
        
        # Draw device renderers
        self.device_renderer_top.draw(surface)
        self.device_renderer_bottom.draw(surface)
        
        # Draw timeline
        self.timeline.draw(surface)
        
        # Draw play/pause button
        self.play_button.draw(surface)
        
        # Draw selection buttons
        self.top_selection_button.draw(surface)
        self.bottom_selection_button.draw(surface)
        
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
        if self.top_selection_button.handle_event(event):
            return True
            
        if self.bottom_selection_button.handle_event(event):
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
        
    def _select_top(self):
        """Select the top signal as the best"""
        if self.selection_callback:
            self.selection_callback("top")
        else:
            print("Selected Signal A as best")
            
    def _select_bottom(self):
        """Select the bottom signal as the best"""
        if self.selection_callback:
            self.selection_callback("bottom")
        else:
            print("Selected Signal B as best") 