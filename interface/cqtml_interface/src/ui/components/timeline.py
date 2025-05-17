import pygame
from ...ui.theme import COLORS, SIZES, FONTS


class Timeline:
    """
    A timeline/seekbar component for audio and signal visualization.
    """
    
    def __init__(self, x, y, width, duration, on_seek=None):
        """
        Initialize a timeline component.
        
        Args:
            x (int): X position
            y (int): Y position
            width (int): Timeline width
            duration (float): Total duration in seconds
            on_seek (function, optional): Callback for when the timeline is seeked
        """
        self.x = x
        self.y = y
        self.width = width
        self.duration = duration
        self.on_seek = on_seek
        self.height = SIZES['timeline_height']
        
        # Timeline state
        self.progress = 0.0  # 0.0 to 1.0
        self.time_text = "0:00 / 0:00"
        self.dragging = False
        
        # Generate the timeline surface
        self.update_surface()
        
    def update_surface(self):
        """Update the timeline's appearance based on state"""
        # Create timeline surface
        self.surface = pygame.Surface((self.width, self.height + 20), pygame.SRCALPHA)
        
        # Draw background
        bg_rect = pygame.Rect(0, 10, self.width, self.height)
        pygame.draw.rect(self.surface, COLORS['dark_panel'], bg_rect, border_radius=self.height // 2)
        
        # Draw progress bar
        progress_width = int(self.width * self.progress)
        if progress_width > 0:
            progress_rect = pygame.Rect(0, 10, progress_width, self.height)
            pygame.draw.rect(self.surface, COLORS['primary'], progress_rect, 
                             border_radius=self.height // 2)
        
        # Draw marker (draggable handle)
        marker_x = int(self.width * self.progress)
        marker_width = SIZES['timeline_marker_width']
        marker_height = SIZES['timeline_marker_height']
        marker_y = 10 + (self.height - marker_height) // 2
        
        marker_rect = pygame.Rect(marker_x - marker_width // 2, marker_y, 
                                  marker_width, marker_height)
        pygame.draw.rect(self.surface, COLORS['highlight'], marker_rect, 
                         border_radius=marker_width // 2)
        
        # Draw time text
        font_name, font_size = FONTS['small']
        font = pygame.font.SysFont(font_name, font_size)
        time_surf = font.render(self.time_text, True, COLORS['text'])
        
        # Position text below the timeline
        time_x = (self.width - time_surf.get_width()) // 2
        time_y = self.height + 12
        self.surface.blit(time_surf, (time_x, time_y))
        
    def update_progress(self, progress, current_time=None, total_time=None):
        """
        Update the timeline's progress.
        
        Args:
            progress (float): Progress value between 0.0 and 1.0
            current_time (float, optional): Current time in seconds
            total_time (float, optional): Total time in seconds (if different from self.duration)
        """
        # Update progress value
        self.progress = max(0.0, min(1.0, progress))
        
        # Update time text if times are provided
        if current_time is not None:
            if total_time is None:
                total_time = self.duration
                
            # Format times as MM:SS
            current_min, current_sec = divmod(int(current_time), 60)
            total_min, total_sec = divmod(int(total_time), 60)
            
            self.time_text = f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}"
            
        # Update the surface
        self.update_surface()
        
    def handle_event(self, event):
        """
        Handle pygame events for the timeline.
        
        Args:
            event (pygame.event.Event): Pygame event to handle
            
        Returns:
            bool: True if the timeline was seeked
        """
        seeked = False
        pos = pygame.mouse.get_pos()
        local_x = pos[0] - self.x
        
        # Check if mouse is over timeline
        if (self.x <= pos[0] <= self.x + self.width and 
            self.y <= pos[1] <= self.y + self.height + 20):
            
            # Handle mouse press
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
                
                # Update progress immediately
                new_progress = max(0.0, min(1.0, local_x / self.width))
                if new_progress != self.progress:
                    self.progress = new_progress
                    self.update_surface()
                    seeked = True
                    
                    if self.on_seek:
                        self.on_seek(self.progress * self.duration)
                        
        # Handle mouse release (anywhere)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            
        # Handle dragging
        if event.type == pygame.MOUSEMOTION and self.dragging:
            new_progress = max(0.0, min(1.0, local_x / self.width))
            if new_progress != self.progress:
                self.progress = new_progress
                self.update_surface()
                seeked = True
                
                if self.on_seek:
                    self.on_seek(self.progress * self.duration)
                    
        return seeked
        
    def draw(self, surface):
        """
        Draw the timeline on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        surface.blit(self.surface, (self.x, self.y))
        
    def set_position(self, x, y):
        """
        Set the timeline's position.
        
        Args:
            x (int): X position
            y (int): Y position
        """
        self.x = x
        self.y = y
        
    def set_duration(self, duration):
        """
        Set the timeline's total duration.
        
        Args:
            duration (float): Duration in seconds
        """
        self.duration = duration
        self.update_surface() 