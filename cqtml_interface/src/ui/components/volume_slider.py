import pygame
import math
from ..theme import COLORS, SIZES, FONTS


class VolumeSlider:
    """
    Volume slider component for controlling audio volume in real-time.
    """
    
    def __init__(self, x, y, width=120, height=20, initial_volume=10.0, min_volume=0.0, max_volume=100.0):
        """
        Initialize the volume slider.
        
        Args:
            x (int): X position
            y (int): Y position  
            width (int): Slider width
            height (int): Slider height
            initial_volume (float): Initial volume value
            min_volume (float): Minimum volume value
            max_volume (float): Maximum volume value
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.width = width
        self.height = height
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.volume = initial_volume
        self.dragging = False
        self.hover = False
        
        # Visual properties
        self.track_color = COLORS['button']
        self.track_hover_color = COLORS['button_hover']
        self.handle_color = COLORS['primary']
        self.handle_hover_color = COLORS['highlight']  # Use highlight instead of primary_light
        self.text_color = COLORS['text']
        
        # Handle properties
        self.handle_radius = height // 2 + 2
        self.track_height = max(4, height // 3)
        
        # Fonts
        self.font = pygame.font.Font(None, 16)
        
    def get_handle_x(self):
        """Get the X position of the volume handle based on current volume."""
        volume_ratio = (self.volume - self.min_volume) / (self.max_volume - self.min_volume)
        return self.rect.x + int(volume_ratio * self.width)
    
    def set_volume(self, volume):
        """Set the volume value and clamp it to valid range."""
        self.volume = max(self.min_volume, min(self.max_volume, volume))
    
    def get_volume(self):
        """Get the current volume value."""
        return self.volume
    
    def handle_event(self, event):
        """
        Handle pygame events for the volume slider.
        
        Args:
            event (pygame.event.Event): Pygame event to handle
            
        Returns:
            bool: True if the event was handled, False otherwise
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_x, mouse_y = event.pos
                
                # Check if clicking on handle or track
                handle_x = self.get_handle_x()
                handle_rect = pygame.Rect(
                    handle_x - self.handle_radius,
                    self.rect.centery - self.handle_radius,
                    self.handle_radius * 2,
                    self.handle_radius * 2
                )
                
                if handle_rect.collidepoint(mouse_x, mouse_y) or self.rect.collidepoint(mouse_x, mouse_y):
                    self.dragging = True
                    # Set volume based on click position
                    relative_x = mouse_x - self.rect.x
                    volume_ratio = max(0, min(1, relative_x / self.width))
                    self.volume = self.min_volume + volume_ratio * (self.max_volume - self.min_volume)
                    return True
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            
            # Update hover state
            handle_x = self.get_handle_x()
            handle_rect = pygame.Rect(
                handle_x - self.handle_radius,
                self.rect.centery - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2
            )
            
            self.hover = handle_rect.collidepoint(mouse_x, mouse_y) or self.rect.collidepoint(mouse_x, mouse_y)
            
            # Handle dragging
            if self.dragging:
                relative_x = mouse_x - self.rect.x
                volume_ratio = max(0, min(1, relative_x / self.width))
                self.volume = self.min_volume + volume_ratio * (self.max_volume - self.min_volume)
                return True
                
        return False
    
    def draw(self, surface):
        """
        Draw the volume slider.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # Draw track
        track_rect = pygame.Rect(
            self.rect.x,
            self.rect.centery - self.track_height // 2,
            self.width,
            self.track_height
        )
        
        track_color = self.track_hover_color if self.hover else self.track_color
        pygame.draw.rect(surface, track_color, track_rect, border_radius=self.track_height // 2)
        
        # Draw filled portion of track
        volume_ratio = (self.volume - self.min_volume) / (self.max_volume - self.min_volume)
        filled_width = int(volume_ratio * self.width)
        
        if filled_width > 0:
            filled_rect = pygame.Rect(
                self.rect.x,
                self.rect.centery - self.track_height // 2,
                filled_width,
                self.track_height
            )
            pygame.draw.rect(surface, self.handle_color, filled_rect, border_radius=self.track_height // 2)
        
        # Draw handle
        handle_x = self.get_handle_x()
        handle_y = self.rect.centery
        
        handle_color = self.handle_hover_color if (self.hover or self.dragging) else self.handle_color
        
        # Draw handle shadow
        pygame.draw.circle(surface, (0, 0, 0, 50), (handle_x + 1, handle_y + 1), self.handle_radius)
        
        # Draw handle
        pygame.draw.circle(surface, handle_color, (handle_x, handle_y), self.handle_radius)
        pygame.draw.circle(surface, COLORS['text'], (handle_x, handle_y), self.handle_radius, 2)
        
        # Draw volume text
        volume_text = f"Volume: {self.volume:.1f}x"
        text_surface = self.font.render(volume_text, True, self.text_color)
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.rect.centerx
        text_rect.bottom = self.rect.top - 5
        
        surface.blit(text_surface, text_rect)
        
        # Draw volume level indicators
        if self.volume >= 1.0:
            # Draw 🔊 for normal/high volume
            volume_icon = "🔊" if self.volume >= 1.5 else "🔉"
        elif self.volume >= 0.3:
            # Draw 🔉 for medium volume  
            volume_icon = "🔉"
        elif self.volume > 0.0:
            # Draw 🔈 for low volume
            volume_icon = "🔈"
        else:
            # Draw 🔇 for muted
            volume_icon = "🔇"
        
        # Try to render emoji, fallback to text
        try:
            icon_surface = self.font.render(volume_icon, True, self.text_color)
            icon_rect = icon_surface.get_rect()
            icon_rect.centery = self.rect.centery
            icon_rect.right = self.rect.left - 8
            surface.blit(icon_surface, icon_rect)
        except:
            # Fallback to simple text indicator
            if self.volume == 0.0:
                icon_text = "MUTE"
            elif self.volume < 0.5:
                icon_text = "LOW"
            elif self.volume < 1.5:
                icon_text = "MID"
            else:
                icon_text = "HIGH"
                
            icon_surface = self.font.render(icon_text, True, self.text_color)
            icon_rect = icon_surface.get_rect()
            icon_rect.centery = self.rect.centery
            icon_rect.right = self.rect.left - 8
            surface.blit(icon_surface, icon_rect) 