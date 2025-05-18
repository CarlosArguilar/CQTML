import pygame
from ...ui.theme import COLORS, SIZES, FONTS


class Button:
    """
    A reusable button component with different states and callback functionality.
    """
    
    def __init__(self, x, y, width=None, height=None, text="", callback=None, 
                 toggle=False, icon=None, style="default"):
        """
        Initialize a button.
        
        Args:
            x (int): X position
            y (int): Y position
            width (int, optional): Button width (default from theme)
            height (int, optional): Button height (default from theme)
            text (str, optional): Button text
            callback (function, optional): Function to call when clicked
            toggle (bool, optional): Whether the button toggles on/off
            icon (pygame.Surface, optional): Icon to display
            style (str, optional): Button style ('default', 'primary', 'success', etc.)
        """
        self.x = x
        self.y = y
        self.width = width if width is not None else SIZES['button_width']
        self.height = height if height is not None else SIZES['button_height']
        self.text = text
        self.callback = callback
        self.toggle = toggle
        self.toggled = False
        self.icon = icon
        self.style = style
        
        # Button states
        self.hovered = False
        self.clicked = False
        self.enabled = True
        
        # Colors based on style
        self.update_colors()
        
        # Generate the button surface
        self.update_surface()
        
    def update_colors(self):
        """Update button colors based on style and state"""
        if self.style == "primary":
            self.bg_color = COLORS['primary']
            self.hover_color = tuple(min(c + 20, 255) for c in COLORS['primary'])
            self.active_color = tuple(min(c + 40, 255) for c in COLORS['primary'])
            self.text_color = COLORS['text']
        elif self.style == "success":
            self.bg_color = COLORS['success']
            self.hover_color = tuple(min(c + 20, 255) for c in COLORS['success'])
            self.active_color = tuple(min(c + 40, 255) for c in COLORS['success'])
            self.text_color = COLORS['text']
        else:  # default
            self.bg_color = COLORS['button']
            self.hover_color = COLORS['button_hover']
            self.active_color = COLORS['button_active']
            self.text_color = COLORS['text']
            
        # Disabled state
        self.disabled_color = COLORS['button_disabled']
        self.disabled_text_color = COLORS['text_disabled']
        
    def update_surface(self):
        """Update the button's appearance based on state"""
        # Determine current color based on state
        if not self.enabled:
            color = self.disabled_color
            text_color = self.disabled_text_color
        elif self.clicked or self.toggled:
            color = self.active_color
            text_color = self.text_color
        elif self.hovered:
            color = self.hover_color
            text_color = self.text_color
        else:
            color = self.bg_color
            text_color = self.text_color
        
        # Create button surface
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw rounded rectangle
        radius = SIZES['button_radius']
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, color, rect, border_radius=radius)
        
        # Setup text
        font_name, font_size = FONTS['button']
        font = pygame.font.SysFont(font_name, font_size)
        text_surf = font.render(self.text, True, text_color)
        
        # Calculate text position
        text_x = (self.width - text_surf.get_width()) // 2
        text_y = (self.height - text_surf.get_height()) // 2
        
        # If there's an icon, adjust positions
        if self.icon:
            # Scale icon to fit
            icon_size = min(self.height - 10, 24)  # Max icon size
            icon = pygame.transform.scale(self.icon, (icon_size, icon_size))
            
            # Position icon and text
            if self.text:
                # Icon on left, text on right
                icon_x = 10
                text_x = icon_x + icon_size + 5
                
                # Center both
                total_width = icon_size + 5 + text_surf.get_width()
                offset = (self.width - total_width) // 2
                icon_x += offset
                text_x += offset
            else:
                # Only icon, center it
                icon_x = (self.width - icon_size) // 2
                
            icon_y = (self.height - icon_size) // 2
            self.surface.blit(icon, (icon_x, icon_y))
            
        # Draw text if any
        if self.text:
            self.surface.blit(text_surf, (text_x, text_y))
            
    def handle_event(self, event):
        """
        Handle pygame events for the button.
        
        Args:
            event (pygame.event.Event): Pygame event to handle
            
        Returns:
            bool: True if the button was clicked
        """
        if not self.enabled:
            return False
            
        pos = pygame.mouse.get_pos()
        clicked = False
        
        # Check if mouse is over button
        if (self.x <= pos[0] <= self.x + self.width and 
            self.y <= pos[1] <= self.y + self.height):
            
            if not self.hovered:
                self.hovered = True
                self.update_surface()
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.clicked = True
                self.update_surface()
                
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.clicked:
                    clicked = True
                    if self.toggle:
                        self.toggled = not self.toggled
                    if self.callback:
                        self.callback()
                self.clicked = False
                self.update_surface()
        else:
            if self.hovered:
                self.hovered = False
                self.update_surface()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.clicked = False
                self.update_surface()
                
        return clicked
        
    def draw(self, surface):
        """
        Draw the button on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        surface.blit(self.surface, (self.x, self.y))
        
    def set_position(self, x, y):
        """
        Set the button's position.
        
        Args:
            x (int): X position
            y (int): Y position
        """
        self.x = x
        self.y = y
        
    def set_enabled(self, enabled):
        """
        Enable or disable the button.
        
        Args:
            enabled (bool): Whether the button is enabled
        """
        if self.enabled != enabled:
            self.enabled = enabled
            self.update_surface()
            
    def set_text(self, text):
        """
        Set the button's text.
        
        Args:
            text (str): New button text
        """
        if self.text != text:
            self.text = text
            self.update_surface()
            
    def set_toggled(self, toggled):
        """
        Set the toggle state of the button.
        
        Args:
            toggled (bool): Whether the button is toggled
        """
        if self.toggled != toggled:
            self.toggled = toggled
            self.update_surface() 