import pygame
import numpy as np
from ...ui.theme import COLORS, SIZES


class DeviceRenderer:
    """
    Renders device signals as animated circles with varying intensity.
    """
    
    def __init__(self, x, y, n_devices, signal_processor=None, horizontal=True):
        """
        Initialize a device renderer.
        
        Args:
            x (int): X position of the first device
            y (int): Y position of the first device
            n_devices (int): Number of devices to render
            signal_processor (SignalProcessor, optional): Signal processor for getting values
            horizontal (bool, optional): Whether to layout devices horizontally (True) or vertically (False)
        """
        self.x = x
        self.y = y
        self.n_devices = n_devices
        self.signal_processor = signal_processor
        self.horizontal = horizontal
        
        # Device properties
        self.radius = SIZES['device_circle_radius']
        self.spacing = SIZES['device_circle_spacing']
        
        # Color for each device (use defaults or random)
        self.colors = []
        for i in range(n_devices):
            if i < len(COLORS['device_base']):
                self.colors.append(COLORS['device_base'][i])
            else:
                # Generate a random color for additional devices
                r = np.random.randint(100, 230)
                g = np.random.randint(100, 230)
                b = np.random.randint(100, 230)
                self.colors.append((r, g, b))
        
        # Current values for each device [0, 1]
        self.values = [0.0] * n_devices
        
        # Generate device surfaces
        self.surfaces = [None] * n_devices
        self.update_surfaces()
        
    def update_surfaces(self):
        """Update all device surfaces based on current values"""
        for i in range(self.n_devices):
            self._update_device_surface(i)
            
    def _update_device_surface(self, device_idx):
        """Update the surface for a specific device"""
        # Create surface for device with padding for glow effect
        padding = int(self.radius * 0.5)
        size = (self.radius + padding) * 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Get base color for this device
        base_color = self.colors[device_idx]
        
        # Draw outer glow (if value > 0)
        if self.values[device_idx] > 0:
            # Calculate glow radius and alpha
            glow_radius = self.radius + padding * self.values[device_idx]
            glow_alpha = int(100 * self.values[device_idx])
            
            # Create a glow color with alpha
            glow_color = (*base_color, glow_alpha)
            
            # Draw glow circle
            pygame.draw.circle(
                surface,
                glow_color,
                (size // 2, size // 2),
                glow_radius
            )
            
        # Draw device base
        pygame.draw.circle(
            surface,
            base_color,
            (size // 2, size // 2),
            self.radius
        )
        
        # Draw inner circle with varying brightness
        inner_color = self._get_brightness_color(base_color, self.values[device_idx])
        inner_radius = int(self.radius * 0.8)
        
        pygame.draw.circle(
            surface,
            inner_color,
            (size // 2, size // 2),
            inner_radius
        )
        
        # Store the surface
        self.surfaces[device_idx] = surface
        
    def _get_brightness_color(self, base_color, value):
        """
        Get a color with adjusted brightness based on value.
        
        Args:
            base_color (tuple): Base RGB color
            value (float): Value between 0 and 1
            
        Returns:
            tuple: Adjusted RGB color
        """
        # For low values, darken the color
        if value < 0.3:
            factor = 0.3 + 0.7 * value / 0.3  # 0.3 to 1.0
            return tuple(int(c * factor) for c in base_color)
        
        # For high values, make it brighter (towards white)
        else:
            factor = (value - 0.3) / 0.7  # 0 to 1.0
            return tuple(int(c + (255 - c) * factor) for c in base_color)
            
    def update_values(self, current_time):
        """
        Update device values based on the current time.
        
        Args:
            current_time (float): Current time in seconds
        """
        if self.signal_processor is None:
            return
            
        # Get values for each device at the current time
        for i in range(self.n_devices):
            try:
                # Get value from signal processor
                rgb, alpha = self.signal_processor.get_value_at_time(i, current_time)
                self.values[i] = alpha
                
                # If we have RGB values, update the device color
                if rgb is not None and any(c > 0 for c in rgb):
                    self.colors[i] = tuple(int(c * 255) for c in rgb)
            except Exception as e:
                print(f"Error updating device {i}: {e}")
                self.values[i] = 0.0
                
        # Update device surfaces
        self.update_surfaces()
        
    def draw(self, surface):
        """
        Draw all devices on the given surface.
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        for i in range(self.n_devices):
            # Calculate position for this device
            if self.horizontal:
                pos_x = self.x + i * self.spacing
                pos_y = self.y
            else:
                pos_x = self.x
                pos_y = self.y + i * self.spacing
                
            # Get the size of the device surface
            dev_surface = self.surfaces[i]
            size = dev_surface.get_width()
            
            # Calculate position to center the device surface
            draw_x = pos_x - size // 2
            draw_y = pos_y - size // 2
            
            # Draw the device
            surface.blit(dev_surface, (draw_x, draw_y))
            
    def set_position(self, x, y):
        """
        Set the position of the device renderer.
        
        Args:
            x (int): X position
            y (int): Y position
        """
        self.x = x
        self.y = y
        
    def set_signal_processor(self, signal_processor):
        """
        Set the signal processor for the device renderer.
        
        Args:
            signal_processor (SignalProcessor): Signal processor to use
        """
        self.signal_processor = signal_processor
        
        # Update number of devices if needed
        if signal_processor is not None:
            if self.n_devices != signal_processor.n_devices:
                self.n_devices = signal_processor.n_devices
                self.values = [0.0] * self.n_devices
                
                # Update colors if needed
                while len(self.colors) < self.n_devices:
                    if len(self.colors) < len(COLORS['device_base']):
                        self.colors.append(COLORS['device_base'][len(self.colors)])
                    else:
                        # Generate a random color
                        r = np.random.randint(100, 230)
                        g = np.random.randint(100, 230)
                        b = np.random.randint(100, 230)
                        self.colors.append((r, g, b))
                        
                # Update surfaces
                self.surfaces = [None] * self.n_devices
                self.update_surfaces() 