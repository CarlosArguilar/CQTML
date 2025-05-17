"""
UI theme and styling constants
"""

# Colors (RGB)
COLORS = {
    # Main colors
    'background': (30, 30, 40),
    'dark_panel': (20, 20, 30),
    'light_panel': (50, 50, 60),
    
    # Interactive elements
    'button': (70, 70, 90),
    'button_hover': (90, 90, 110),
    'button_active': (100, 100, 130),
    'button_disabled': (50, 50, 60),
    
    # Text
    'text': (220, 220, 230),
    'text_secondary': (170, 170, 180),
    'text_disabled': (100, 100, 110),
    
    # Accents
    'primary': (65, 105, 225),    # Royal blue
    'secondary': (106, 90, 205),  # Slate blue
    'highlight': (250, 218, 94),  # Gold yellow
    
    # Status
    'success': (50, 200, 120),
    'warning': (255, 180, 50),
    'error': (240, 70, 70),
    
    # Device circles (default colors)
    'device_base': [(65, 105, 225),   # Blue
                    (106, 90, 205),   # Purple
                    (220, 20, 60),    # Crimson
                    (255, 140, 0)],   # Dark Orange
}

# Sizes and measurements
SIZES = {
    'window_width': 1024,
    'window_height': 768,
    
    # UI element sizes
    'button_height': 40,
    'button_width': 120,
    'button_radius': 5,
    'slider_height': 20,
    'slider_button_radius': 10,
    'padding': 15,
    'margin': 10,
    
    # Device circles
    'device_circle_radius': 50,
    'device_circle_spacing': 120,
    'device_circle_y': 300,  # Vertical position
    
    # Timeline
    'timeline_height': 30,
    'timeline_marker_width': 4,
    'timeline_marker_height': 20,
}

# Fonts
FONTS = {
    'small': ('Arial', 14),
    'medium': ('Arial', 18),
    'large': ('Arial', 24),
    'button': ('Arial', 16),
    'title': ('Arial', 28, True),  # Bold
} 