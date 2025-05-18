# CQTML Signal Visualizer Interface

A modular PyGame interface for visualizing signal data with audio playback in two modes: visualization and comparison.

## Overview

This interface allows visualization of signal data from multiple devices with synchronized audio playback. It's designed for signal processing applications where you need to visualize how devices (e.g., lights) respond over time according to signal values.

## Features

- **Visualization Mode**: Display a single signal with synchronized audio
- **Comparison Mode**: Compare two signals side-by-side with the same audio
- **Interactive Timeline**: Seek through the signal/audio with a draggable timeline
- **Play/Pause Control**: Control audio playback
- **Best Signal Selection**: In comparison mode, select which signal is preferred
- **Improved Visual Design**: Clean layout with proper spacing and alignment of UI elements

## Requirements

- Python 3.6+
- PyGame
- PyTorch
- NumPy
- SciPy

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/cqtml_interface.git
cd cqtml_interface
```

2. Install the required packages:
```
pip install pygame torch numpy scipy
```

## Usage

### Basic Usage

Run the application in visualization mode:
```
python main.py --mode visualization
```

Run the application in comparison mode:
```
python main.py --mode comparison
```

### Example Scripts

#### Visualization Mode Example
```
python examples/test_visualization.py
```

With colored signals:
```
python examples/test_visualization.py --color
```

#### Comparison Mode Example
```
python examples/test_comparison.py
```

With colored signals:
```
python examples/test_comparison.py --color
```

## Interface Controls

- **Play/Pause**: Click the play/pause button to control audio playback
- **Timeline**: Click or drag the timeline to seek through the signal/audio
- **Escape Key**: Press ESC to exit the application
- **Selection Buttons**: In comparison mode, click "Select Left" or "Select Right" to choose the best signal

## Visual Design

The interface has been designed with careful attention to:
- Proper alignment and spacing of UI elements
- Visual hierarchy to focus on important content
- Consistent color scheme for intuitive understanding
- Clear device visualization with 3D effect and glow
- Device labels for easy identification

## API Usage

To use the interface in your own Python code:

```python
import torch
from main import SignalVisualizerApp

# Create your signal and audio tensors
signal_tensor = torch.rand(4, 200)  # 4 devices, 200 timesteps
audio_tensor = torch.sin(torch.linspace(0, 8 * 3.14, 44100 * 5))  # 5 seconds audio

# Create and run the application
app = SignalVisualizerApp()
app.set_visualization_mode(signal_tensor, audio_tensor, sample_rate=44100)
app.run()
```

For comparison mode:

```python
app.set_comparison_mode(
    signal_tensor1,
    signal_tensor2, 
    audio_tensor,
    sample_rate=44100,
    selection_callback=my_callback_function
)
```

## Project Structure

```
cqtml_interface/
├── src/
│   ├── core/
│   │   ├── signal_processor.py     # Signal processing utilities
│   │   └── audio_processor.py      # Audio processing utilities
│   ├── ui/
│   │   ├── components/             # Reusable UI components
│   │   │   ├── button.py           # Button component
│   │   │   ├── timeline.py         # Timeline/seekbar component
│   │   │   └── device_renderer.py  # Renders the device visualization
│   │   ├── screens/                # Different screens/modes
│   │   │   ├── visualization.py    # Visualization mode screen
│   │   │   └── comparison.py       # Comparison mode screen
│   │   └── theme.py                # UI theme constants
│   └── utils/
│       └── helpers.py              # Helper functions
├── examples/
│   ├── test_visualization.py       # Example for visualization mode
│   └── test_comparison.py          # Example for comparison mode
└── main.py                         # Main entry point
```

## License

MIT 