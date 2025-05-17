# SignalVisualizer

A Python class for visualizing multi-device signals over time.

## Overview

The `SignalVisualizer` class provides a way to visualize signal data from multiple devices over time. It accepts PyTorch tensors with either 2 or 3 dimensions:

- 2D tensor: `[devices × timesteps]` - Simple signal intensity values
- 3D tensor: `[devices × timesteps × channels]` - RGBA color channels (with alpha representing intensity)

## Requirements

- Python 3.6+
- PyTorch
- Matplotlib
- NumPy

## Usage

```python
import torch
from signal_visualizer import SignalVisualizer

# Create or load your signals
signals = torch.rand(3, 100)  # 3 devices with 100 timesteps each

# Create a visualizer instance
visualizer = SignalVisualizer(signals, duration=10, max_value=1)

# Generate and display the visualization
fig, ax = visualizer.visualization()
fig.savefig("signals.png")  # Save to file
```

### Parameters

- `signals`: PyTorch tensor with signal data
- `duration`: Time duration in seconds that the signal represents (default: 10s)
- `max_value`: Maximum value for signal normalization (default: 1)

## Example

Run the test script to see examples:

```
python test_signal_visualizer.py
```

This will generate two example visualizations:
- 2D signals (grayscale intensity)
- 3D signals (RGBA channels)

The resulting images will be saved as PNG files in the current directory. 