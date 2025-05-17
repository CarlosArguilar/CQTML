from freemusic import FreeMusic

# Create dataset
dataset = FreeMusic(window=1e6, normalize=True)

# Get a sample
audio = dataset[0]