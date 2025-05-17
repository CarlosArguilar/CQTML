from freemusic import FreeMusic

# Create dataset
dataset = FreeMusic(window=16384, normalize=True)

# Get a sample
audio = dataset[0]