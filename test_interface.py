from cqtml_interface import visualize_signal

import torch
import librosa

# Load first 10 seconds of audio at 22.5kHz as mono
audio, sr = librosa.load("/home/std/Music/Don't You Worry Child (Radio Edit).wav", 
                        sr=22500, 
                        duration=10.0,
                        mono=True)

# Convert to torch tensor
audio_tensor = torch.from_numpy(audio).float()
signal_tensor = torch.tensor([[0,1,0,1,0,1,0,1,0,1]])

visualize_signal(audio_tensor, signal_tensor, sample_rate=22500, duration=10)