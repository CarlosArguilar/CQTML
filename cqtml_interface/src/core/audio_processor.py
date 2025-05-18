import pygame
import torch
import numpy as np
import io
from scipy.io import wavfile


class AudioProcessor:
    """
    Handles audio processing and playback for the interface.
    Manages audio timing and synchronization with signal visualization.
    """
    
    def __init__(self, audio_tensor, sample_rate):
        """
        Initialize the audio processor.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor (mono)
            sample_rate (int): Audio sample rate in Hz
        """
        self.audio_tensor = audio_tensor
        self.sample_rate = sample_rate
        self.duration = len(audio_tensor) / sample_rate
        self.playing = False
        self.current_time = 0.0
        self.channel = None
        
        # Initialize pygame mixer if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, channels=1)
        
        # Convert audio tensor to pygame Sound object
        self._prepare_audio()
        
    def _prepare_audio(self):
        """Convert the torch tensor to a pygame Sound object"""
        # Normalize audio to int16 range for WAV format
        audio_np = self.audio_tensor.cpu().numpy()
        
        # Ensure we have a properly scaled float array between -1 and 1
        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()
        
        # Convert to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Create a WAV file in memory using scipy
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_int16)
        buffer.seek(0)
        
        # Load the WAV into pygame Sound object
        self.sound = pygame.mixer.Sound(buffer)
        
    def play(self, start_time=None):
        """
        Play the audio from the specified time.
        
        Args:
            start_time (float, optional): Start time in seconds
        """
        if start_time is not None:
            self.current_time = max(0, min(start_time, self.duration))
        
        # Stop any current playback
        if self.channel and self.channel.get_busy():
            self.channel.stop()
            
        # We need to re-create the audio from the current position
        if self.current_time > 0 and self.current_time < self.duration:
            # Calculate samples to skip
            start_sample = int(self.current_time * self.sample_rate)
            
            # Create a new audio tensor starting from the current position
            remaining_audio = self.audio_tensor[start_sample:]
            
            # Create a temporary audio processor with the remaining audio
            # First, normalize to int16 range
            audio_np = remaining_audio.cpu().numpy()
            if np.abs(audio_np).max() > 1.0:
                audio_np = audio_np / np.abs(audio_np).max()
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Create a WAV file in memory
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, audio_int16)
            buffer.seek(0)
            
            # Create a temporary sound for this playback
            temp_sound = pygame.mixer.Sound(buffer)
            
            # Play the temporary sound
            self.channel = temp_sound.play()
        else:
            # Playing from the beginning, use the original sound
            self.channel = self.sound.play()
        
        self.playing = True
        self.play_start_time = pygame.time.get_ticks()
        
    def pause(self):
        """Pause audio playback"""
        if self.playing and self.channel:
            # Stop the channel - pygame doesn't have a real pause for Sound
            self.channel.stop()
            self.playing = False
            
            # Update current time
            elapsed = (pygame.time.get_ticks() - self.play_start_time) / 1000.0
            self.current_time = min(self.current_time + elapsed, self.duration)
            
    def stop(self):
        """Stop audio playback"""
        if self.channel:
            self.channel.stop()
        self.playing = False
        self.current_time = 0.0
        
    def set_time(self, time_sec):
        """
        Set the current playback time.
        
        Args:
            time_sec (float): Time in seconds
        """
        self.current_time = max(0, min(time_sec, self.duration))
        
        if self.playing:
            # If already playing, restart from new position
            was_playing = True
            self.pause()
        else:
            was_playing = False
            
        # Resume if it was playing
        if was_playing:
            self.play()
            
    def get_current_time(self):
        """
        Get the current playback time.
        
        Returns:
            float: Current time in seconds
        """
        if self.playing:
            # Check if channel exists and is still busy
            if self.channel and self.channel.get_busy():
                elapsed = (pygame.time.get_ticks() - self.play_start_time) / 1000.0
                return min(self.current_time + elapsed, self.duration)
            else:
                # Channel is no longer busy but we still think we're playing
                # This means audio has finished
                self.playing = False
                return self.current_time
        else:
            return self.current_time
            
    def get_progress(self):
        """
        Get current playback progress as a ratio.
        
        Returns:
            float: Progress between 0.0 and 1.0
        """
        return self.get_current_time() / self.duration if self.duration > 0 else 0.0 