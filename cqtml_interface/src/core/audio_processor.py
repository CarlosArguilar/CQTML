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
    
    def __init__(self, audio_tensor, sample_rate, volume_boost=3.0):
        """
        Initialize the audio processor.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor (mono)
            sample_rate (int): Audio sample rate in Hz
            volume_boost (float): Volume amplification factor (default: 3.0)
        """
        self.audio_tensor = audio_tensor
        self.sample_rate = sample_rate
        self.duration = len(audio_tensor) / sample_rate
        self.playing = False
        self.current_time = 0.0
        self.channel = None
        self.volume_boost = volume_boost
        self.runtime_volume = 1.0  # Runtime volume multiplier (0.0 to 10.0+)
        self.last_prepared_volume = 1.0  # Track last volume used for audio preparation
        
        # Initialize pygame mixer if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, channels=1)
        
        # Set global volume to maximum
        pygame.mixer.music.set_volume(1.0)
        
        # Prepare the base audio data (normalized without runtime volume)
        self._prepare_base_audio()
        
        # Convert audio tensor to pygame Sound object
        self._prepare_audio()
        
    def _prepare_base_audio(self):
        """Prepare the base audio data without runtime volume"""
        # Convert tensor to numpy
        audio_np = self.audio_tensor.cpu().numpy()
        
        # Apply initial volume boost
        audio_np = audio_np * self.volume_boost
        
        # Normalize to prevent clipping but preserve the boosted signal
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val
        
        # Store the normalized base audio for runtime volume application
        self.base_audio = audio_np
        
    def _prepare_audio(self):
        """Convert the audio to a pygame Sound object with current runtime volume"""
        # Apply runtime volume to base audio
        final_audio = self.base_audio * self.runtime_volume
        
        # Clip to prevent distortion
        final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (final_audio * 32767).astype(np.int16)
        
        # Create a WAV file in memory using scipy
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_int16)
        buffer.seek(0)
        
        # Load the WAV into pygame Sound object
        self.sound = pygame.mixer.Sound(buffer)
        
        # Set pygame volume to maximum since we handle volume in the audio data
        self.sound.set_volume(1.0)
        
        # Update the tracked volume
        self.last_prepared_volume = self.runtime_volume
        
    def _create_audio_from_position(self, start_sample):
        """Create audio data starting from a specific sample position"""
        # Get audio data from start position
        remaining_audio = self.base_audio[start_sample:]
        
        # Apply runtime volume
        final_audio = remaining_audio * self.runtime_volume
        
        # Clip to prevent distortion
        final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (final_audio * 32767).astype(np.int16)
        
        # Create a WAV file in memory
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_int16)
        buffer.seek(0)
        
        # Create and return a temporary sound
        temp_sound = pygame.mixer.Sound(buffer)
        temp_sound.set_volume(1.0)
        return temp_sound
        
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
            
        # Check if we need to regenerate audio due to volume change
        if abs(self.runtime_volume - self.last_prepared_volume) > 0.01:
            self._prepare_audio()
            
        # We need to re-create the audio from the current position
        if self.current_time > 0 and self.current_time < self.duration:
            # Calculate samples to skip
            start_sample = int(self.current_time * self.sample_rate)
            
            # Create audio from current position with current volume
            temp_sound = self._create_audio_from_position(start_sample)
            
            # Play the temporary sound
            self.channel = temp_sound.play()
        else:
            # Playing from the beginning, use the prepared sound
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
    
    def set_runtime_volume(self, volume):
        """
        Set the runtime volume multiplier.
        
        Args:
            volume (float): Volume multiplier (0.0 = mute, 1.0 = normal, 10.0 = 10x louder)
        """
        old_volume = self.runtime_volume
        self.runtime_volume = max(0.0, volume)
        
        # If volume changed significantly and we're playing, restart playback with new volume
        if abs(self.runtime_volume - old_volume) > 0.01 and self.playing:
            current_time = self.get_current_time()
            self.pause()
            self.current_time = current_time
            self.play()
    
    def get_runtime_volume(self):
        """Get the current runtime volume multiplier."""
        return self.runtime_volume 