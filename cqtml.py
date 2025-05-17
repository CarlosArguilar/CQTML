import librosa
import librosa.display
import numpy as np
import torch
import warnings
from IPython.display import Audio
from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt

class CQTProcessor:
    """A class for consistent Constant-Q Transform processing with ML integration.
    
    Features:
    - Parameter consistency between forward/inverse transforms
    - Automatic audio length standardization
    - Memory-efficient processing
    - Type checking and validation
    - GPU support for PyTorch tensors
    - Support for file paths, numpy arrays, and PyTorch tensors as input
    """
    
    def __init__(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sr: int = 22050,
        hop_length: int = 64,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        fmin: Optional[float] = 32.70,  # C1 note frequency
        max_duration: Optional[float] = None,  # in seconds
        padding_mode: str = 'end'  # 'end', 'beginning', or 'both'
    ):
        """Initialize CQT processor with audio input and processing parameters.
        
        Args:
            audio: Audio input as file path, numpy array, or PyTorch tensor
            sr: Sample rate of the audio
            hop_length: Number of samples between successive CQT columns
            n_bins: Number of frequency bins
            bins_per_octave: Number of bins per octave
            fmin: Minimum frequency
            max_duration: Maximum duration in seconds
            padding_mode: How to pad audio: 'end', 'beginning', or 'both'
        """
        self._validate_inputs(sr, hop_length, n_bins, bins_per_octave, padding_mode)
        
        # Store parameters as private attributes with type hints
        self._sr: int = sr
        self._hop_length: int = hop_length
        self._n_bins: int = n_bins
        self._bins_per_octave: int = bins_per_octave
        self._fmin: Optional[float] = fmin
        self._max_duration: Optional[float] = max_duration
        self._padding_mode: str = padding_mode
        
        # Process audio based on input type
        self._process_audio_input(audio)
        self._compute_cqt()

    @classmethod
    def from_cqt_tensor(cls, cqt_tensor: torch.Tensor, sr: int = 22050, 
                        hop_length: int = 64, n_bins: int = 84, 
                        bins_per_octave: int = 12, fmin: Optional[float] = 32.70):
        """Create a CQTProcessor instance from a pre-computed CQT tensor.
        
        Args:
            cqt_tensor: A PyTorch tensor with shape [2, n_bins, time] representing the CQT
                        where channel 0 is real part and channel 1 is imaginary part
            sr: Sample rate of the audio
            hop_length: Number of samples between successive CQT columns 
            n_bins: Number of frequency bins
            bins_per_octave: Number of bins per octave
            fmin: Minimum frequency
            
        Returns:
            A CQTProcessor instance with the pre-computed CQT.
        """
        # Validate input tensor shape
        if cqt_tensor.dim() != 3 or cqt_tensor.shape[0] != 2:
            raise ValueError("CQT tensor must have shape [2, n_bins, time]")
            
        if cqt_tensor.shape[1] != n_bins:
            warnings.warn(f"CQT tensor has {cqt_tensor.shape[1]} bins but n_bins={n_bins}")
        
        # Create an instance with placeholder audio
        # We create a small dummy audio array since we'll override the CQT values
        dummy_audio = np.zeros(1024, dtype=np.float32)
        instance = cls(
            dummy_audio, 
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin
        )
        
        # Extract real and imaginary parts from tensor
        if cqt_tensor.device != torch.device('cpu'):
            cqt_tensor = cqt_tensor.cpu()
            
        # Override the computed CQT with the provided tensor values
        instance._cqt_real = cqt_tensor[0].numpy()
        instance._cqt_imag = cqt_tensor[1].numpy()
        
        # Estimate the audio length from the CQT
        time_steps = cqt_tensor.shape[2]
        estimated_samples = time_steps * hop_length
        instance.original_length = estimated_samples
        
        return instance

    def _validate_inputs(self, sr, hop_length, n_bins, bins_per_octave, padding_mode):
        """Ensure parameters are valid before processing."""
        if hop_length <= 0:
            raise ValueError("Hop length must be positive integer")
        if n_bins % bins_per_octave != 0:
            warnings.warn("n_bins not divisible by bins_per_octave might cause unexpected results")
        if padding_mode not in ['end', 'beginning', 'both']:
            raise ValueError("Invalid padding mode. Choose 'end', 'beginning', or 'both'")

    def _process_audio_input(self, audio):
        """Process different audio input types."""
        if isinstance(audio, (str, Path)):
            # Store the path for reference
            self.audio_path = Path(audio)
            self._load_from_file()
        elif isinstance(audio, np.ndarray):
            # Process numpy array directly
            self.audio_path = None
            self._process_numpy_audio(audio)
        elif isinstance(audio, torch.Tensor):
            # Convert torch tensor to numpy and process
            self.audio_path = None
            self._process_tensor_audio(audio)
        else:
            raise TypeError("Audio input must be a file path, numpy array, or torch tensor")

    def _load_from_file(self):
        """Load audio from file path."""
        y, sr = librosa.load(self.audio_path, sr=self._sr)
        self._standardize_audio(y)
    
    def _process_numpy_audio(self, audio_array: np.ndarray):
        """Process audio from numpy array."""
        # Ensure it's 1D (mono) audio
        if audio_array.ndim > 1:
            warnings.warn("Multi-channel audio detected, converting to mono")
            audio_array = librosa.to_mono(audio_array)
        self._standardize_audio(audio_array)
    
    def _process_tensor_audio(self, audio_tensor: torch.Tensor):
        """Process audio from torch tensor."""
        # Convert to numpy
        audio_array = audio_tensor.cpu().numpy()
        # Handle different tensor shapes
        if audio_array.ndim > 1:
            # If 2D, assume it's [channels, samples]
            if audio_array.ndim == 2:
                audio_array = librosa.to_mono(audio_array)
            else:
                warnings.warn("Tensor with > 2 dimensions detected, flattening to 1D")
                audio_array = audio_array.flatten()
        self._standardize_audio(audio_array)
    
    def _standardize_audio(self, y: np.ndarray):
        """Standardize audio length."""
        target_samples = int(self._sr * self._max_duration) if self._max_duration else None
        self.audio, self.original_length = self._standardize_length(y, target_samples)

    def _standardize_length(self, y: np.ndarray, target_samples: Optional[int]) -> Tuple[np.ndarray, int]:
        """Trim or pad audio to target length."""
        if target_samples is None:
            return y, len(y)
        
        current_samples = len(y)
        if current_samples > target_samples:
            warnings.warn(f"Trimming audio from {current_samples/self._sr:.2f}s to {target_samples/self._sr:.2f}s")
            return y[:target_samples], target_samples
        
        if current_samples < target_samples:
            padding = target_samples - current_samples
            return self._apply_padding(y, padding), target_samples
        
        return y, current_samples

    def _apply_padding(self, y: np.ndarray, padding: int) -> np.ndarray:
        """Apply padding according to specified mode."""
        if self._padding_mode == 'end':
            return np.pad(y, (0, padding), mode='constant')
        elif self._padding_mode == 'beginning':
            return np.pad(y, (padding, 0), mode='constant')
        else:  # both
            left = padding // 2
            right = padding - left
            return np.pad(y, (left, right), mode='constant')

    def _compute_cqt(self):
        """Compute and store CQT representation."""
        cqt = librosa.cqt(
            self.audio,
            sr=self._sr,
            hop_length=self._hop_length,
            n_bins=self._n_bins,
            bins_per_octave=self._bins_per_octave,
            fmin=self._fmin
        )
        
        # Store as separate real/imaginary components
        self._cqt_real = np.real(cqt)
        self._cqt_imag = np.imag(cqt)

    @property
    def cqt_image(self) -> np.ndarray:
        """Get CQT representation as numpy array (channels last)."""
        return np.stack([self._cqt_real, self._cqt_imag], axis=-1)

    @property
    def time_steps(self) -> int:
        """Get number of time steps in the CQT image."""
        return self.cqt_image.shape[1]

    @property
    def freq_bins(self) -> int:
        """Get number of frequency bins."""
        return self._n_bins

    def export_to_pytorch(self, device: str = 'cpu') -> torch.Tensor:
        """Export CQT to PyTorch tensor with channel-first format.
        
        Args:
            device: Target device for the tensor ('cpu' or 'cuda')
        """
        tensor = torch.from_numpy(self.cqt_image).permute(2, 0, 1).float()
        return tensor.to(device)

    def plot(self):
        """Visualize the CQT magnitude spectrogram."""
        magnitude = np.sqrt(self._cqt_real**2 + self._cqt_imag**2)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            magnitude_db,
            sr=self._sr,
            hop_length=self._hop_length,
            x_axis='time',
            y_axis='cqt_hz',
            bins_per_octave=self._bins_per_octave,
            fmin=self._fmin
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Transform')
        plt.show()

    def reconstruct_audio(self) -> np.ndarray:
        """Reconstruct audio from CQT representation."""
        cqt_complex = self._cqt_real + 1j * self._cqt_imag
        reconstructed = librosa.icqt(
            cqt_complex,
            sr=self._sr,
            hop_length=self._hop_length,
            bins_per_octave=self._bins_per_octave,
            fmin=self._fmin
        )
        
        # Remove padding if needed
        if self._max_duration and len(reconstructed) > self.original_length:
            return reconstructed[:self.original_length]
        return reconstructed

    def play(self):
        """Play the reconstructed audio."""
        return Audio(self.reconstruct_audio(), rate=self._sr)

    # Property getters for parameters
    @property
    def sr(self) -> int:
        return self._sr

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def parameters(self) -> dict:
        """Get all processing parameters as a dictionary."""
        return {
            'sr': self._sr,
            'hop_length': self._hop_length,
            'n_bins': self._n_bins,
            'bins_per_octave': self._bins_per_octave,
            'fmin': self._fmin,
            'max_duration': self._max_duration,
            'padding_mode': self._padding_mode
        }
