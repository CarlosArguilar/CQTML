import librosa
import librosa.display
import numpy as np
import torch
import warnings
from IPython.display import Audio
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt

class CQTProcessor:
    """A class for consistent Constant-Q Transform processing with ML integration.
    
    Features:
    - Parameter consistency between forward/inverse transforms
    - Automatic audio length standardization
    - Memory-efficient processing
    - Type checking and validation
    - GPU support for PyTorch tensors
    """
    
    def __init__(
        self,
        audio_path: Union[str, Path],
        sr: int = 22050,
        hop_length: int = 128,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        fmin: Optional[float] = 32.70,  # C1 note frequency
        max_duration: Optional[float] = None,  # in seconds
        padding_mode: str = 'end'  # 'end', 'beginning', or 'both'
    ):
        """Initialize CQT processor with audio file and processing parameters."""
        self._validate_inputs(sr, hop_length, n_bins, bins_per_octave, padding_mode)
        
        # Store parameters as private attributes with type hints
        self._sr: int = sr
        self._hop_length: int = hop_length
        self._n_bins: int = n_bins
        self._bins_per_octave: int = bins_per_octave
        self._fmin: Optional[float] = fmin
        self._max_duration: Optional[float] = max_duration
        self._padding_mode: str = padding_mode
        
        # Load and process audio
        self.audio_path = Path(audio_path)
        self._load_and_process_audio()
        self._compute_cqt()

    def _validate_inputs(self, sr, hop_length, n_bins, bins_per_octave, padding_mode):
        """Ensure parameters are valid before processing."""
        if hop_length <= 0:
            raise ValueError("Hop length must be positive integer")
        if n_bins % bins_per_octave != 0:
            warnings.warn("n_bins not divisible by bins_per_octave might cause unexpected results")
        if padding_mode not in ['end', 'beginning', 'both']:
            raise ValueError("Invalid padding mode. Choose 'end', 'beginning', or 'both'")

    def _load_and_process_audio(self):
        """Load audio with length standardization."""
        y, sr = librosa.load(self.audio_path, sr=self._sr)
        
        # Handle max duration and padding
        target_samples = int(self._sr * self._max_duration) if self._max_duration else None
        self.audio, self.original_length = self._standardize_length(y, target_samples)
        
    def _standardize_length(self, y: np.ndarray, target_samples: Optional[int]) -> tuple[np.ndarray, int]:
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
