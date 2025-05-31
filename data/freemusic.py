"""
Dataset class for the Free Music Archive dataset.
"""

from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import datasets
import librosa
from typing import List, Tuple, Dict, Optional, Callable, Union
import math
from tqdm import tqdm
import os
import pickle
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from core.cqtml import CQTProcessor

class FreeMusic(data.Dataset):
    """Free Music Archive Dataset.
    Args:
        sample_rate (int, optional): Target sample rate for audio.
        max_duration (float, optional): Maximum duration in seconds for each data point.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        epoch_size (int, optional): If not None, the dataset will have this many samples.
                                   If None, the dataset size will be the actual number of chunks.
        verbose (bool, optional): If true, show progress bar during initialization.
        cache_dir (str, optional): Directory to store cached indexing data.
        use_cache (bool, optional): Whether to use cached indexing data.
        force_rebuild (bool, optional): If true, force rebuilding the index even if cache exists.
        transform (callable, optional): Optional transform to be applied to audio chunks.
        output_format (str, optional): 'audio' for raw audio, 'cqt' for CQT transform.
        cqt_params (dict, optional): Parameters for CQT transform if output_format='cqt'.
        cache_cqt (bool, optional): Whether to cache CQT transforms.
    """

    def __init__(self, sample_rate=22050, max_duration=30.0, normalize=True, epoch_size=None, 
                 verbose=True, cache_dir='.cache/freemusic', use_cache=True, force_rebuild=False,
                 transform=None, output_format='audio', cqt_params=None, cache_cqt=True):
        self.normalize = normalize
        self.target_sr = sample_rate
        self.max_duration = max_duration
        self.samples_per_chunk = int(sample_rate * max_duration)
        self.epsilon = 10e-8  # fudge factor for normalization
        self.verbose = verbose
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        self.transform = transform
        self.output_format = output_format.lower()
        self.cache_cqt = cache_cqt
        
        # Validate output format
        if self.output_format not in ['audio', 'cqt']:
            raise ValueError("output_format must be either 'audio' or 'cqt'")
            
        # Set up CQT parameters - match defaults from CQTProcessor
        self.cqt_params = cqt_params or {}
        self.cqt_params.setdefault('hop_length', 64)  # Changed from 128 to 64 to match CQTProcessor
        self.cqt_params.setdefault('n_bins', 84)  
        self.cqt_params.setdefault('bins_per_octave', 12)
        self.cqt_params.setdefault('fmin', 32.70)  # C1 note
        
        # Create CQT cache if needed
        if self.output_format == 'cqt' and self.cache_cqt:
            # Set maximum number of cached items (adjust based on memory requirements)
            self._cqt_cache_size = 1000
            self._cqt_cache = {}
        
        # Load the dataset
        self.dataset = datasets.load_dataset("benjamin-paine/free-music-archive-small")
        self.train_dataset = self.dataset["train"]
        self.num_files = len(self.train_dataset)
        
        # Calculate chunk info or load from cache
        self._setup_chunk_index()
        
        # Set epoch size
        self.epoch_size = epoch_size if epoch_size is not None else self.total_chunks
    
    def _get_cache_path(self):
        """Generate a unique cache file path based on dataset parameters."""
        # Create a unique identifier based on crucial parameters
        params = {
            'dataset_name': 'benjamin-paine/free-music-archive-small',
            'num_files': self.num_files,  # To detect dataset changes
            'sample_rate': self.target_sr,
            'max_duration': self.max_duration,
        }
        
        # Create a hash of the parameters for the filename
        param_str = json.dumps(params, sort_keys=True)
        cache_id = hashlib.md5(param_str.encode()).hexdigest()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Return the full cache path
        return self.cache_dir / f"index_{cache_id}.pkl"
    
    def _load_cache(self):
        """Attempt to load cached indexing data."""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            if self.verbose:
                print(f"No cache found at {cache_path}")
            return False
        
        try:
            if self.verbose:
                print(f"Loading cached index from {cache_path}")
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Load data from cache
            self.chunks_per_file = cache_data['chunks_per_file']
            self.total_chunks = cache_data['total_chunks']
            self.chunk_to_file_map = cache_data['chunk_to_file_map']
            
            if self.verbose:
                print(f"Loaded cached index with {self.total_chunks} chunks")
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self):
        """Save indexing data to cache."""
        cache_path = self._get_cache_path()
        
        try:
            if self.verbose:
                print(f"Saving index cache to {cache_path}")
            
            cache_data = {
                'chunks_per_file': self.chunks_per_file,
                'total_chunks': self.total_chunks,
                'chunk_to_file_map': self.chunk_to_file_map
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            if self.verbose:
                print(f"Cache saved successfully")
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to save cache: {e}")
            return False
    
    def _setup_chunk_index(self):
        """Set up chunk indexing, using cache if available."""
        # Skip cache if force_rebuild is True
        if self.use_cache and not self.force_rebuild:
            # Try to load from cache
            if self._load_cache():
                return
        
        # If we get here, we need to build the index
        self._build_chunk_index()
        
        # Save to cache if requested
        if self.use_cache:
            self._save_cache()

    def _build_chunk_index(self):
        """
        Pre-calculate the chunk information for each audio file.
        This improves performance by avoiding recalculating this for each __getitem__ call.
        """
        self.chunks_per_file = []
        self.total_chunks = 0
        self.chunk_to_file_map = []  # Maps global chunk index to (file_idx, local_chunk_idx)
        
        # Create iterator with or without progress bar
        if self.verbose:
            file_iter = tqdm(
                range(self.num_files), 
                desc="Indexing audio chunks", 
                unit="files"
            )
        else:
            file_iter = range(self.num_files)
        
        for i in file_iter:
            audio_data = self.train_dataset[i]["audio"]["array"]
            orig_sr = self.train_dataset[i]["audio"]["sampling_rate"]
            
            # Calculate resampled length
            if orig_sr != self.target_sr:
                resampled_length = int(len(audio_data) * (self.target_sr / orig_sr))
            else:
                resampled_length = len(audio_data)
            
            # Calculate number of chunks for this file
            num_chunks = math.ceil(resampled_length / self.samples_per_chunk)
            self.chunks_per_file.append(num_chunks)
            
            # Update the chunk mapping
            for local_chunk_idx in range(num_chunks):
                self.chunk_to_file_map.append((i, local_chunk_idx))
            
            # Update total chunks
            self.total_chunks += num_chunks
            
            # Update progress bar with chunks info
            if self.verbose:
                file_iter.set_postfix(total_chunks=self.total_chunks)
    
    def _get_audio_chunk(self, file_idx: int, chunk_idx: int) -> torch.Tensor:
        """
        Retrieves a specific chunk of audio from a file.
        
        Args:
            file_idx: Index of the file in the dataset
            chunk_idx: Index of the chunk within the file
            
        Returns:
            The audio chunk as a tensor
        """
        audio_data = self.train_dataset[file_idx]["audio"]["array"]
        orig_sr = self.train_dataset[file_idx]["audio"]["sampling_rate"]
        
        # Resample if needed
        if orig_sr != self.target_sr:
            audio_data = librosa.resample(
                y=audio_data.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=self.target_sr
            )
        
        # Convert to tensor if it's not already
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float32)
        
        # Calculate start and end positions for the chunk
        start_pos = chunk_idx * self.samples_per_chunk
        end_pos = min(start_pos + self.samples_per_chunk, len(audio_data))
        
        # Extract the chunk
        chunk = audio_data[start_pos:end_pos]
        
        # Pad if necessary
        if len(chunk) < self.samples_per_chunk:
            padding = torch.zeros(self.samples_per_chunk - len(chunk), dtype=torch.float32)
            chunk = torch.cat([chunk, padding])
        
        # Normalize if needed
        if self.normalize:
            norm = torch.norm(chunk) + self.epsilon
            chunk = chunk / norm
        
        return chunk
    
    def _compute_cqt(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute CQT for an audio tensor.
        
        Args:
            audio_tensor: Audio tensor to transform
            
        Returns:
            CQT image tensor in (channels, height, width) format
        """
        try:
            # Process the audio chunk using CQTProcessor
            processor = CQTProcessor(
                audio=audio_tensor,
                sr=self.target_sr,
                **self.cqt_params
            )
            
            # Get CQT as tensor (channel-first format)
            return processor.export_to_pytorch()
            
        except Exception as e:
            if self.verbose:
                print(f"CQT computation failed: {e}. Returning zeros.")
            
            # Return zeros with correct shape if CQT fails
            # Estimate dimensions based on typical CQT outputs
            n_bins = self.cqt_params.get('n_bins', 84)
            hop_length = self.cqt_params.get('hop_length', 128)
            est_width = int(self.samples_per_chunk / hop_length)
            
            return torch.zeros((2, n_bins, est_width), dtype=torch.float32)
    
    def _get_cqt_with_cache(self, chunk_key: Tuple[int, int], audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get CQT with caching. Uses in-memory cache for performance.
        
        Args:
            chunk_key: Tuple of (file_idx, chunk_idx)
            audio_tensor: Audio tensor to transform if not cached
            
        Returns:
            CQT image tensor
        """
        # Check if in cache
        if chunk_key in self._cqt_cache:
            return self._cqt_cache[chunk_key]
            
        # Compute the CQT
        cqt_tensor = self._compute_cqt(audio_tensor)
        
        # Store in cache
        if len(self._cqt_cache) >= self._cqt_cache_size:
            # Simple LRU approach: remove a random item
            self._cqt_cache.pop(next(iter(self._cqt_cache)))
        
        self._cqt_cache[chunk_key] = cqt_tensor
        return cqt_tensor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the chunk to fetch
        Returns:
            tensor: Audio chunk or CQT image tensor, depending on output_format
        """
        # If epoch_size is set, map the index to a random chunk
        if self.epoch_size is not None and self.epoch_size != self.total_chunks:
            chunk_index = index % self.total_chunks
        else:
            chunk_index = index
        
        # Get file and chunk indices
        file_idx, local_chunk_idx = self.chunk_to_file_map[chunk_index]
        
        # Get the audio chunk
        audio_chunk = self._get_audio_chunk(file_idx, local_chunk_idx)
        
        # Apply custom transform if provided (has priority over output_format)
        if self.transform:
            return self.transform(audio_chunk)
        
        # Return based on output format
        if self.output_format == 'audio':
            return audio_chunk
        elif self.output_format == 'cqt':
            # Use cached CQT computation if enabled
            if self.cache_cqt:
                return self._get_cqt_with_cache((file_idx, local_chunk_idx), audio_chunk)
            else:
                return self._compute_cqt(audio_chunk)

    def __len__(self):
        """Returns the number of chunks in the dataset."""
        if self.epoch_size is not None:
            return self.epoch_size
        return self.total_chunks 