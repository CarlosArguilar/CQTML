import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from cqtml_interface.api import compare_signals


class ComparisonDatasetGenerator:
    """
    Generates comparison datasets by running model inference twice on the same input
    and determining preferences using cqtml_interface.api.compare_signals
    """
    
    def __init__(self, model, dataset, save_path="comparison_dataset.pkl"):
        """
        Args:
            model: The CQT ViT model to generate outputs
            dataset: The FreeMusic dataset with CQT format
            save_path: Path to save the comparison dataset
        """
        self.model = model
        self.dataset = dataset
        self.save_path = Path(save_path)
        self.comparisons = []
        
    def generate_comparison_dataset(self, num_samples=100, batch_size=1, 
                                   subset_indices=None, verbose=True, 
                                   temperature=1.2, use_stochastic=True):
        """
        Generate comparison dataset using model's stochastic generation capabilities
        
        Args:
            num_samples: Number of comparison pairs to generate
            batch_size: Batch size for inference (keep small for diversity)
            subset_indices: Specific indices to use from dataset
            verbose: Show progress bar
            temperature: Temperature for stochastic sampling (higher = more random)
            use_stochastic: Use model's stochastic generation mode
        """
        # Create subset if specified
        if subset_indices is not None:
            subset = Subset(self.dataset, subset_indices[:num_samples])
        else:
            # Use first num_samples from dataset
            indices = list(range(min(num_samples, len(self.dataset))))
            subset = Subset(self.dataset, indices)
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        if verbose:
            progress_bar = tqdm(dataloader, desc="Generating comparisons")
        else:
            progress_bar = dataloader
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_data = batch_data.to(device)
            
            with torch.no_grad():
                if use_stochastic and hasattr(self.model, 'generate_stochastic'):
                    # Use model's built-in stochastic generation
                    output_a = self.model.generate_deterministic(batch_data)
                    output_b = self.model.generate_stochastic(batch_data, temperature=temperature)
                elif use_stochastic:
                    # Fallback: use forward with different temperature settings
                    output_a = self.model(batch_data, temperature=1.0, stochastic=False)
                    output_b = self.model(batch_data, temperature=temperature, stochastic=True)
                else:
                    # Generate two deterministic outputs (should be identical, for testing)
                    output_a = self.model(batch_data)
                    output_b = self.model(batch_data)
            
            # Process each sample in the batch
            for i in range(batch_data.size(0)):
                # Get the actual dataset index for this sample
                actual_idx = subset.indices[batch_idx * batch_size + i] if hasattr(subset, 'indices') else batch_idx * batch_size + i
                
                # Get the original audio for this sample (same chunk that produced the CQT)
                file_idx, local_chunk_idx = self.dataset.chunk_to_file_map[actual_idx]
                audio_sample = self.dataset._get_audio_chunk(file_idx, local_chunk_idx).cpu()
                
                # CQT input and model outputs
                input_sample = batch_data[i].cpu()  # Original CQT input tensor
                output_a_sample = output_a[i].cpu()  # Model output A (deterministic)
                output_b_sample = output_b[i].cpu()  # Model output B (stochastic)
                
                # Compare the two outputs using original audio for playback
                try:
                    # compare_signals expects: (audio_tensor, signal_tensor_a, signal_tensor_b)
                    # audio_tensor: 1D audio for playback, signal tensors: device activations
                    preference = compare_signals(audio_sample, output_a_sample, output_b_sample)
                    
                    # Store the comparison (convert to numpy for storage)
                    comparison = {
                        'input': input_sample.numpy(),  # Original CQT input
                        'audio': audio_sample.numpy(),  # Original audio
                        'output_a': output_a_sample.numpy(),  # Model output A (deterministic)
                        'output_b': output_b_sample.numpy(),  # Model output B (stochastic)
                        'preference': preference,  # Result from compare_signals
                        'temperature': temperature,  # Temperature used for stochastic generation
                        'use_stochastic': use_stochastic,  # Whether stochastic mode was used
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'actual_idx': actual_idx
                    }
                    
                    self.comparisons.append(comparison)
                    
                    if verbose and len(self.comparisons) % 10 == 0:
                        progress_bar.set_postfix({
                            'comparisons': len(self.comparisons),
                            'last_pref': preference,
                            'temperature': temperature
                        })
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Comparison failed for sample {batch_idx}:{i}: {e}")
                    continue
        
        if verbose:
            print(f"Generated {len(self.comparisons)} comparison pairs")
            print(f"Used temperature: {temperature}")
            print(f"Stochastic mode: {use_stochastic}")
            self._print_statistics()
    
    def _print_statistics(self):
        """Print statistics about the generated comparisons"""
        if not self.comparisons:
            return
        
        preferences = [comp['preference'] for comp in self.comparisons]
        
        if isinstance(preferences[0], (int, float)):
            # Numeric preferences
            unique_prefs, counts = np.unique(preferences, return_counts=True)
            print("Preference distribution:")
            for pref, count in zip(unique_prefs, counts):
                print(f"  {pref}: {count} ({count/len(preferences)*100:.1f}%)")
        else:
            # Other preference format
            print(f"Generated {len(preferences)} preferences")
    
    def save_dataset(self, path=None):
        """Save the comparison dataset to disk"""
        save_path = Path(path) if path else self.save_path
        
        dataset_info = {
            'comparisons': self.comparisons,
            'metadata': {
                'num_comparisons': len(self.comparisons),
                'model_info': str(type(self.model).__name__),
                'dataset_info': str(type(self.dataset).__name__)
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"Saved {len(self.comparisons)} comparisons to {save_path}")
        return save_path
    
    def load_dataset(self, path=None):
        """Load a previously saved comparison dataset"""
        load_path = Path(path) if path else self.save_path
        
        with open(load_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        self.comparisons = dataset_info['comparisons']
        print(f"Loaded {len(self.comparisons)} comparisons from {load_path}")
        return dataset_info
    
    def get_training_data(self):
        """
        Convert comparisons to training format for reward model
        
        Returns:
            List of (input, output_preferred, output_rejected) tuples
        """
        training_data = []
        
        for comp in self.comparisons:
            input_data = comp['input']
            preference = comp['preference']
            
            # Determine preferred and rejected outputs based on preference
            if preference == 0 or preference == 'a':
                preferred = comp['output_a']
                rejected = comp['output_b']
            elif preference == 1 or preference == 'b':
                preferred = comp['output_b']
                rejected = comp['output_a']
            else:
                # Skip ambiguous preferences
                continue
            
            training_data.append({
                'input': input_data,
                'preferred': preferred,
                'rejected': rejected
            })
        
        return training_data
    
    @staticmethod
    def create_quick_comparison_dataset(model, dataset, num_samples=50, save_path=None, 
                                       temperature=1.2, use_stochastic=True):
        """
        Quick utility method to generate a small comparison dataset
        
        Args:
            model: CQT ViT model
            dataset: FreeMusic dataset with CQT format  
            num_samples: Number of comparison pairs
            save_path: Path to save dataset
            temperature: Temperature for stochastic sampling (higher = more random)
            use_stochastic: Use model's stochastic generation mode
        """
        generator = ComparisonDatasetGenerator(
            model=model, 
            dataset=dataset,
            save_path=save_path or "quick_comparison_dataset.pkl"
        )
        
        generator.generate_comparison_dataset(
            num_samples=num_samples,
            batch_size=1,
            verbose=True,
            temperature=temperature,
            use_stochastic=use_stochastic
        )
        
        saved_path = generator.save_dataset()
        return generator, saved_path


# Utility function for easy dataset generation
def generate_comparison_dataset(model, dataset, num_samples=100, save_path="comparison_dataset.pkl", 
                              temperature=1.2, use_stochastic=True):
    """
    Simple function to generate comparison dataset
    
    Args:
        model: CQT ViT model
        dataset: FreeMusic dataset with CQT format
        num_samples: Number of comparison pairs
        save_path: Where to save the dataset
        temperature: Temperature for stochastic sampling (higher = more random)
        use_stochastic: Use model's stochastic generation mode
        
    Returns:
        Path to saved dataset
    """
    generator = ComparisonDatasetGenerator(model, dataset, save_path)
    generator.generate_comparison_dataset(
        num_samples=num_samples, 
        temperature=temperature, 
        use_stochastic=use_stochastic
    )
    return generator.save_dataset() 