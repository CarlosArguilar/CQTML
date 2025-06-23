import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from cqtml_interface.api import compare_signals


class ComparisonDatasetGenerator:
    """
    Generates comparison datasets by running model inference with different configurations
    and determining preferences using cqtml_interface.api.compare_signals
    """
    
    def __init__(self, model, dataset, save_path="comparison_dataset.pkl"):
        """
        Args:
            model: The GRPO-compatible CQT ViT model to generate outputs
            dataset: The FreeMusic dataset with CQT format
            save_path: Path to save the comparison dataset
        """
        self.model = model
        self.dataset = dataset
        self.save_path = Path(save_path)
        self.comparisons = []
        
    def generate_comparison_dataset(self, num_samples=100, batch_size=1, 
                                   subset_indices=None, verbose=True, 
                                   temperature_a=0.5, temperature_b=1.5,
                                   actions_per_sample=2):
        """
        Generate comparison dataset using model's GRPO-compatible action generation
        
        Args:
            num_samples: Number of comparison pairs to generate
            batch_size: Batch size for inference (keep small for diversity)
            subset_indices: Specific indices to use from dataset
            verbose: Show progress bar
            temperature_a: Temperature for first action generation (lower = more deterministic)
            temperature_b: Temperature for second action generation (higher = more exploratory)
            actions_per_sample: Number of actions to generate per sample (we'll pick best/worst)
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
            
            # Process each sample in the batch
            for i in range(batch_data.size(0)):
                # Get the actual dataset index for this sample
                actual_idx = subset.indices[batch_idx * batch_size + i] if hasattr(subset, 'indices') else batch_idx * batch_size + i
                
                # Get the original audio for this sample (same chunk that produced the CQT)
                file_idx, local_chunk_idx = self.dataset.chunk_to_file_map[actual_idx]
                audio_sample = self.dataset._get_audio_chunk(file_idx, local_chunk_idx).cpu()
                
                # CQT input sample
                input_sample = batch_data[i:i+1]  # Keep batch dimension for model
                
                try:
                    # Generate actions with different temperatures for comparison
                    actions_a = self.model.generate_actions(
                        states=[input_sample.squeeze(0).cpu()],  # Remove batch dim, move to CPU
                        num_actions_per_state=actions_per_sample,
                        temperature=temperature_a
                    )
                    
                    actions_b = self.model.generate_actions(
                        states=[input_sample.squeeze(0).cpu()],  # Remove batch dim, move to CPU  
                        num_actions_per_state=actions_per_sample,
                        temperature=temperature_b
                    )
                    
                    # Select representative actions
                    # For temperature_a (lower), select first action (more deterministic)
                    # For temperature_b (higher), select action with highest variance (most exploratory)
                    output_a_sample = torch.tensor(actions_a[0][0], dtype=torch.float32)  # First action
                    
                    # For output_b, select the action with highest variance
                    if len(actions_b[0]) > 1:
                        variances = [np.var(action) for action in actions_b[0]]
                        most_diverse_idx = np.argmax(variances)
                        output_b_sample = torch.tensor(actions_b[0][most_diverse_idx], dtype=torch.float32)
                    else:
                        output_b_sample = torch.tensor(actions_b[0][0], dtype=torch.float32)
                    
                    # Compare the two outputs using original audio for playback
                    # compare_signals expects: (audio_tensor, signal_tensor_a, signal_tensor_b)
                    # audio_tensor: 1D audio for playback, signal tensors: device activations
                    preference = compare_signals(audio_sample, output_a_sample, output_b_sample)
                    
                    # Store the comparison (convert to numpy for storage)
                    comparison = {
                        'input': input_sample.squeeze(0).cpu().numpy(),  # Original CQT input
                        'audio': audio_sample.numpy(),  # Original audio
                        'output_a': output_a_sample.numpy(),  # Model output A (lower temp)
                        'output_b': output_b_sample.numpy(),  # Model output B (higher temp)
                        'preference': preference,  # Result from compare_signals
                        'temperature_a': temperature_a,  # Temperature used for output A
                        'temperature_b': temperature_b,  # Temperature used for output B
                        'actions_per_sample': actions_per_sample,  # Number of actions generated
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'actual_idx': actual_idx
                    }
                    
                    self.comparisons.append(comparison)
                    
                    if verbose and len(self.comparisons) % 10 == 0:
                        progress_bar.set_postfix({
                            'comparisons': len(self.comparisons),
                            'last_pref': preference,
                            'temp_a': temperature_a,
                            'temp_b': temperature_b
                        })
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Comparison failed for sample {batch_idx}:{i}: {e}")
                    continue
        
        if verbose:
            print(f"Generated {len(self.comparisons)} comparison pairs")
            print(f"Temperature A: {temperature_a} (more deterministic)")
            print(f"Temperature B: {temperature_b} (more exploratory)")
            print(f"Actions per sample: {actions_per_sample}")
            self._print_statistics()
    
    def generate_comparison_dataset_single_temp(self, num_samples=100, batch_size=1,
                                               subset_indices=None, verbose=True,
                                               temperature=1.2, actions_per_sample=4):
        """
        Generate comparison dataset using multiple actions at the same temperature
        
        Args:
            num_samples: Number of comparison pairs to generate
            batch_size: Batch size for inference
            subset_indices: Specific indices to use from dataset
            verbose: Show progress bar
            temperature: Temperature for action generation
            actions_per_sample: Number of actions to generate per sample (select best vs worst)
        """
        # Create subset if specified
        if subset_indices is not None:
            subset = Subset(self.dataset, subset_indices[:num_samples])
        else:
            indices = list(range(min(num_samples, len(self.dataset))))
            subset = Subset(self.dataset, indices)
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        
        if verbose:
            progress_bar = tqdm(dataloader, desc="Generating comparisons (single temp)")
        else:
            progress_bar = dataloader
        
        for batch_idx, batch_data in enumerate(progress_bar):
            for i in range(batch_data.size(0)):
                actual_idx = subset.indices[batch_idx * batch_size + i] if hasattr(subset, 'indices') else batch_idx * batch_size + i
                
                # Get audio and input
                file_idx, local_chunk_idx = self.dataset.chunk_to_file_map[actual_idx]
                audio_sample = self.dataset._get_audio_chunk(file_idx, local_chunk_idx).cpu()
                input_sample = batch_data[i:i+1]
                
                try:
                    # Generate multiple actions at same temperature
                    actions = self.model.generate_actions(
                        states=[input_sample.squeeze(0).cpu()],
                        num_actions_per_state=actions_per_sample,
                        temperature=temperature
                    )
                    
                    # Select most and least diverse actions
                    action_list = actions[0]
                    if len(action_list) >= 2:
                        # Calculate variance for each action
                        variances = [np.var(action) for action in action_list]
                        
                        # Select least diverse (most consistent) and most diverse
                        least_diverse_idx = np.argmin(variances)
                        most_diverse_idx = np.argmax(variances)
                        
                        output_a_sample = torch.tensor(action_list[least_diverse_idx], dtype=torch.float32)
                        output_b_sample = torch.tensor(action_list[most_diverse_idx], dtype=torch.float32)
                    else:
                        # Fallback: use the same action (should result in tie)
                        output_a_sample = torch.tensor(action_list[0], dtype=torch.float32)
                        output_b_sample = torch.tensor(action_list[0], dtype=torch.float32)
                    
                    # Compare outputs
                    preference = compare_signals(audio_sample, output_a_sample, output_b_sample)
                    
                    comparison = {
                        'input': input_sample.squeeze(0).cpu().numpy(),
                        'audio': audio_sample.numpy(),
                        'output_a': output_a_sample.numpy(),  # Less diverse
                        'output_b': output_b_sample.numpy(),  # More diverse
                        'preference': preference,
                        'temperature': temperature,
                        'actions_per_sample': actions_per_sample,
                        'comparison_type': 'single_temperature_variance',
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'actual_idx': actual_idx
                    }
                    
                    self.comparisons.append(comparison)
                    
                    if verbose and len(self.comparisons) % 10 == 0:
                        progress_bar.set_postfix({
                            'comparisons': len(self.comparisons),
                            'last_pref': preference,
                            'temp': temperature
                        })
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Comparison failed for sample {batch_idx}:{i}: {e}")
                    continue
        
        if verbose:
            print(f"Generated {len(self.comparisons)} comparison pairs (single temperature)")
            print(f"Temperature: {temperature}")
            print(f"Actions per sample: {actions_per_sample}")
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
                'dataset_info': str(type(self.dataset).__name__),
                'generation_method': 'grpo_compatible'
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
                                       temperature_a=0.5, temperature_b=1.5, 
                                       comparison_method='dual_temperature'):
        """
        Quick utility method to generate a small comparison dataset
        
        Args:
            model: GRPO-compatible CQT ViT model
            dataset: FreeMusic dataset with CQT format  
            num_samples: Number of comparison pairs
            save_path: Path to save dataset
            temperature_a: Lower temperature for more deterministic actions
            temperature_b: Higher temperature for more exploratory actions
            comparison_method: 'dual_temperature' or 'single_temperature'
        """
        generator = ComparisonDatasetGenerator(
            model=model, 
            dataset=dataset,
            save_path=save_path or "quick_comparison_dataset.pkl"
        )
        
        if comparison_method == 'dual_temperature':
            generator.generate_comparison_dataset(
                num_samples=num_samples,
                batch_size=1,
                verbose=True,
                temperature_a=temperature_a,
                temperature_b=temperature_b
            )
        else:  # single_temperature
            generator.generate_comparison_dataset_single_temp(
                num_samples=num_samples,
                batch_size=1,
                verbose=True,
                temperature=(temperature_a + temperature_b) / 2
            )
        
        saved_path = generator.save_dataset()
        return generator, saved_path


# Utility function for easy dataset generation
def generate_comparison_dataset(model, dataset, num_samples=100, save_path="comparison_dataset.pkl", 
                              temperature_a=0.5, temperature_b=1.5, comparison_method='dual_temperature'):
    """
    Simple function to generate comparison dataset with GRPO-compatible model
    
    Args:
        model: GRPO-compatible CQT ViT model
        dataset: FreeMusic dataset with CQT format
        num_samples: Number of comparison pairs
        save_path: Where to save the dataset
        temperature_a: Lower temperature for more deterministic actions  
        temperature_b: Higher temperature for more exploratory actions
        comparison_method: 'dual_temperature' or 'single_temperature'
        
    Returns:
        Path to saved dataset
    """
    generator = ComparisonDatasetGenerator(model, dataset, save_path)
    
    if comparison_method == 'dual_temperature':
        generator.generate_comparison_dataset(
            num_samples=num_samples, 
            temperature_a=temperature_a, 
            temperature_b=temperature_b
        )
    else:  # single_temperature
        generator.generate_comparison_dataset_single_temp(
            num_samples=num_samples,
            temperature=(temperature_a + temperature_b) / 2
        )
    
    return generator.save_dataset() 