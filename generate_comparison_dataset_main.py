#!/usr/bin/env python3
"""
Main script to generate comparison dataset with GRPO-compatible CQT ViT model and FreeMusic dataset.

This script demonstrates the complete workflow for creating preference data
that can be used to train reward models for GRPO optimization.

Usage:
    python generate_comparison_dataset_main.py [options]

Examples:
    # Generate 50 comparisons with default settings
    python generate_comparison_dataset_main.py --num_samples 50
    
    # Use dual temperature method with custom temperatures
    python generate_comparison_dataset_main.py --method dual_temperature --temp_a 0.3 --temp_b 2.0
    
    # Use single temperature method with variance comparison
    python generate_comparison_dataset_main.py --method single_temperature --temperature 1.5
    
    # Save to custom location
    python generate_comparison_dataset_main.py --output comparison_data.pkl --num_samples 100
"""

import argparse
import torch
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the repository root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cqt_vit_model import CQTViTModel
from data.freemusic import FreeMusic
from preferences.comparison_dataset_generator import generate_comparison_dataset


def setup_model(num_devices=4, distribution_size=32, device='auto', use_half_precision=False, 
                use_8bit=False, gradient_checkpointing=False, dataset=None):
    """
    Set up the GRPO-compatible CQT ViT model.
    
    Args:
        num_devices: Number of output devices
        distribution_size: Size of probability distributions
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        use_half_precision: Use FP16 for memory efficiency
        use_8bit: Use 8-bit quantization
        gradient_checkpointing: Enable gradient checkpointing
        dataset: FreeMusic dataset to get actual CQT shape from (optional)
        
    Returns:
        Configured model and CQT shape tuple
    """
    print("🤖 Setting up GRPO-compatible CQT ViT model...")
    
    # Get actual CQT shape from dataset if provided, otherwise use calculation
    if dataset is not None:
        sample_cqt = dataset[0]
        cqt_shape = tuple(sample_cqt.shape)
        print(f"Using actual CQT shape from dataset: {cqt_shape}")
    else:
        # CQT shape based on FreeMusic dataset defaults (fallback)
        # (channels, frequency_bins, time_steps)
        cqt_shape = (2, 84, int(22050 * 30.0 / 64) + 1)  # +1 to match actual output
    
    print(f"Model configuration:")
    print(f"  - CQT shape: {cqt_shape}")
    print(f"  - Output devices: {num_devices}")
    print(f"  - Distribution size: {distribution_size}")
    print(f"  - Device: {device}")
    print(f"  - Half precision: {use_half_precision}")
    print(f"  - 8-bit quantization: {use_8bit}")
    print(f"  - Gradient checkpointing: {gradient_checkpointing}")
    
    model = CQTViTModel.create_model(
        cqt_shape=cqt_shape,
        num_devices=num_devices,
        distribution_size=distribution_size,
        device=device,
        use_half_precision=use_half_precision,
        use_8bit=use_8bit,
        gradient_checkpointing=gradient_checkpointing
    )
    
    print(f"✅ Model created successfully!")
    return model, cqt_shape


def setup_dataset(max_duration=5.0, sample_rate=22050, num_files_limit=None, verbose=True):
    """
    Set up the FreeMusic dataset for CQT generation.
    
    Args:
        max_duration: Maximum duration per audio chunk in seconds
        sample_rate: Target sample rate
        num_files_limit: Limit number of files for testing (None for all)
        verbose: Show progress during dataset setup
        
    Returns:
        Configured FreeMusic dataset
    """
    print("🎵 Setting up FreeMusic dataset...")
    
    print(f"Dataset configuration:")
    print(f"  - Max duration: {max_duration}s")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Output format: CQT")
    print(f"  - Cache enabled: True")
    
    # Set up CQT parameters to match model expectations
    cqt_params = {
        'hop_length': 64,      # Matches model expectations
        'n_bins': 84,          # Standard frequency bins
        'bins_per_octave': 12, # Standard chromatic scale
        'fmin': 32.70          # C1 note
    }
    
    dataset = FreeMusic(
        sample_rate=sample_rate,
        max_duration=max_duration,
        normalize=True,
        output_format='cqt',
        cqt_params=cqt_params,
        cache_cqt=True,
        verbose=verbose
    )
    
    print(f"✅ FreeMusic dataset loaded!")
    print(f"  - Total files: {dataset.num_files}")
    print(f"  - Total chunks: {dataset.total_chunks}")
    print(f"  - Samples per chunk: {dataset.samples_per_chunk}")
    
    # Limit dataset size for testing if requested
    if num_files_limit is not None:
        max_chunks = sum(dataset.chunks_per_file[:num_files_limit])
        dataset.epoch_size = min(max_chunks, dataset.total_chunks)
        print(f"  - Limited to {dataset.epoch_size} chunks (first {num_files_limit} files)")
    
    return dataset


def generate_comparison_data(model, dataset, args):
    """
    Generate comparison dataset using the specified method.
    
    Args:
        model: GRPO-compatible CQT ViT model
        dataset: FreeMusic dataset
        args: Command line arguments
        
    Returns:
        Path to saved comparison dataset
    """
    print(f"🔥 Generating comparison dataset...")
    print(f"Method: {args.method}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output file: {args.output}")
    
    if args.method == 'dual_temperature':
        print(f"Temperature A (deterministic): {args.temp_a}")
        print(f"Temperature B (exploratory): {args.temp_b}")
        
        saved_path = generate_comparison_dataset(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            save_path=args.output,
            temperature_a=args.temp_a,
            temperature_b=args.temp_b,
            comparison_method='dual_temperature'
        )
        
    elif args.method == 'single_temperature':
        print(f"Temperature: {args.temperature}")
        print("Comparing least diverse vs most diverse actions")
        
        saved_path = generate_comparison_dataset(
            model=model,
            dataset=dataset,
            num_samples=args.num_samples,
            save_path=args.output,
            temperature_a=args.temperature,  # Will be averaged in the function
            temperature_b=args.temperature,
            comparison_method='single_temperature'
        )
    
    else:
        raise ValueError(f"Unknown comparison method: {args.method}")
    
    print(f"✅ Comparison dataset generated and saved to: {saved_path}")
    return saved_path


def analyze_dataset(saved_path):
    """
    Load and analyze the generated comparison dataset.
    
    Args:
        saved_path: Path to the saved comparison dataset
    """
    print(f"\n📊 Analyzing generated comparison dataset...")
    
    import pickle
    
    try:
        with open(saved_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        comparisons = dataset_info['comparisons']
        metadata = dataset_info['metadata']
        
        print(f"Dataset metadata:")
        print(f"  - Number of comparisons: {metadata['num_comparisons']}")
        print(f"  - Model type: {metadata['model_info']}")
        print(f"  - Dataset type: {metadata['dataset_info']}")
        print(f"  - Generation method: {metadata['generation_method']}")
        
        if comparisons:
            # Analyze preferences
            preferences = [comp['preference'] for comp in comparisons]
            pref_counts = {}
            for pref in preferences:
                pref_counts[pref] = pref_counts.get(pref, 0) + 1
            
            print(f"\nPreference distribution:")
            total = len(preferences)
            for pref, count in pref_counts.items():
                pref_name = "A (deterministic/less diverse)" if pref == 0 else "B (exploratory/more diverse)"
                print(f"  - {pref_name}: {count} ({count/total*100:.1f}%)")
            
            # Analyze data shapes
            sample_comp = comparisons[0]
            print(f"\nData shapes:")
            print(f"  - Input CQT: {sample_comp['input'].shape}")
            print(f"  - Output A: {sample_comp['output_a'].shape}")
            print(f"  - Output B: {sample_comp['output_b'].shape}")
            print(f"  - Audio: {sample_comp['audio'].shape}")
            
            # Show temperature info
            if 'temperature_a' in sample_comp:
                print(f"\nTemperature settings:")
                print(f"  - Temperature A: {sample_comp['temperature_a']}")
                print(f"  - Temperature B: {sample_comp['temperature_b']}")
            elif 'temperature' in sample_comp:
                print(f"\nTemperature setting: {sample_comp['temperature']}")
        
        print(f"✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Failed to analyze dataset: {e}")


def main():
    """Main function to run the comparison dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparison dataset with GRPO-compatible CQT ViT model and FreeMusic dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of comparison pairs to generate')
    parser.add_argument('--max_duration', type=float, default=5.0,
                       help='Maximum duration per audio chunk in seconds')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Target sample rate for audio')
    parser.add_argument('--num_files_limit', type=int, default=None,
                       help='Limit to first N files for testing (None for all)')
    
    # Model arguments
    parser.add_argument('--num_devices', type=int, default=4,
                       help='Number of output devices')
    parser.add_argument('--distribution_size', type=int, default=32,
                       help='Size of probability distributions')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for model')
    parser.add_argument('--use_half_precision', action='store_true',
                       help='Use FP16 for memory efficiency')
    parser.add_argument('--use_8bit', action='store_true',
                       help='Use 8-bit quantization')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing')
    
    # Comparison generation arguments
    parser.add_argument('--method', type=str, default='dual_temperature',
                       choices=['dual_temperature', 'single_temperature'],
                       help='Comparison generation method')
    parser.add_argument('--temp_a', type=float, default=0.5,
                       help='Temperature A (deterministic) for dual_temperature method')
    parser.add_argument('--temp_b', type=float, default=1.5,
                       help='Temperature B (exploratory) for dual_temperature method')
    parser.add_argument('--temperature', type=float, default=1.2,
                       help='Temperature for single_temperature method')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for comparison dataset')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show detailed progress information')
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Analyze the generated dataset')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"comparison_dataset_{args.method}_{timestamp}.pkl"
    
    print("🚀 Starting Comparison Dataset Generation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Set up dataset first
        dataset = setup_dataset(
            max_duration=args.max_duration,
            sample_rate=args.sample_rate,
            num_files_limit=args.num_files_limit,
            verbose=args.verbose
        )
        print()
        
        # Set up model using actual dataset shape
        model, cqt_shape = setup_model(
            num_devices=args.num_devices,
            distribution_size=args.distribution_size,
            device=args.device,
            use_half_precision=args.use_half_precision,
            use_8bit=args.use_8bit,
            gradient_checkpointing=args.gradient_checkpointing,
            dataset=dataset
        )
        print()
        
        # Generate comparison data
        saved_path = generate_comparison_data(model, dataset, args)
        print()
        
        # Analyze results if requested
        if args.analyze:
            analyze_dataset(saved_path)
        
        print(f"\n🎉 Success! Comparison dataset generation completed.")
        print(f"📁 Dataset saved to: {saved_path}")
        print(f"\n🔗 Next steps:")
        print(f"1. Use this dataset to train a reward model")
        print(f"2. Integrate the reward model with GRPO training")
        print(f"3. Fine-tune the policy model using human preferences")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 