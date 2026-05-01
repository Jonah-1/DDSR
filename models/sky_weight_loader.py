"""
Sky weight loader - Transfer pretrained weights from DGGT to ReconDrive.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict


def load_dggt_sky_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load DGGT sky model weights from checkpoint.

    Args:
        checkpoint_path: Path to DGGT checkpoint file

    Returns:
        Dictionary of sky model state_dict with 'sky_model.' prefix removed
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Extract sky_model weights and remove prefix
    sky_state = {}
    for k, v in state_dict.items():
        if 'sky_model' in k:
            new_k = k.replace('sky_model.', '')
            sky_state[new_k] = v

    return sky_state


def adapt_sky_weights_to_recondrive(
    checkpoint_path: str,
    recondrive_sky_model,
    verbose: bool = True
) -> bool:
    """
    Load compatible weights from DGGT checkpoint into ReconDrive sky model.

    This function:
    1. Loads DGGT sky model weights
    2. Filters for compatible shapes
    3. Applies them to ReconDrive model using strict=False

    Args:
        checkpoint_path: Path to DGGT checkpoint
        recondrive_sky_model: ReconDrive SkyGaussian instance
        verbose: Print detailed loading information

    Returns:
        True if loading succeeded, False otherwise
    """
    try:
        # Load DGGT weights
        dggt_sky_state = load_dggt_sky_checkpoint(checkpoint_path)

        if verbose:
            print(f"\n{'='*80}")
            print("Loading Sky Model Weights from DGGT")
            print(f"{'='*80}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"DGGT weights: {len(dggt_sky_state)} keys, {sum(v.numel() for v in dggt_sky_state.values()):,} parameters")

        # Get current model state
        model_state = recondrive_sky_model.state_dict()

        # Analyze compatibility
        matched_layers = {}
        skipped_layers = {}

        for k, v in dggt_sky_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    matched_layers[k] = v
                else:
                    skipped_layers[k] = (v.shape, model_state[k].shape)
            else:
                skipped_layers[k] = (v.shape, None)

        if verbose:
            print(f"\n✓ Matched layers ({len(matched_layers)}):")
            total_matched = 0
            for k, v in matched_layers.items():
                num_params = v.numel()
                total_matched += num_params
                print(f"  • {k:<40} {str(v.shape):<25} {num_params:>10,} params")

            if skipped_layers:
                print(f"\n⚠ Skipped layers ({len(skipped_layers)}) - shape mismatch:")
                for k, shapes in skipped_layers.items():
                    dggt_shape, model_shape = shapes
                    print(f"  • {k:<40} DGGT: {str(dggt_shape):<20} ReconDrive: {str(model_shape)}")

            print(f"\n{'='*80}")
            print(f"Loading Summary:")
            print(f"  Loaded:  {len(matched_layers):>3} layers, {total_matched:>10,} parameters")
            print(f"  Skipped: {len(skipped_layers):>3} layers")
            print(f"{'='*80}\n")

        # Load matched weights using strict=False
        missing_keys, unexpected_keys = recondrive_sky_model.load_state_dict(
            matched_layers,
            strict=False
        )

        if verbose:
            if missing_keys:
                print(f"ℹ Missing keys (expected - will train fresh): {missing_keys}")
            if unexpected_keys:
                print(f"ℹ Unexpected keys: {unexpected_keys}")

        return True

    except Exception as e:
        print(f"✗ Error loading sky weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_sky_weights_loaded(sky_model) -> Dict[str, any]:
    """
    Verify that sky model weights have been loaded and are not random.

    Args:
        sky_model: ReconDrive SkyGaussian instance

    Returns:
        Dictionary with verification statistics
    """
    stats = {
        'bg_pcd_mean': sky_model.bg_pcd.mean().item(),
        'bg_pcd_std': sky_model.bg_pcd.std().item(),
        'bg_scales_mean': sky_model.bg_scales.mean().item(),
        'bg_scales_std': sky_model.bg_scales.std().item(),
    }

    # Extract MLP weights from Sequential
    # bg_field.layers[0] is Linear(3, 64), [2] is Linear(64, 64), [4] is Linear(64, 3)
    linear_layers = [m for m in sky_model.bg_field.layers if isinstance(m, nn.Linear)]
    if len(linear_layers) >= 1:
        stats['bg_field_first_layer_weight_mean'] = linear_layers[0].weight.mean().item()
        stats['bg_field_first_layer_weight_std'] = linear_layers[0].weight.std().item()

    if len(linear_layers) >= 3:
        stats['bg_field_output_layer_weight_mean'] = linear_layers[-1].weight.mean().item()
        stats['bg_field_output_layer_weight_std'] = linear_layers[-1].weight.std().item()

    return stats
