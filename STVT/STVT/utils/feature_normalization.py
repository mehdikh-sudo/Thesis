"""
Lightweight helpers to z-score normalize per-modality features before concatenation.

This keeps each feature dimension zero-mean, unit-variance within a modality,
which is a common baseline when mixing modalities (e.g., RGB, Flow, ResNet).
"""

import os
from typing import Sequence, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA


def _zscore(feature: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-dimension z-score.

    Args:
        feature: Array shaped (num_samples, dim).
        eps: Small constant to avoid divide-by-zero.

    Returns:
        normalized, mean, std (all per-dimension).
    """
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {feature.shape}")

    mean = feature.mean(axis=0)
    std = feature.std(axis=0)
    std_safe = np.where(std < eps, 1.0, std)  # avoid exploding tiny std
    normed = (feature - mean) / std_safe
    return normed, mean, std_safe


def normalize_and_concat_modalities(
    modalities: Sequence[np.ndarray], eps: float = 1e-8
) -> Tuple[np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Z-score each modality separately, then concatenate on the feature dimension.

    Args:
        modalities: Iterable of arrays shaped (num_samples, dim_i) with aligned rows.
        eps: Small constant to avoid divide-by-zero.

    Returns:
        concatenated_normalized: Array shaped (num_samples, sum dim_i).
        means: List of per-dimension means for each modality.
        stds: List of per-dimension stds (after epsilon guard) for each modality.

    Example:
        rgb_flow_resnet_norm, means, stds = normalize_and_concat_modalities([rgb, flow, resnet])
    """
    if len(modalities) == 0:
        raise ValueError("No modalities provided.")

    # Ensure all modalities have matching sample count
    num_samples = modalities[0].shape[0]
    for i, m in enumerate(modalities):
        if m.shape[0] != num_samples:
            raise ValueError(f"Modality {i} has {m.shape[0]} samples; expected {num_samples}.")

    normed_list = []
    means = []
    stds = []
    for m in modalities:
        normed, mean, std = _zscore(m, eps=eps)
        normed_list.append(normed)
        means.append(mean)
        stds.append(std)

    concatenated = np.concatenate(normed_list, axis=1)
    return concatenated, means, stds


def normalize_rgb_flow_resnet(
    rgb: np.ndarray, flow: np.ndarray, resnet: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Convenience wrapper for the common triple-modality case.
    """
    return normalize_and_concat_modalities([rgb, flow, resnet], eps=eps)


def apply_pca(
    features: np.ndarray, target_dim: int = 512
) -> np.ndarray:
    """
    Apply PCA to reduce feature dimensions.

    Args:
        features: Array shaped (num_samples, dim).
        target_dim: Target dimensionality (default: 512).

    Returns:
        Reduced features array shaped (num_samples, min(target_dim, num_samples, dim)).

    Note:
        PCA is fitted per-video (no global PCA across videos).
        If target_dim exceeds num_samples or original dim, it will be capped.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {features.shape}")

    num_samples, dim = features.shape
    n_components = min(target_dim, num_samples, dim)

    if n_components < 2:
        raise ValueError(
            f"Not enough frames/dimensions for PCA: num_samples={num_samples}, "
            f"dim={dim}, target_dim={target_dim}"
        )

    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False)
    reduced = pca.fit_transform(features)

    return reduced


def _load_feature_file(path: str) -> np.ndarray:
    """
    Load a .npy feature file and return array shaped (num_samples, dim).

    Args:
        path: Path to .npy file.

    Returns:
        Array shaped (num_samples, dim), ensuring 2D shape.
    """
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        # Flatten all but first dimension (assumed to be time/samples)
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def load_and_normalize_from_directories(
    rgb_dir: str,
    flow_dir: str,
    resnet_dir: str,
    filename: str,
    eps: float = 1e-8,
    align_time: bool = True,
    use_pca: bool = False,
    pca_dim: int = 512,
) -> Tuple[np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Load RGB, Flow, and ResNet features from separate directories for a single video
    and normalize them.

    Args:
        rgb_dir: Directory containing RGB feature .npy files.
        flow_dir: Directory containing Flow feature .npy files.
        resnet_dir: Directory containing ResNet/STVT feature .npy files.
        filename: Name of the feature file (e.g., 'video_1.npy') - must exist in all three directories.
        eps: Small constant to avoid divide-by-zero.
        align_time: If True, align features to the minimum temporal length (default: True).
        use_pca: If True, apply PCA to reduce dimensions to pca_dim (default: False).
        pca_dim: Target dimensionality for PCA (default: 512).

    Returns:
        concatenated_normalized: Array shaped (num_samples, sum dim_i or pca_dim) with normalized and concatenated features.
        means: List of per-dimension means for each modality [rgb_mean, flow_mean, resnet_mean].
        stds: List of per-dimension stds for each modality [rgb_std, flow_std, resnet_std].

    Example:
        normalized, means, stds = load_and_normalize_from_directories(
            rgb_dir="/path/to/rgb",
            flow_dir="/path/to/flow",
            resnet_dir="/path/to/resnet",
            filename="video_1.npy",
            use_pca=True,
            pca_dim=512
        )
    """
    rgb_path = os.path.join(rgb_dir, filename)
    flow_path = os.path.join(flow_dir, filename)
    resnet_path = os.path.join(resnet_dir, filename)

    # Check if all files exist
    for path, modality in [(rgb_path, "RGB"), (flow_path, "Flow"), (resnet_path, "ResNet")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{modality} feature file not found: {path}")

    # Load features
    rgb_feat = _load_feature_file(rgb_path)
    flow_feat = _load_feature_file(flow_path)
    resnet_feat = _load_feature_file(resnet_path)

    # Align temporal dimension if requested
    if align_time:
        min_length = min(rgb_feat.shape[0], flow_feat.shape[0], resnet_feat.shape[0])
        if not (rgb_feat.shape[0] == flow_feat.shape[0] == resnet_feat.shape[0]):
            print(f"Warning: Temporal mismatch for {filename}: RGB={rgb_feat.shape[0]}, "
                  f"Flow={flow_feat.shape[0]}, ResNet={resnet_feat.shape[0]}. Truncating to {min_length}.")
        rgb_feat = rgb_feat[:min_length]
        flow_feat = flow_feat[:min_length]
        resnet_feat = resnet_feat[:min_length]

    # Normalize and concatenate
    normalized, means, stds = normalize_rgb_flow_resnet(rgb_feat, flow_feat, resnet_feat, eps=eps)

    # Apply PCA if requested
    if use_pca:
        normalized = apply_pca(normalized, target_dim=pca_dim)

    return normalized, means, stds


def process_all_videos_from_directories(
    rgb_dir: str,
    flow_dir: str,
    resnet_dir: str,
    output_dir: Optional[str] = None,
    eps: float = 1e-8,
    align_time: bool = True,
    use_pca: bool = False,
    pca_dim: int = 512,
) -> dict:
    """
    Load and normalize features for all matching videos across three directories.

    Args:
        rgb_dir: Directory containing RGB feature .npy files.
        flow_dir: Directory containing Flow feature .npy files.
        resnet_dir: Directory containing ResNet/STVT feature .npy files.
        output_dir: Optional directory to save normalized features. If None, features are not saved.
        eps: Small constant to avoid divide-by-zero.
        align_time: If True, align features to the minimum temporal length (default: True).
        use_pca: If True, apply PCA to reduce dimensions to pca_dim (default: False).
        pca_dim: Target dimensionality for PCA (default: 512).

    Returns:
        Dictionary mapping filenames to tuples of (normalized_features, means, stds).

    Example:
        results = process_all_videos_from_directories(
            rgb_dir="/path/to/rgb",
            flow_dir="/path/to/flow",
            resnet_dir="/path/to/resnet",
            output_dir="/path/to/output",
            use_pca=True,
            pca_dim=512
        )
    """
    # Get all .npy files from RGB directory
    rgb_files = {f for f in os.listdir(rgb_dir) if f.endswith(".npy")}

    # Find files that exist in all three directories
    common_files = []
    for fname in sorted(rgb_files):
        flow_path = os.path.join(flow_dir, fname)
        resnet_path = os.path.join(resnet_dir, fname)
        if os.path.exists(flow_path) and os.path.exists(resnet_path):
            common_files.append(fname)
        else:
            print(f"Warning: Skipping {fname} - not found in all directories")

    if len(common_files) == 0:
        raise ValueError("No matching files found across all three directories")

    print(f"Processing {len(common_files)} videos...")

    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    results = {}
    for fname in common_files:
        try:
            normalized, means, stds = load_and_normalize_from_directories(
                rgb_dir, flow_dir, resnet_dir, fname, eps=eps, align_time=align_time,
                use_pca=use_pca, pca_dim=pca_dim
            )

            results[fname] = (normalized, means, stds)

            # Save if output directory is provided
            if output_dir is not None:
                output_path = os.path.join(output_dir, fname)
                np.save(output_path, normalized)
                print(f"✓ Saved: {fname} (shape: {normalized.shape})")

        except Exception as e:
            print(f"✗ Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Processed {len(results)} videos successfully.")
    return results


# Example usage:
#
# # For a single video (with PCA to dimension 512):
# from STVT.utils.feature_normalization import load_and_normalize_from_directories
# 
# normalized, means, stds = load_and_normalize_from_directories(
#     rgb_dir="/path/to/rgb/features",
#     flow_dir="/path/to/flow/features",
#     resnet_dir="/path/to/resnet/features",
#     filename="video_1.npy",
#     use_pca=True,
#     pca_dim=512
# )
#
# # For all videos in directories (with PCA):
# from STVT.utils.feature_normalization import process_all_videos_from_directories
#
results = process_all_videos_from_directories(
    flow_dir="/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/SumMe/Features/I3d flow",
    rgb_dir="/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/SumMe/Features/I3d RGB",
    resnet_dir='/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/SumMe/Features/resnet-stvt',
    output_dir='/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/SumMe/Features/normalized_concat_rfr',
    use_pca=True,
    pca_dim=512
)
