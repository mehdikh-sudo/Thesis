# """
# Computes SP (score prediction) MACs for the STVT model
# (https://github.com/nchucvml/STVT), following the CSTA-paper convention of
# reporting SP as MACs-per-video (excluding key-shot selection).

# Requires the STVT repo's `STVT` package to be importable -- run this from
# inside your STVT-main/STVT/ folder (same place train.py lives), or add that
# path with sys.path.insert as done below.

# Usage:
#     pip install fvcore --break-system-packages   # one-time
#     python compute_macs_stvt_sp.py
# """

# import os
# import sys
# import cv2
# import torch
# from fvcore.nn import FlopCountAnalysis

# # -----------------------
# # SETTINGS
# # -----------------------
# STVT_REPO_PATH = "."          # path to STVT-main/STVT (folder containing train.py)
# DATASET = "SumMe"             # 'TVSum' or 'SumMe' -- picks the config inside STVT()
# SEQUENCE_LEN = 16             # matches --sequence default in train.py (frames per chunk)
# FEATURE_DIM = 512             # your PCA-reduced feature dim (matches num_channels in STVT.py)

# # video_dir used only to count avg frames/video -- point at whichever raw video
# # folder you use (RGB or flow, frame counts should match)
# video_dir = "/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/SumMe/videos"

# sys.path.insert(0, STVT_REPO_PATH)
# from STVT.models.STVT import STVT


# def macs_per_chunk():
#     model = STVT(dataset=DATASET)
#     model.eval()
#     # input shape [batch, 512, 4, 4] -> represents SEQUENCE_LEN=16 frames of
#     # 512-d features arranged as a 4x4 grid (img_dim=4, patch_dim=1 -> 16 patches)
#     dummy = torch.randn(1, FEATURE_DIM, 4, 4)
#     flops = FlopCountAnalysis(model, dummy)
#     flops.unsupported_ops_warnings(False)
#     return flops.total()


# def get_frame_counts(video_dir):
#     counts = []
#     if not os.path.isdir(video_dir):
#         return counts
#     video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
#     for video in video_files:
#         cap = cv2.VideoCapture(os.path.join(video_dir, video))
#         n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         if n > 0:
#             counts.append(n)
#     return counts


# def main():
#     per_chunk_macs = macs_per_chunk()
#     print(f"SP MACs per {SEQUENCE_LEN}-frame chunk: {per_chunk_macs:.3e} "
#           f"({per_chunk_macs / 1e9:.4f} GMACs)")

#     frame_counts = get_frame_counts(video_dir)
#     if not frame_counts:
#         print(f"\nNo .mp4 files found in {video_dir} -- edit `video_dir` at the "
#               f"top of this script, or enter avg_frames manually below.")
#         return

#     avg_frames = sum(frame_counts) / len(frame_counts)
#     # STVT processes non-overlapping chunks of SEQUENCE_LEN frames; a partial
#     # final chunk still requires a full forward pass -> ceil division
#     avg_num_chunks = -(-avg_frames // SEQUENCE_LEN)

#     sp_macs_per_video = per_chunk_macs * avg_num_chunks

#     print(f"\nDataset: {DATASET}")
#     print(f"Videos found: {len(frame_counts)}")
#     print(f"Average frames/video: {avg_frames:.1f}")
#     print(f"Average chunks/video (ceil, {SEQUENCE_LEN}-frame windows): {avg_num_chunks:.1f}")
#     print(f"\n--> SP MACs per video: {sp_macs_per_video:.3e} ({sp_macs_per_video / 1e9:.2f} G)")
#     print("\nThis SP number pairs with your FE total (I3D two-stream or ResNet, "
#           "whichever features feed this run) to complete the FE/SP row for STVT "
#           "in your comparison table.")


# if __name__ == "__main__":
#     main()


"""
Computes SP (score prediction) MACs for the STVT model
(https://github.com/nchucvml/STVT), following the CSTA-paper convention of
reporting SP as MACs-per-video (excluding key-shot selection).

Requires the STVT repo's `STVT` package to be importable -- run this from
inside your STVT-main/STVT/ folder (same place train.py lives), or add that
path with sys.path.insert as done below.

Usage:
    pip install fvcore --break-system-packages   # one-time
    python compute_macs_stvt_sp.py
"""

import os
import sys
import cv2
import torch
from fvcore.nn import FlopCountAnalysis

# -----------------------
# SETTINGS
# -----------------------
STVT_REPO_PATH = "."          # path to STVT-main/STVT (folder containing train.py)
DATASET = "SumMe"             # 'TVSum' or 'SumMe' -- picks the config inside STVT()
SEQUENCE_LEN = 16             # matches --sequence default in train.py (frames per chunk)
FEATURE_DIM = 512             # your PCA-reduced feature dim (matches num_channels in STVT.py)

# video_dir used only to count avg frames/video -- point at whichever raw video
# folder you use (RGB or flow, frame counts should match)
video_dir = "/Users/mehdikhosravi/Master/Thesis/STVT-main/Feature extraction/"+DATASET+"/videos"

sys.path.insert(0, STVT_REPO_PATH)
# NOTE: STVT() factory hardcodes embedding_dim=768/num_heads=12/hidden_dim=3072
# for BOTH datasets in the original repo -- it does NOT know about your halved
# SumMe config, so we build the model directly instead of calling STVT(dataset=...).
from STVT.models.STVT import SpatioTemporal_Vision_Transformer

# Set to True for the halved-width SumMe model, False for the standard TVSum model
USE_HALVED_WIDTH = True


def macs_per_chunk():
    if USE_HALVED_WIDTH:
        embedding_dim, num_heads, hidden_dim = 384, 6, 1536   # your halved SumMe config
    else:
        embedding_dim, num_heads, hidden_dim = 768, 12, 3072  # standard config

    model = SpatioTemporal_Vision_Transformer(
        img_dim=4,
        patch_dim=1,
        out_dim=2,
        num_channels=FEATURE_DIM,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=12,
        hidden_dim=hidden_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )
    model.eval()
    # input shape [batch, 512, 4, 4] -> represents SEQUENCE_LEN=16 frames of
    # 512-d features arranged as a 4x4 grid (img_dim=4, patch_dim=1 -> 16 patches)
    dummy = torch.randn(1, FEATURE_DIM, 4, 4)
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    return flops.total()


def get_frame_counts(video_dir):
    counts = []
    if not os.path.isdir(video_dir):
        return counts
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    for video in video_files:
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            counts.append(n)
    return counts


def main():
    per_chunk_macs = macs_per_chunk()
    print(f"SP MACs per {SEQUENCE_LEN}-frame chunk: {per_chunk_macs:.3e} "
          f"({per_chunk_macs / 1e9:.4f} GMACs)")

    frame_counts = get_frame_counts(video_dir)
    if not frame_counts:
        print(f"\nNo .mp4 files found in {video_dir} -- edit `video_dir` at the "
              f"top of this script, or enter avg_frames manually below.")
        return

    avg_frames = sum(frame_counts) / len(frame_counts)
    # STVT processes non-overlapping chunks of SEQUENCE_LEN frames; a partial
    # final chunk still requires a full forward pass -> ceil division
    avg_num_chunks = -(-avg_frames // SEQUENCE_LEN)

    sp_macs_per_video = per_chunk_macs * avg_num_chunks

    print(f"\nDataset: {DATASET}")
    print(f"Videos found: {len(frame_counts)}")
    print(f"Average frames/video: {avg_frames:.1f}")
    print(f"Average chunks/video (ceil, {SEQUENCE_LEN}-frame windows): {avg_num_chunks:.1f}")
    print(f"\n--> SP MACs per video: {sp_macs_per_video:.3e} ({sp_macs_per_video / 1e9:.2f} G)")
    print("\nThis SP number pairs with your FE total (I3D two-stream or ResNet, "
          "whichever features feed this run) to complete the FE/SP row for STVT "
          "in your comparison table.")


if __name__ == "__main__":
    main()