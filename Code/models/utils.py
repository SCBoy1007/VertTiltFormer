import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, List

def init_weights(module: nn.Module) -> None:
    """Initialize network weights using Xavier/Kaiming initialization"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def generate_square_mask(sz: int) -> torch.Tensor:
    """Generate a square attention mask for sequence data"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def normalize_keypoints(
    keypoints: torch.Tensor,
    img_size: Tuple[int, int]
) -> torch.Tensor:
    """Normalize keypoint coordinates to [0, 1] range"""
    h, w = img_size
    keypoints = keypoints.clone()
    keypoints[..., 0] = keypoints[..., 0] / w
    keypoints[..., 1] = keypoints[..., 1] / h
    return keypoints

def denormalize_keypoints(
    keypoints: torch.Tensor,
    img_size: Tuple[int, int]
) -> torch.Tensor:
    """Denormalize keypoint coordinates from [0, 1] range to pixel coordinates"""
    h, w = img_size
    keypoints = keypoints.clone()
    keypoints[..., 0] = keypoints[..., 0] * w
    keypoints[..., 1] = keypoints[..., 1] * h
    return keypoints

def generate_heatmap(
    keypoints: torch.Tensor,
    img_size: Tuple[int, int],
    sigma: float = 2.0
) -> torch.Tensor:
    """Generate heatmaps from keypoint coordinates
    
    Args:
        keypoints: Tensor of shape (B, N, 2) containing keypoint coordinates
        img_size: Tuple of (height, width) for output heatmap size
        sigma: Gaussian sigma for heatmap generation
    
    Returns:
        Tensor of shape (B, N, H, W) containing heatmaps
    """
    B, N, _ = keypoints.shape
    H, W = img_size
    
    x = torch.arange(W, device=keypoints.device)
    y = torch.arange(H, device=keypoints.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    xx = xx.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    yy = yy.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    
    x_coord = keypoints[..., 0].view(B, N, 1, 1)
    y_coord = keypoints[..., 1].view(B, N, 1, 1)
    
    heatmaps = torch.exp(
        -((xx - x_coord) ** 2 + (yy - y_coord) ** 2) / (2 * sigma ** 2)
    )
    
    return heatmaps

def keypoints_to_heatmaps(
    keypoints: torch.Tensor,
    img_size: Tuple[int, int],
    sigma: float = 2.0
) -> torch.Tensor:
    """Convert keypoint coordinates to heatmaps
    
    Args:
        keypoints: Tensor of shape (B, N, 2) in normalized coordinates [0, 1]
        img_size: Target heatmap size (H, W)
        sigma: Gaussian sigma for heatmap generation
    
    Returns:
        Tensor of shape (B, N, H, W) containing heatmaps
    """
    # Denormalize keypoints to pixel coordinates
    keypoints_px = denormalize_keypoints(keypoints, img_size)
    
    # Generate heatmaps
    return generate_heatmap(keypoints_px, img_size, sigma)

def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Extract keypoint coordinates from heatmaps
    
    Args:
        heatmaps: Tensor of shape (B, N, H, W)
        threshold: Confidence threshold for keypoint detection
    
    Returns:
        Tuple of:
        - keypoints: Tensor of shape (B, N, 2) containing normalized coordinates
        - confidence: Tensor of shape (B, N) containing confidence scores
    """
    B, N, H, W = heatmaps.shape
    
    # Find peak locations
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_vals, max_idx = torch.max(heatmaps_flat, dim=2)
    
    # Convert indices to coordinates
    y_coord = (max_idx // W).float() / H
    x_coord = (max_idx % W).float() / W
    
    keypoints = torch.stack([x_coord, y_coord], dim=2)
    confidence = max_vals
    
    # Zero out low confidence detections
    mask = confidence > threshold
    confidence = confidence * mask
    keypoints = keypoints * mask.unsqueeze(-1)
    
    return keypoints, confidence

def compute_pck(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.2
) -> torch.Tensor:
    """Compute PCK (Percentage of Correct Keypoints) metric
    
    Args:
        pred: Predicted keypoints of shape (B, N, 2)
        target: Ground truth keypoints of shape (B, N, 2)
        threshold: Distance threshold relative to max image dimension
    
    Returns:
        PCK score for each keypoint type
    """
    distances = torch.norm(pred - target, dim=2)
    max_dim = torch.tensor([1.0], device=pred.device)  # Normalized coordinates
    threshold = threshold * max_dim
    
    correct = distances <= threshold
    pck = correct.float().mean(dim=0)
    
    return pck

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
