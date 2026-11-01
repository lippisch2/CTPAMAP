import sys
import os
import numpy as np
import nibabel as nib
from typing import Optional
import torch
from torch.nn import functional as F

def register_pre_to_ctpa_fireants(
    pre_data: np.ndarray,
    pre_img: nib.Nifti1Image,
    ctpa_data: np.ndarray,
    ctpa_img: nib.Nifti1Image,
    lung_mask: Optional[np.ndarray] = None,
    pre_lung_mask: Optional[np.ndarray] = None,
    aggressive: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import SimpleITK as sitk
        from fireants.io.image import Image, BatchedImages
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
    except ImportError as e:
        raise ImportError(f"fireants not available: {e}. Install with: pip install fireants")

    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  FireANTs device: {device}")

    def _nib_to_fireants(data: np.ndarray, img: nib.Nifti1Image) -> Image:
        sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)).copy())
        affine = img.affine
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        direction = affine[:3, :3] / spacing
        origin = affine[:3, 3]
        sitk_img.SetSpacing(spacing.tolist())
        sitk_img.SetDirection(direction.flatten().tolist())
        sitk_img.SetOrigin(origin.tolist())
        return Image(sitk_img, device=device)

    print("  Converting volumes to FireANTs images...")
    fixed_img = _nib_to_fireants(ctpa_data, ctpa_img)
    moving_img = _nib_to_fireants(pre_data, pre_img)

    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    scales = [4, 2, 1]
    if aggressive:
        iters_affine = [300, 150, 75]
        iters_deform = [300, 150, 75]
    else:
        iters_affine = [200, 100, 50]
        iters_deform = [200, 100, 50]

    print("  Stage 1: affine registration...")
    affine = AffineRegistration(
        scales=scales, iterations=iters_affine,
        fixed_images=fixed_batch, moving_images=moving_batch,
        loss_type='cc', cc_kernel_size=5,
        optimizer='Adam', optimizer_lr=3e-3,
    )
    affine.optimize()
    init_affine = affine.get_affine_matrix().detach()
    del affine

    print("  Stage 2: deformable registration (greedy compositive)...")
    reg = GreedyRegistration(
        scales=scales, iterations=iters_deform,
        fixed_images=fixed_batch, moving_images=moving_batch,
        loss_type='cc', cc_kernel_size=5,
        deformation_type='compositive',
        optimizer='Adam', optimizer_lr=0.5,
        init_affine=init_affine,
    )
    reg.optimize()

    print("  Computing warped image...")
    with torch.no_grad():
        moved = reg.evaluate(fixed_batch, moving_batch)
    warped_np = moved[0, 0].cpu().numpy().astype(np.float32)

    ddf = None
    if lung_mask is not None and lung_mask.sum() > 100:
        print("  Extracting displacement field...")
        with torch.no_grad():
            warped_norm = reg.get_warped_coordinates(fixed_batch, moving_batch)
            B = warped_norm.shape[0]
            spatial = list(warped_norm.shape[1:-1])
            identity = torch.eye(3, 4, device=warped_norm.device).unsqueeze(0).expand(B, -1, -1)
            orig_norm = F.affine_grid(identity, [B, 1, *spatial], align_corners=True)
            disp_norm = warped_norm - orig_norm
            torch2phy = fixed_batch.get_torch2phy()
            R = torch2phy[0, :3, :3]
            disp_phys = torch.einsum('ij,...j->...i', R, disp_norm)
            ddf = disp_phys.permute(3, 0, 1, 2).cpu().numpy().astype(np.float32)

    del reg, fixed_batch, moving_batch, fixed_img, moving_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  FireANTs: done ({warped_np.shape})")
    return warped_np, ddf
