import os
import numpy as np
import nibabel as nib
from typing import Optional

try:
    import ants
except ImportError:
    ants = None

# These functions mirror the SimpleITK backend's convention: the rest of the
# pipeline works with nibabel-ordered arrays (X, Y, Z) and nibabel affines. ANTs
# expects images in (Z, Y, X) ordering with explicit spacing/origin/direction,
# so we must convert at the boundary exactly like _nib_to_sitk / _sitk_to_nib
# do. Failing to transpose (as the original did) produced axis-swapped output
# relative to the SimpleITK and elastix backends.


def _nib_to_ants(data: np.ndarray, img: nib.Nifti1Image):
    if ants is None:
        raise ImportError("ants is not installed. Install with: pip install antspyx")

    affine = img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = (affine[:3, :3] / spacing).flatten().tolist()
    origin = affine[:3, 3].tolist()
    spacing = spacing.tolist()

    # nibabel/numpy order is (X, Y, Z); ANTs image arrays are (Z, Y, X).
    arr = np.transpose(data.astype(np.float32), (2, 1, 0)).copy()
    img_ants = ants.from_numpy(
        arr,
        spacing=spacing,
        origin=origin,
        direction=direction,
        has_components=False,
        is_rgb=False,
    )
    return img_ants


def _ants_to_nib_np(image) -> np.ndarray:
    """Convert an ANTs image back to a nibabel-ordered (X, Y, Z) numpy array."""
    arr = image.numpy().astype(np.float32)
    # ANTs array is (Z, Y, X) -> back to (X, Y, Z).
    return np.transpose(arr, (2, 1, 0)).copy()


def register_pre_to_ctpa_antspy(
    pre_data: np.ndarray,
    pre_img: nib.Nifti1Image,
    ctpa_data: np.ndarray,
    ctpa_img: nib.Nifti1Image,
    lung_mask: Optional[np.ndarray] = None,
    pre_lung_mask: Optional[np.ndarray] = None,
    aggressive: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if ants is None:
        raise ImportError("ants is not installed. Install with: pip install antspyx")

    fixed = _nib_to_ants(ctpa_data, ctpa_img)
    moving = _nib_to_ants(pre_data, pre_img)

    mask_obj = None
    if lung_mask is not None and lung_mask.sum() > 100:
        mask_obj = _nib_to_ants(lung_mask.astype(np.float32), ctpa_img)
        mask_obj = mask_obj.threshold_image(0.5, 1e9)

    print("  backend: ANTsPy SyN registration")
    print(f"  fixed shape:  {fixed.shape}, spacing: {fixed.spacing}")
    print(f"  moving shape: {moving.shape}, spacing: {moving.spacing}")
    if mask_obj:
        print(f"  mask shape:   {mask_obj.shape}, spacing: {mask_obj.spacing}")

    n_threads = os.cpu_count() or 1
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n_threads)
    os.environ["ANTS_NUMBER_OF_THREADS"] = str(n_threads)
    print(f"  threads: {n_threads}")

    result = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='SyNRA',
        mask=mask_obj,
        verbose=False,
    )

    warped = result['warpedmovout']
    # Convert back to nibabel ordering so this array aligns with ctpa_data.
    warped_np = _ants_to_nib_np(warped)
    ddf = _extract_ddf(result)
    print(f"  ANTsPy: done ({warped_np.shape})")

    return warped_np, ddf


def _extract_ddf(result: dict) -> Optional[np.ndarray]:
    """Extract displacement field (mm) from ANTsPy registration result.

    ANTs warp images are stored components-last, i.e. shape (Z, Y, X, 3). We
    transpose to (3, Z, Y, X), matching the DDF convention used by the rest of
    the pipeline (extract_displacement_field_sitk, vessel, fireants backends).
    """
    try:
        for p in result.get('fwdtransforms', []):
            if p.endswith('.nii.gz'):
                warp_img = ants.image_read(p)
                arr = warp_img.numpy()
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    # (Z, Y, X, 3) -> (3, Z, Y, X)
                    return np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)
        return None
    except Exception:
        return None
