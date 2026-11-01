import sys
import numpy as np
import nibabel as nib
from typing import Optional

OUTSIDE_VAL = -1000.0


def register_pre_to_ctpa_elastix(
    pre_data: np.ndarray,
    pre_img: nib.Nifti1Image,
    ctpa_data: np.ndarray,
    ctpa_img: nib.Nifti1Image,
    lung_mask: Optional[np.ndarray] = None,
    pre_lung_mask: Optional[np.ndarray] = None,
    aggressive: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import itk
    except ImportError:
        raise ImportError(
            "itk-elastix is not installed. Install with: pip install itk-elastix"
        )

    mask_array = None
    if lung_mask is not None and lung_mask.sum() > 100:
        mask_array = lung_mask.astype(bool)
        print(f"  using lung mask with {int(mask_array.sum())} voxels")

    if mask_array is not None:
        ctpa_masked = np.where(mask_array, ctpa_data, OUTSIDE_VAL).astype(np.float32)
        pre_masked = np.where(mask_array, pre_data, OUTSIDE_VAL).astype(np.float32)
    else:
        ctpa_masked = ctpa_data.astype(np.float32)
        pre_masked = pre_data.astype(np.float32)

    fixed = _nib_to_itk(ctpa_masked, ctpa_img)
    moving = _nib_to_itk(pre_masked, pre_img)

    print(f"  backend: ITK-Elastix rigid+affine{' + BSpline' if aggressive else ''} registration")
    print(f"  fixed shape:  {itk.GetArrayFromImage(fixed).shape}")
    print(f"  moving shape: {itk.GetArrayFromImage(moving).shape}")

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap('rigid', 4)
    )
    parameter_object.AddParameterMap(
        parameter_object.GetDefaultParameterMap('affine', 4)
    )
    if aggressive:
        parameter_object.AddParameterMap(
            parameter_object.GetDefaultParameterMap('bspline', 4)
        )

    sys.stdout.write("\r  elastix registration...")
    sys.stdout.flush()
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed, moving,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    result_np = _itk_to_nib(result_image)
    print(f"\r  elastix: done ({result_np.shape})  ")

    if mask_array is not None:
        result_np[~mask_array] = pre_data[~mask_array]

    return result_np, None


def _nib_to_itk(data: np.ndarray, img: nib.Nifti1Image):
    import itk

    sitk_data = np.transpose(data.astype(np.float32), (2, 1, 0)).copy()
    itk_img = itk.GetImageFromArray(sitk_data)

    affine = img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = affine[:3, :3] / spacing
    origin = affine[:3, 3]

    itk_img.SetSpacing(spacing.tolist())
    itk_img.SetDirection(itk.GetMatrixFromArray(direction))
    itk_img.SetOrigin(origin.tolist())

    return itk_img


def _itk_to_nib(itk_img) -> np.ndarray:
    sitk_data = itk.GetArrayFromImage(itk_img).astype(np.float32)
    data = np.transpose(sitk_data, (2, 1, 0))
    return data
