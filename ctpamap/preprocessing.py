import numpy as np
import SimpleITK as sitk
import nibabel as nib


def reorient_to_ras(data: np.ndarray, img: nib.Nifti1Image) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Reorient voxels to RAS (closest canonical).

    Builds a fresh image from the *passed* `data` (which may already have been
    modified by the caller) before reorienting, so the returned array always
    reflects `data` rather than silently re-reading the original voxels.
    """
    src = nib.Nifti1Image(data, img.affine, header=img.header)
    img_ras = nib.as_closest_canonical(src)
    data_ras = np.asanyarray(img_ras.dataobj).astype(np.float32)
    return data_ras, img_ras


def _nib_to_sitk(data: np.ndarray, img: nib.Nifti1Image) -> sitk.Image:
    nib_shape = data.shape
    sitk_data = np.transpose(data, (2, 1, 0)).copy()
    sitk_img = sitk.GetImageFromArray(sitk_data)
    affine = img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = affine[:3, :3] / spacing
    origin = affine[:3, 3]
    sitk_img.SetSpacing(spacing.tolist())
    sitk_img.SetDirection(direction.flatten().tolist())
    sitk_img.SetOrigin(origin.tolist())
    return sitk_img


def _sitk_to_nib(sitk_img: sitk.Image) -> tuple[np.ndarray, nib.Nifti1Image]:
    sitk_data = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    data = np.transpose(sitk_data, (2, 1, 0))
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    origin = np.array(sitk_img.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    new_img = nib.Nifti1Image(data, affine)
    return data, new_img


def resample_to_spacing(
    data: np.ndarray, img: nib.Nifti1Image, target_spacing: tuple[float, float, float]
) -> tuple[np.ndarray, nib.Nifti1Image]:
    sitk_img = _nib_to_sitk(data, img)
    original_size = sitk_img.GetSize()
    original_spacing = sitk_img.GetSpacing()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(sitk_img)
    return _sitk_to_nib(resampled)


def resample_to_common_grid(
    data1: np.ndarray, img1: nib.Nifti1Image,
    data2: np.ndarray, img2: nib.Nifti1Image,
    target_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, nib.Nifti1Image, np.ndarray, nib.Nifti1Image]:
    sitk1 = _nib_to_sitk(data1, img1)
    sitk2 = _nib_to_sitk(data2, img2)

    bounds1 = np.array([sitk1.TransformIndexToPhysicalPoint([0, 0, 0]),
                        sitk1.TransformIndexToPhysicalPoint([s - 1 for s in sitk1.GetSize()])])
    bounds2 = np.array([sitk2.TransformIndexToPhysicalPoint([0, 0, 0]),
                        sitk2.TransformIndexToPhysicalPoint([s - 1 for s in sitk2.GetSize()])])

    origin = np.minimum(bounds1[0], bounds2[0])
    far = np.maximum(bounds1[1], bounds2[1])
    new_size = [max(2, int(round((far[i] - origin[i]) / target_spacing[i])) + 1) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(origin.tolist())
    resampler.SetOutputDirection(np.eye(3).flatten().tolist())
    resampler.SetInterpolator(sitk.sitkLinear)

    r1 = resampler.Execute(sitk1)
    r2 = resampler.Execute(sitk2)

    out1_data, _ = _sitk_to_nib(r1)
    out2_data, _ = _sitk_to_nib(r2)

    affine = np.eye(4)
    affine[:3, :3] = np.diag(target_spacing)
    affine[:3, 3] = origin
    out_img = nib.Nifti1Image(out1_data, affine)
    out_img.header.set_zooms(target_spacing)

    return out1_data, out_img, out2_data, out_img


def clip_hu(data: np.ndarray, low: float = -1000, high: float = 1500) -> np.ndarray:
    return np.clip(data, low, high)


def light_denoise(data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    sitk_data = np.transpose(data, (2, 1, 0)).copy()
    sitk_img = sitk.GetImageFromArray(sitk_data)
    smoothed = sitk.CurvatureFlow(sitk_img, timeStep=0.125, numberOfIterations=5)
    result = sitk.GetArrayFromImage(smoothed).astype(np.float32)
    return np.transpose(result, (2, 1, 0))
