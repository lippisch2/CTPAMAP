import sys
import numpy as np
import SimpleITK as sitk
from typing import Optional
import nibabel as nib
from ctpamap.preprocessing import _nib_to_sitk


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return 0.0 if d < 1e-10 else float(np.sum(a * b) / d)


def _mask_sitk_from_numpy(mask_np: np.ndarray, reference: sitk.Image) -> sitk.Image:
    m = np.transpose(mask_np.astype(np.uint8), (2, 1, 0)).copy()
    s = sitk.GetImageFromArray(m)
    s.CopyInformation(reference)
    return s


def _apply_transform(
    img: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetInterpolator(interpolator)
    r.SetTransform(transform)
    return r.Execute(img)


def _demons_disp(
    fixed: sitk.Image,
    moving: sitk.Image,
    iterations: int = 40,
    std: float = 2.0,
) -> sitk.Image:
    d = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    d.SetNumberOfIterations(iterations)
    d.SetStandardDeviations(std)
    disp = d.Execute(fixed, moving)
    # Demons returns sitkVectorFloat32; DisplacementFieldTransform needs sitkVectorFloat64
    if disp.GetPixelID() != sitk.sitkVectorFloat64:
        disp = sitk.Cast(disp, sitk.sitkVectorFloat64)
    return disp


def _disp_tfm(disp: sitk.Image) -> sitk.DisplacementFieldTransform:
    if disp.GetPixelID() != sitk.sitkVectorFloat64:
        # Convert via numpy
        arr = sitk.GetArrayFromImage(disp).astype(np.float64)
        new_disp = sitk.GetImageFromArray(arr, isVector=True)
        new_disp.CopyInformation(disp)
        disp = new_disp
    return sitk.DisplacementFieldTransform(disp)


def _z_pre_scan(
    fixed_d: sitk.Image,
    moving_d: sitk.Image,
    center_d: tuple,
    search_mm: tuple = (-12, 12, 3),
) -> float:
    """Coarse z-translation scan on a downsampled pair.

    Returns the best z translation (mm) seeding the NCC optimizer when the pre
    and CTPA volumes are expected to differ by several whole slices (--rigid-z).
    Only ~9 resamples on the already-downsampled grid, so negligible cost.
    """
    f_np = sitk.GetArrayFromImage(fixed_d).astype(np.float32)
    best_dz, best_ncc = 0.0, -1.0
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(fixed_d)
    res.SetInterpolator(sitk.sitkLinear)
    for dz in range(search_mm[0], search_mm[1] + 1, search_mm[2]):
        t = sitk.Euler3DTransform()
        t.SetCenter(center_d)
        t.SetParameters([0.0, 0.0, 0.0, 0.0, 0.0, float(dz)])
        res.SetTransform(t)
        w = sitk.GetArrayFromImage(res.Execute(moving_d)).astype(np.float32)
        v = _ncc(f_np, w)
        if v > best_ncc:
            best_ncc, best_dz = v, dz
    return best_dz


def _rigid_align_sitk(
    fixed: sitk.Image,
    moving: sitk.Image,
    center: tuple,
    aggressive: bool = False,
    shrink: int = 4,
    z_pre_search: bool = False,
) -> sitk.Euler3DTransform:
    """Rigid (Euler 3D) alignment via multi-resolution NCC.

    Replaces the old brute-force NCC grid loop (up to 7x7x7 resamples in a
    Python loop) with a coarse-to-fine gradient optimizer that converges to
    sub-voxel accuracy. `z_pre_search` additionally seeds the optimizer with a
    coarse z translation scan for scans that differ by several whole slices.

    Returns an Euler3DTransform with its parameters expressed around the
    full-resolution centre.
    """
    if shrink > 1:
        fixed_d = sitk.Shrink(fixed, [shrink] * 3)
        moving_d = sitk.Shrink(moving, [shrink] * 3)
        center_d = tuple(fixed_d.TransformContinuousIndexToPhysicalPoint(
            [(s - 1) / 2.0 for s in fixed_d.GetSize()]))
    else:
        fixed_d, moving_d, center_d = fixed, moving, center

    print(f"  rigid NCC alignment ({fixed_d.GetSize()}{', z pre-scan' if z_pre_search else ''})")

    init = sitk.Euler3DTransform()
    init.SetCenter(center_d)
    if z_pre_search:
        best_dz = _z_pre_scan(fixed_d, moving_d, center_d)
        init.SetTranslation((0.0, 0.0, best_dz))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsANTSNeighborhoodCorrelation(radius=6)
    # Random sampling can produce an empty point set on toy-sized inputs; fall
    # back to full sampling below e.g. 64^3 voxels so small/test volumes work.
    if np.prod(fixed_d.GetSize()) >= 4096:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([max(1, shrink), 2, 1])
    reg.SetSmoothingSigmasPerLevel([max(0.5, shrink / 2.0), 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsPowell(
        numberOfIterations=120 if aggressive else 60,
        stepLength=0.2,
        stepTolerance=1e-6,
        valueTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=True)
    try:
        reg.Execute(fixed_d, moving_d)
        params = init.GetParameters()  # optimized in place
    except Exception as e:
        print(f"  WARNING: rigid NCC alignment failed ({e}); using identity (Demons will try to correct).")
        params = [0.0] * 6

    tfm = sitk.Euler3DTransform()
    tfm.SetCenter(center)
    tfm.SetParameters(params)
    print("  rigid alignment done.")
    return tfm


def register_pre_to_ctpa(
    pre_data: np.ndarray,
    pre_img: nib.Nifti1Image,
    ctpa_data: np.ndarray,
    ctpa_img: nib.Nifti1Image,
    lung_mask: Optional[np.ndarray] = None,
    pre_lung_mask: Optional[np.ndarray] = None,
    aggressive: bool = False,
    lite: bool = False,
    rigid_shrink: int = 4,
    rigid_z_search: bool = False,
) -> tuple[np.ndarray, nib.Nifti1Image, sitk.Euler3DTransform, sitk.CompositeTransform]:
    """Register pre -> CTPA.

    Pipeline:
      1. Rigid NCC alignment via a multi-resolution ImageRegistrationMethod
         optimizer on a downsampled grid (`rigid_z_search` adds a z pre-scan).
      2. Intensity Demons deformable refinement (lung-masked).
      3. (aggressive) SDF lung-boundary alignment.

    `rigid_shrink` controls the downsampling factor for step 1.
    """
    fixed = _nib_to_sitk(ctpa_data, ctpa_img)
    moving = _nib_to_sitk(pre_data, pre_img)

    center = tuple(
        fixed.TransformContinuousIndexToPhysicalPoint(
            [(s - 1) / 2.0 for s in fixed.GetSize()]
        )
    )

    # === Step 1: Rigid NCC alignment (multi-resolution optimizer) ===
    shrink = max(1, 6 if (aggressive and lite) else rigid_shrink)
    rigid_tfm = _rigid_align_sitk(
        fixed, moving, center,
        aggressive=aggressive, shrink=shrink, z_pre_search=rigid_z_search,
    )

    # Intermediate: pre warped by rigid (needed for Demons cost)
    pre_rigid = _apply_transform(moving, fixed, rigid_tfm)

    # === Step 2: Deformable refinement (intensity Demons) ===
    if lung_mask is not None and lung_mask.sum() > 100:
        msk = _mask_sitk_from_numpy(lung_mask, fixed)
        mf = sitk.MaskImageFilter()
        mf.SetOutsideValue(-1000.0)
        masked_fixed = mf.Execute(fixed, msk)
        masked_moving = mf.Execute(pre_rigid, msk)

        demon_iters = 80 if aggressive else 40
        demon_std = 1.5 if aggressive else 2.0
        sys.stdout.write("\r  deformable registration...")
        sys.stdout.flush()
        disp_intensity = _demons_disp(masked_fixed, masked_moving, demon_iters, demon_std)
        tfm_intensity = _disp_tfm(disp_intensity)
        used_mask = True
    else:
        used_mask = False

    # === Step 3 (aggressive): lung-boundary SDF alignment ===
    if aggressive and used_mask:
        if pre_lung_mask is not None and pre_lung_mask.sum() > 50:
            # Warp pre lung mask to fixed space (through rigid + first Demons)
            msk_pre = sitk.CompositeTransform(3)
            msk_pre.AddTransform(tfm_intensity)
            msk_pre.AddTransform(rigid_tfm)

            pre_lung_sitk = _mask_sitk_from_numpy(pre_lung_mask, moving)
            pre_lung_in_fixed = _apply_transform(
                sitk.Cast(pre_lung_sitk, sitk.sitkFloat32),
                fixed, msk_pre,
                interpolator=sitk.sitkNearestNeighbor,
            )
            pre_lung_in_fixed = sitk.Cast(pre_lung_in_fixed > 0.5, sitk.sitkUInt8)

            # SDF of CTPA lung mask
            sdf_fixed = sitk.SignedMaurerDistanceMap(
                msk, insideIsPositive=True, squaredDistance=False
            )
            # SDF of warped pre lung mask
            sdf_moving = sitk.SignedMaurerDistanceMap(
                pre_lung_in_fixed, insideIsPositive=True, squaredDistance=False
            )

            sdf_norm = 50.0
            blend_alpha = 0.3

            mf2 = sitk.MaskImageFilter()
            mf2.SetOutsideValue(-1000.0)

            # Intermediate: pre warped by rigid + intensity Demons (for SDF Demons cost)
            pre_stage1 = _apply_transform(pre_rigid, fixed, tfm_intensity)

            bf = mf2.Execute(fixed, msk)
            bm = mf2.Execute(pre_stage1, msk)
            bf = bf + sitk.Cast(sdf_fixed / sdf_norm, sitk.sitkFloat32) * 200.0 * blend_alpha
            bm = bm + sitk.Cast(sdf_moving / sdf_norm, sitk.sitkFloat32) * 200.0 * blend_alpha

            sys.stdout.write("\r  SDF lung-boundary alignment...")
            sys.stdout.flush()
            disp_sdf = _demons_disp(bf, bm, 200 if aggressive else 100, 1.0)
            tfm_sdf = _disp_tfm(disp_sdf)
            used_sdf = True
        else:
            used_sdf = False
    else:
        used_sdf = False

    # === Build composite transform for a single final resample from original moving ===
    composite = sitk.CompositeTransform(3)

    # Transforms are applied in the order they are added.
    # Displacement maps fixed → fixed+disp (within fixed space).
    # rigid_tfm maps fixed → pre space.
    # composite = rigid ∘ dispₙ ∘ ... ∘ disp₁  => maps fixed → pre

    if used_sdf:
        composite.AddTransform(tfm_sdf)
    if used_mask:
        composite.AddTransform(tfm_intensity)
    composite.AddTransform(rigid_tfm)

    # Single resample from original moving with BSpline for sharpness
    final_res = sitk.ResampleImageFilter()
    final_res.SetReferenceImage(fixed)
    final_res.SetInterpolator(sitk.sitkBSpline)
    final_res.SetTransform(composite)
    warped = final_res.Execute(moving)

    # Convert back to nibabel
    warped_sitk = sitk.GetArrayFromImage(warped).astype(np.float32)
    registered_data = np.transpose(warped_sitk, (2, 1, 0))

    sys.stdout.write("\r  registration complete.           \n")
    sys.stdout.flush()
    return registered_data, ctpa_img, rigid_tfm, composite


def extract_displacement_field_sitk(
    composite: sitk.CompositeTransform,
    reference: sitk.Image,
) -> np.ndarray:
    """Extract displacement field (mm) from a SimpleITK composite transform.

    Returns a (3, Z, Y, X) float array in physical units (mm) — components
    ordered x, y, z, in sitk voxel order. This matches the DDF convention of
    the antspy, vessel, and fireants backends so callers can handle any backend
    uniformly.
    """
    ddf = sitk.TransformToDisplacementField(
        composite, sitk.sitkVectorFloat64, reference.GetSize(),
        reference.GetOrigin(), reference.GetSpacing(), reference.GetDirection(),
    )
    arr = sitk.GetArrayFromImage(ddf).astype(np.float32)
    return np.transpose(arr, (3, 0, 1, 2))
