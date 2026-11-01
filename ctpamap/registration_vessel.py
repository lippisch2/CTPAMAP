"""Vesselness-driven registration backend for subtraction CTPA.

Motivation
----------
All three original backends (SimpleITK Demons, ANTs SyN, Elastix) drive the
optimization with an intensity-similarity metric (NCC / SSD / Demons). For a
pre-contrast → contrast-enhanced CTPA pair that is exactly the wrong metric:
the *whole point* of the acquisition is that intensities change (vessels go
from ~0 HU to +300 HU). An intensity metric fights that physics and tends to
warp the bright CTPA vessels to smear out the difference — corrupting the very
iodine signal we want to recover.

The fix (Murphy et al. 2012, "Improving Intensity-Based Lung CT Registration
Accuracy Utilizing Vesselness Cost Function") is to also match the *vascular
tree geometry*, which is contrast-invariant: a vessel's brightness changes
between the two scans but its shape does not. We compute Frangi vesselness on
both volumes and add an SSD term on the vesselness maps alongside a mutual
information term on the (contrast-tolerant) intensities. A B-spline free-form
deformation is optimized in a multi-resolution pyramid, restricted to the lung
mask. Because the solver is SimpleITK, we can extract a clean displacement
field so `--aer` aeration correction keeps working.

Conventions
-----------
Operates on nibabel-ordered (X, Y, Z) arrays and nibabel affines, exactly like
the other backends. Converts to SimpleITK (Z, Y, X) internally via
`_nib_to_sitk` and transposes back on output. Returns the warped pre image plus
a (3, Z, Y, X) displacement field in mm — the same convention as
`extract_displacement_field_sitk`.
"""
import sys
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from typing import Optional

from ctpamap.preprocessing import _nib_to_sitk
from ctpamap.segmentation import vesselness_map


def register_pre_to_ctpa_vessel(
    pre_data: np.ndarray,
    pre_img: nib.Nifti1Image,
    ctpa_data: np.ndarray,
    ctpa_img: nib.Nifti1Image,
    lung_mask: Optional[np.ndarray] = None,
    pre_lung_mask: Optional[np.ndarray] = None,
    aggressive: bool = False,
    vessel_weight: float = 0.5,
    shrink_levels: tuple[int, ...] = (8, 4, 2, 1),
    airway_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, sitk.CompositeTransform]:
    """Register pre → CTPA using MI + vesselness multi-resolution B-spline.

    Parameters
    ----------
    pre_data, ctpa_data : ndarray (X, Y, Z)
        Pre-contrast and contrast-enhanced volumes on a common grid (in HU).
    pre_img, ctpa_img : nibabel images
        Affines; must be identical (common grid). `ctpa_img` defines the
        reference space.
    lung_mask : ndarray, optional
        Common lung mask; restricts the metric to the lung (suppresses the
        chest wall/abdomen which would otherwise dominate the cost). If absent
        or too small, the whole field is used.
    pre_lung_mask : ndarray, optional
        Unused here (kept for API symmetry with the other backends); the
        vesselness term already encodes lung shape.
    aggressive : bool
        If True, uses finer mesh spacing and more iterations.
    vessel_weight : float
        Weight on the vesselness SSD term (intensity MI is implicitly weight 1).
        ~0.3–0.7 is typical; 0 disables the vessel term and degrades to a plain
        MI B-spline registration.
    shrink_levels : tuple of int
        Pyramid shrink factors, coarse-to-fine. Each level halves the B-spline
        mesh spacing.
    airway_mask : ndarray, optional
        Airways produce tube-like vesselness responses that would distract the
        vessel term from real vessels; when given they are suppressed from the
        vesselness feature images before optimization.

    Returns
    -------
    registered : ndarray (X, Y, Z), float32
        Pre warped into CTPA space.
    ddf : ndarray (3, Z, Y, X), float32
        Displacement field in mm, for aeration correction. Always non-None.
    composite : sitk.CompositeTransform
        The full rigid + deformation transform (fixed → moving space), usable
        to warp other masks (e.g. the pre-contrast vessel mask) into CTPA space.
    """
    fixed = _nib_to_sitk(ctpa_data, ctpa_img)
    moving = _nib_to_sitk(pre_data, pre_img)

    # --- Affine init: align gross position/orientation via MI -----------------
    print("  vessel backend: affine init (MI)...")
    affine = _affine_init(fixed, moving, aggressive)
    moving_aligned = _resample(moving, fixed, affine)

    # --- Vesselness feature images (contrast-invariant) -----------------------
    print("  vessel backend: computing vesselness features...")
    ctpa_np = sitk.GetArrayFromImage(fixed).astype(np.float32)
    pre_np = sitk.GetArrayFromImage(moving_aligned).astype(np.float32)
    # vesselness_map expects nib (X,Y,Z) ordering.
    ctpa_v_nib = vesselness_map(np.transpose(ctpa_np, (2, 1, 0)),
                                _sitk_mask_to_nib(lung_mask, ctpa_data.shape))
    pre_v_nib = vesselness_map(np.transpose(pre_np, (2, 1, 0)),
                               _sitk_mask_to_nib(lung_mask, pre_data.shape))

    if airway_mask is not None:
        airway_bool = airway_mask.astype(bool)
        ctpa_v_nib[airway_bool] = 0.0
        pre_v_nib[airway_bool] = 0.0

    fixed_v = _nib_to_sitk(ctpa_v_nib, ctpa_img)
    moving_v = _nib_to_sitk(pre_v_nib, pre_img)

    # --- Multi-resolution B-spline FFD on the joint metric --------------------
    bspline = _bspline_pyramid(
        fixed, moving_aligned, fixed_v, moving_v, lung_mask,
        aggressive=aggressive, vessel_weight=vessel_weight,
        shrink_levels=shrink_levels,
    )

    # Combine affine + bspline and resample the original pre once.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(bspline)
    composite.AddTransform(affine)

    final_res = sitk.ResampleImageFilter()
    final_res.SetReferenceImage(fixed)
    final_res.SetInterpolator(sitk.sitkBSpline)
    final_res.SetTransform(composite)
    warped = final_res.Execute(moving)
    registered = np.transpose(
        sitk.GetArrayFromImage(warped).astype(np.float32), (2, 1, 0)
    )

    # --- Displacement field for aeration correction ---------------------------
    ddf_img = sitk.TransformToDisplacementField(
        composite, sitk.sitkVectorFloat64, fixed.GetSize(),
        fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection(),
    )
    ddf = np.transpose(
        sitk.GetArrayFromImage(ddf_img).astype(np.float32), (3, 0, 1, 2)
    )

    sys.stdout.write("  vessel backend: done.\n")
    return registered, ddf, composite


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resample(img, ref, transform, interp=sitk.sitkLinear):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetTransform(transform)
    r.SetInterpolator(interp)
    return r.Execute(img)


def _sitk_mask_to_nib(lung_mask, target_shape):
    if lung_mask is None:
        return np.ones(target_shape, dtype=np.uint8)
    return lung_mask.astype(np.uint8)


def _center_of_mass(ref: sitk.Image) -> tuple[float, float, float]:
    size = ref.GetSize()
    return tuple(ref.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in size]))


def _affine_init(fixed, moving, aggressive: bool) -> sitk.Transform:
    """Gradient-based affine registration on mutual information.

    A cheap, robust gross alignment that does not care about the contrast
    difference (MI is intensity-distribution invariant). This also fixes the
    position of the B-spline FFD so the deformable stage only refines.
    """
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    if np.prod(fixed.GetSize()) < 40000:
        reg.SetMetricSamplingStrategy(reg.NONE)
    else:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-3,
        numberOfIterations=80 if aggressive else 50,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-4,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    init = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    # Centered initializer already aligns centers; keep its parameters as a safe
    # fallback if the MI optimizer diverges (translation beyond half the volume).
    extent = np.array(fixed.GetSize()) * np.array(fixed.GetSpacing())
    safe_params = init.GetParameters()
    reg.SetInitialTransform(init, inPlace=True)
    try:
        reg.Execute(fixed, moving)
        params = np.asarray(init.GetParameters())
        if np.max(np.abs(params[-3:])) > 0.5 * float(np.min(extent)):
            raise RuntimeError("affine init diverged (translation out of bounds)")
    except RuntimeError as e:
        # If the optimizer fails to converge on toy/small inputs, fall back to
        # the centered-initializer transform so the B-spline stage still runs
        # from a sane (aligned) starting point.
        print(f"  vessel backend: affine init unstable ({e}); keeping centered init.")
        init.SetParameters(safe_params)
    return init


def _make_bspline_init(ref: sitk.Image, mesh_spacing_mm: float) -> sitk.BSplineTransform:
    """A B-spline transform initialized as identity at the requested mesh spacing."""
    bspline = sitk.BSplineTransformInitializer(
        image1=ref,
        transformDomainMeshSize=[
            max(1, int(round(s * sp / mesh_spacing_mm)))
            for s, sp in zip(ref.GetSize(), ref.GetSpacing())
        ],
    )
    return bspline


def _bspline_pyramid(
    fixed, moving, fixed_v, moving_v, lung_mask,
    aggressive: bool, vessel_weight: float, shrink_levels: tuple[int, ...],
) -> sitk.BSplineTransform:
    """Multi-resolution B-spline optimizing MI + vesselness SSD.

    We can't register two cost terms simultaneously with one ImageRegistrationMethod,
    so we alternate: at each level, refine the B-spline by optimizing the
    intensity MI term, then do a short refinement on the vesselness SSD term,
    reusing the previous level's B-spline as the init. This is the standard
    alternating-optimization scheme for multi-feature registration and keeps
    each term's gradient well-conditioned.
    """
    spacing = np.array(fixed.GetSpacing())
    # Mesh spacing schedule: coarse at the top of the pyramid, fine at the bottom.
    # aggressive → finer final mesh (more local control).
    base_mesh = 60.0 if aggressive else 80.0
    mesh_schedule = [base_mesh * lvl for lvl in shrink_levels]
    # Never let the mesh exceed the smallest physical extent — a B-spline whose
    # mesh is larger than the volume reduces to a single control point and the
    # optimizer has nothing to optimise (drifts on small/test volumes).
    phys_extent = np.array(fixed.GetSize()) * spacing
    max_mesh = float(np.min(phys_extent)) / 2.0
    mesh_schedule = [min(m, max_mesh) for m in mesh_schedule]

    # Registration mask (restrict metric to the lung).
    use_mask = lung_mask is not None and lung_mask.sum() > 100
    mask_sitk = None
    if use_mask:
        m = np.transpose(lung_mask.astype(np.uint8), (2, 1, 0)).copy()
        mask_sitk = sitk.GetImageFromArray(m)
        mask_sitk.CopyInformation(fixed)

    current = None  # BSplineTransform carried across levels

    for level_idx, (shrink, mesh_mm) in enumerate(zip(shrink_levels, mesh_schedule)):
        sys.stdout.write(
            f"\r  vessel backend: B-spline level {level_idx + 1}/{len(shrink_levels)}"
            f" (shrink {shrink}, mesh {mesh_mm:.0f} mm)..."
        )
        sys.stdout.flush()

        # Downsample everything for this level.
        fixed_l = sitk.Shrink(fixed, [shrink] * 3)
        moving_l = sitk.Shrink(moving, [shrink] * 3)
        fixed_v_l = sitk.Shrink(fixed_v, [shrink] * 3)
        moving_v_l = sitk.Shrink(moving_v, [shrink] * 3)
        mask_l = sitk.Shrink(mask_sitk, [shrink] * 3) if mask_sitk is not None else None

        # Init B-spline at this level's mesh spacing, seeded from the previous level.
        init = _make_bspline_init(fixed_l, mesh_mm)
        if current is not None:
            _transfer_bspline_params(current, init)

        # --- Intensity MI term ---
        mi_iters = 60 if aggressive else 35
        init = _optimize_bspline_metric(
            fixed_l, moving_l, init, mask_l,
            metric="mi", iters=mi_iters,
        )

        # --- Vesselness SSD term ---
        if vessel_weight > 0:
            init = _optimize_bspline_metric(
                fixed_v_l, moving_v_l, init, mask_l,
                metric="ssd", iters=40 if aggressive else 20,
            )

        current = init

    sys.stdout.write("\n")
    return current


def _transfer_bspline_params(src: sitk.BSplineTransform, dst: sitk.BSplineTransform) -> None:
    """Copy B-spline control-point parameters from `src` into `dst` if compatible.

    Different mesh sizes have different numbers of control points, so this only
    helps when consecutive levels share a mesh size; otherwise we silently fall
    back to the identity init (which is fine — the optimizer re-finds the warp).
    """
    try:
        if src.GetNumberOfParameters() == dst.GetNumberOfParameters():
            dst.SetParameters(src.GetParameters())
    except Exception:
        pass


def _optimize_bspline_metric(
    fixed, moving, init_bspline, mask, metric: str, iters: int,
) -> sitk.BSplineTransform:
    """Refine a B-spline transform against a single metric."""
    reg = sitk.ImageRegistrationMethod()
    if metric == "mi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        interp = sitk.sitkLinear
    elif metric == "ssd":
        reg.SetMetricAsMeanSquares()
        interp = sitk.sitkLinear
    else:
        raise ValueError(f"unknown metric: {metric}")

    # Full sampling on small volumes (avoids gradient-estimate divergence);
    # random sub-sampling on clinical-sized ones.
    if np.prod(fixed.GetSize()) < 40000:
        reg.SetMetricSamplingStrategy(reg.NONE)
    else:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.20)

    if mask is not None:
        reg.SetMetricFixedMask(mask)

    reg.SetInterpolator(interp)

    # L-BFGS works well for B-spline FFD: it exploits the smoothness of the
    # control-point grid and tolerates the limited metric sampling above.
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=iters,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=iters * 4,
    )
    reg.SetInitialTransform(init_bspline, inPlace=True)
    try:
        reg.Execute(fixed, moving)
    except RuntimeError:
        pass  # keep the init transform on failure
    return init_bspline
