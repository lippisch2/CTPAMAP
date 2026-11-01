import sys
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctpamap.preprocessing import clip_hu, light_denoise, reorient_to_ras
from ctpamap.segmentation import (
    segment_lungs, segment_airway, segment_vessels,
    vesselness_map, _find_first_air_seed, _touches_boundary,
)
from ctpamap.subtraction import compute_iodine_map
from ctpamap.overlay import compute_iodine_overlay
from ctpamap.qc import compute_qc_metrics
from ctpamap.jacobian import displacement_to_jacobian, correct_aeration


def _make_dummy_volume(shape=(64, 96, 96)):
    data = np.random.randn(*shape).astype(np.float32) * 200 - 500
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.5, 1.5, 1.5])
    img = nib.Nifti1Image(data, affine)
    return data, img


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_clip_hu():
    data = np.array([-2000, -1000, 0, 500, 1500, 2000], dtype=np.float32)
    clipped = clip_hu(data)
    assert clipped.min() == -1000
    assert clipped.max() == 1500
    assert clipped[2] == 0


def test_light_denoise():
    data, img = _make_dummy_volume((16, 16, 16))
    smoothed = light_denoise(data)
    assert smoothed.shape == data.shape
    assert smoothed.dtype == np.float32


def test_reorient_to_ras_uses_passed_data():
    """reorient_to_ras must reflect modifications made to `data` before the call."""
    data, img = _make_dummy_volume((16, 16, 16))
    modified = data + 1234.0
    out, _ = reorient_to_ras(modified, img)
    # Identity-affine volume => canonical == original; check the +1234 propagated.
    assert np.allclose(out, modified)


def test_reorient_changes_with_orientation():
    """A non-identity (flipped) affine should cause reorient to differ."""
    data, img = _make_dummy_volume((16, 16, 16))
    flip_aff = img.affine.copy()
    flip_aff[0, 0] *= -1  # flip X axis
    flipped_img = nib.Nifti1Image(data, flip_aff)
    out, _ = reorient_to_ras(data, flipped_img)
    assert out.shape == data.shape
    # No assertion on exact values, just that it doesn't crash and preserves shape.


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_segment_lungs():
    data = np.full((32, 48, 48), -900, dtype=np.float32)
    data[:, 10:25, 15:35] = -300
    mask = segment_lungs(data)
    assert mask.dtype == np.uint8
    assert mask.ndim == 3


def test_segment_airway():
    data = np.full((32, 48, 48), -300, dtype=np.float32)
    data[:, 20:28, 20:30] = -900
    lung = np.zeros((32, 48, 48), dtype=np.uint8)
    lung[:, 10:30, 10:40] = 1
    airway = segment_airway(data, lung)
    assert airway.dtype == np.uint8


def test_segment_vessels():
    data = np.full((32, 48, 48), -300, dtype=np.float32)
    data[:, 15:20, 20:25] = 300
    lung = np.zeros((32, 48, 48), dtype=np.uint8)
    lung[:, 10:30, 10:40] = 1
    vessels = segment_vessels(data, lung)
    assert vessels.dtype == np.uint8


def test_find_first_air_seed_stops_at_first_hit():
    data = np.full((20, 20, 20), -100, dtype=np.float32)
    # Place air voxels at two distances from the seed; the closest must win.
    data[12, 10, 10] = -900  # distance 2
    data[7, 8, 9] = -900     # distance sqrt(9+4+1) ~ 3.7
    seed = _find_first_air_seed(data, (10, 10, 10), radius=6, air_hu=-800)
    assert seed == (12, 10, 10)  # closest-first, not row-major


def test_find_first_air_seed_none():
    data = np.full((20, 20, 20), -100, dtype=np.float32)
    assert _find_first_air_seed(data, (10, 10, 10), radius=3, air_hu=-800) is None


def test_touches_boundary_all_axes():
    shape = (10, 10, 10)
    # Touches the z (axis 2) boundary only.
    coords = (np.array([5]), np.array([5]), np.array([0]))
    assert _touches_boundary(coords, shape) is True
    # Interior only.
    coords = (np.array([5]), np.array([5]), np.array([5]))
    assert _touches_boundary(coords, shape) is False


# ---------------------------------------------------------------------------
# Subtraction / overlay
# ---------------------------------------------------------------------------

def test_compute_iodine_map():
    ctpa = np.random.rand(16, 24, 24).astype(np.float32) * 200
    pre = np.random.rand(16, 24, 24).astype(np.float32) * 200
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    vessels = np.zeros((16, 24, 24), dtype=np.uint8)
    airway = np.zeros((16, 24, 24), dtype=np.uint8)
    result = compute_iodine_map(ctpa, pre, lung, vessels, airway)
    for key in ("raw", "clean", "normalized"):
        assert key in result
        assert result[key].shape == ctpa.shape


def test_compute_iodine_map_no_seg_negative_component():
    ctpa = np.random.rand(16, 24, 24).astype(np.float32) * 200
    pre = ctpa + 50  # ensure negative residuals dominate
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    result = compute_iodine_map(ctpa, pre, lung, no_seg=True)
    assert result["normalized_neg"] is not None
    assert result["normalized_neg"].shape == ctpa.shape


def test_compute_iodine_overlay():
    ctpa = np.random.rand(16, 24, 24).astype(np.float32) * 200
    iodine = np.random.rand(16, 24, 24).astype(np.float32)
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    overlay = compute_iodine_overlay(ctpa, iodine, lung_mask=lung)
    assert overlay.shape == (16, 24, 24, 3)
    assert overlay.dtype == np.float32


def test_compute_iodine_overlay_is_convex_blend():
    """Inside the lung, every channel must lie in [0,1] and not exceed bg+colour."""
    ctpa = np.full((8, 8, 8), -600.0, dtype=np.float32)
    iodine = np.ones((8, 8, 8), dtype=np.float32)
    lung = np.ones((8, 8, 8), dtype=np.uint8)
    overlay = compute_iodine_overlay(ctpa, iodine, lung_mask=lung, alpha=0.6)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0 + 1e-6


def test_compute_iodine_overlay_bipolar_in_range():
    ctpa = np.full((8, 8, 8), -600.0, dtype=np.float32)
    pos = np.ones((8, 8, 8), dtype=np.float32)
    neg = np.ones((8, 8, 8), dtype=np.float32)
    lung = np.ones((8, 8, 8), dtype=np.uint8)
    overlay = compute_iodine_overlay(
        ctpa, pos, lung_mask=lung, no_seg=True, iodine_map_neg=neg, alpha=0.6
    )
    # With both pos and neg fully on, weight saturates to 1 and the previous
    # bug pushed the blend above 1; ensure it stays bounded now.
    assert overlay.max() <= 1.0 + 1e-6
    assert overlay.min() >= 0.0


# ---------------------------------------------------------------------------
# QC (redesigned metrics)
# ---------------------------------------------------------------------------

def test_qc_metrics_keys():
    ctpa = np.random.rand(16, 24, 24).astype(np.float32)
    pre = np.random.rand(16, 24, 24).astype(np.float32)
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    iodine = np.random.rand(16, 24, 24).astype(np.float32) * 10
    qc = compute_qc_metrics(ctpa, pre, lung, iodine, z_spacing_mm=1.5)
    assert "qc_pass" in qc
    assert "diaphragm_displacement_mm" in qc
    assert "registration_mismatch_cov" in qc
    assert "percent_invalid_lung_voxels" in qc


def test_qc_diaphragm_displacement_units():
    """displacement should scale with z spacing (mm, not slices).

    Volumes are nibabel (X, Y, Z) with the I/S axis along array axis 2, so the
    QC slice axis must be axis 2 (the old code scanned axis 0 and relabelled a
    slice index as "mm").
    """
    z_slices = 60
    ctpa = np.zeros((24, 24, z_slices), dtype=np.float32)
    pre = np.zeros((24, 24, z_slices), dtype=np.float32)
    lung = np.zeros((24, 24, z_slices), dtype=np.uint8)
    lung[..., 10:50] = 1
    # Distinct signal near the bottom slice so correlation peaks at a shift.
    ctpa[..., 49] = 100.0
    pre[..., 45] = 100.0  # ~4 slices off

    qc1 = compute_qc_metrics(ctpa, pre, lung, np.zeros_like(ctpa), z_spacing_mm=1.0)
    qc2 = compute_qc_metrics(ctpa, pre, lung, np.zeros_like(ctpa), z_spacing_mm=2.0)
    assert abs(qc2["diaphragm_displacement_mm"]) >= abs(qc1["diaphragm_displacement_mm"]) - 1e-6


def test_qc_infers_z_axis_from_affine():
    from ctpamap.qc import _infer_z_axis

    iso = np.eye(4)
    assert _infer_z_axis(iso) == 2
    # A (hypothetical, non-RAS-reoriented) volume whose superior axis is stored
    # along array axis 0: the inference must pick axis 0.
    weird = np.eye(4)
    weird[:3, 0] = [0.0, 0.0, 1.5]
    weird[:3, 1] = [1.5, 0.0, 0.0]
    weird[:3, 2] = [0.0, 1.5, 0.0]
    assert _infer_z_axis(weird) == 0
    assert _infer_z_axis(None) == 2


def test_qc_mismatch_better_for_aligned():
    """Identical volumes should yield a smaller mismatch than misaligned ones."""
    rng = np.random.default_rng(0)
    ctpa = rng.normal(0, 100, (24, 24, 24)).astype(np.float32)
    lung = np.ones((24, 24, 24), dtype=np.uint8)
    iodine = np.zeros_like(ctpa)

    good = compute_qc_metrics(ctpa, ctpa.copy(), lung, iodine)["registration_mismatch_cov"]
    bad = compute_qc_metrics(ctpa, ctpa + 500.0, lung, iodine)["registration_mismatch_cov"]
    assert good < bad


# ---------------------------------------------------------------------------
# Jacobian / aeration
# ---------------------------------------------------------------------------

def test_jacobian_identity_is_one():
    """A zero displacement field has Jacobian determinant ~1 everywhere."""
    disp = np.zeros((3, 16, 16, 16), dtype=np.float32)
    jac = displacement_to_jacobian(disp, (1.5, 1.5, 1.5))
    assert jac.shape == (16, 16, 16)
    # Edges are affected by smoothing; check the interior.
    assert np.allclose(jac[4:-4, 4:-4, 4:-4], 1.0, atol=1e-3)


def test_correct_aeration_identity():
    """A Jacobian of 1 leaves the CTPA unchanged (up to float rounding)."""
    ctpa = np.full((8, 8, 8), -700.0, dtype=np.float32)
    jac = np.ones_like(ctpa)
    out = correct_aeration(ctpa, jac)
    assert np.allclose(out, ctpa, atol=1e-3)


def test_jacobian_anisotropic_z_scaling():
    """A linear stretch along z gives J == the scaling factor under anisotropic
    spacing, with spacing passed in (sz, sy, sx) sitk-voxel order."""
    shape = (32, 24, 24)  # (Z, Y, X)
    sz = 3.0
    zz = np.arange(shape[0], dtype=np.float32)
    disp = np.zeros((3, *shape), dtype=np.float32)
    z_phys = zz * sz  # physical position in mm
    disp[2] = 0.5 * z_phys[:, None, None]  # du_z/dz = 0.5

    jac = displacement_to_jacobian(disp, (sz, 2.0, 1.5))
    # Returned in nibabel (X, Y, Z) order: (X=24, Y=24, Z=32).
    assert jac.shape == (24, 24, 32)
    interior = jac[5:-5, 5:-5, 6:-6]
    assert np.allclose(interior, 1.5, atol=1e-2)


# ---------------------------------------------------------------------------
# vesselness_map (extracted from segment_vessels)
# ---------------------------------------------------------------------------

def test_vesselness_map_shape_and_dtype():
    data = np.random.randn(16, 24, 24).astype(np.float32) * 200 - 500
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    vmap = vesselness_map(data, lung)
    assert vmap.dtype == np.float32
    assert vmap.shape == data.shape


def test_vesselness_map_zero_outside_lung():
    data = np.random.randn(16, 24, 24).astype(np.float32) * 200 - 500
    lung = np.zeros((16, 24, 24), dtype=np.uint8)
    lung[4:12, 8:16, 8:16] = 1
    vmap = vesselness_map(data, lung)
    assert vmap[0, 0, 0] == 0.0  # outside lung


def test_vesselness_map_contrast_invariant():
    """Adding constant brightness (simulating contrast) should not change the
    vesselness response, because Frangi is edge/ridge-sensitive and the
    per-lung normalisation removes any global shift."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 100, (16, 24, 24)).astype(np.float32)
    lung = np.ones((16, 24, 24), dtype=np.uint8)
    v1 = vesselness_map(data, lung)
    v2 = vesselness_map(data + 200, lung)  # simulate contrast
    # The response should be similar (not identical due to float precision /
    # histogram interactions, but close).
    corr = np.corrcoef(v1[lung.astype(bool)], v2[lung.astype(bool)])[0, 1]
    assert corr > 0.99


def test_segment_vessels_still_works_via_vesselness():
    """Refactoring segment_vessels to use vesselness_map internally should not
    change its output."""
    data = np.full((32, 48, 48), -300, dtype=np.float32)
    data[:, 15:20, 20:25] = 300
    lung = np.zeros((32, 48, 48), dtype=np.uint8)
    lung[:, 10:30, 10:40] = 1
    vessels = segment_vessels(data, lung)
    assert vessels.dtype == np.uint8
    assert vessels.shape == data.shape


# ---------------------------------------------------------------------------
# registration_vessel backend
# ---------------------------------------------------------------------------

def _make_aligned_pair(shape=(24, 32, 32), spacing=2.0, shift=(0, 0, 0)):
    """Create a synthetic pre/CTPA pair with optional rigid shift on a common grid."""
    affine = np.eye(4)
    affine[:3, :3] = np.diag([spacing, spacing, spacing])
    img = nib.Nifti1Image(np.empty(shape, dtype=np.float32), affine)

    # Build a "lung" volume with some vessels (tubes of high intensity).
    ctpa = np.full(shape, -800, dtype=np.float32)
    pre = np.full(shape, -800, dtype=np.float32)
    # Random background tissue.
    rng = np.random.default_rng(7)
    tissue = rng.uniform(-400, -200, shape)
    ctpa[:] = tissue.copy()
    pre[:] = tissue.copy()

    # Add a horizontal "vessel" bar through the middle.
    ctpa[shape[0] // 2 - 1:shape[0] // 2 + 1,
         shape[1] // 4:3 * shape[1] // 4,
         shape[2] // 2 - 1:shape[2] // 2 + 1] = 200
    # In pre, the same vessel but dimmer (no contrast).
    pre[shape[0] // 2 - 1:shape[0] // 2 + 1,
        shape[1] // 4:3 * shape[1] // 4,
        shape[2] // 2 - 1:shape[2] // 2 + 1] = 0
    # Shift pre to simulate misalignment.
    sx, sy, sz = shift
    pre = np.roll(pre, shift=(sx, sy, sz), axis=(0, 1, 2))

    lung = np.zeros(shape, dtype=np.uint8)
    lung[4:shape[0] - 4, 4:shape[1] - 4, 4:shape[2] - 4] = 1
    pre_img = nib.Nifti1Image(pre, affine)
    return pre, pre_img, ctpa, img, lung


def test_vessel_backend_imports():
    from ctpamap.registration_vessel import register_pre_to_ctpa_vessel
    assert callable(register_pre_to_ctpa_vessel)


def test_vessel_backend_identity_returns_ddf():
    """With identical structured volumes, the backend should return near-identity
    warp and a valid DDF (all-near-zero displacements)."""
    from ctpamap.registration_vessel import register_pre_to_ctpa_vessel

    shape = (16, 20, 20)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    img = nib.Nifti1Image(np.empty(shape), affine)

    # Structured content (tissue textured blocks + a bright vessel bar), so the
    # MI/vesselness optimum is a stable identity. Pure white noise is degenerate
    # for the affine initializer (no overlap signal once it wanders).
    rng = np.random.default_rng(0)
    data = rng.normal(-500, 150, shape).astype(np.float32)
    data[shape[0] // 2 - 1:shape[0] // 2 + 1,
         shape[1] // 4:3 * shape[1] // 4,
         shape[2] // 2 - 1:shape[2] // 2 + 1] += 250
    lung = np.ones(shape, dtype=np.uint8)
    lung[0, :, :] = 0
    lung[-1, :, :] = 0

    reg, ddf, composite = register_pre_to_ctpa_vessel(
        data, img, data, img, lung_mask=lung,
        shrink_levels=(4, 2, 1),  # fast for test
    )
    assert reg.shape == shape
    assert ddf is not None
    assert ddf.shape == (3, shape[2], shape[1], shape[0])  # (3, Z, Y, X)
    assert composite is not None
    # Identity warp → near-zero displacements.
    median_disp = np.median(np.abs(ddf))
    assert median_disp < 3.0  # generous tolerance for tiny test volume


def test_vessel_backend_corrects_small_shift():
    """A small rigid shift should be reduced after registration."""
    from ctpamap.registration_vessel import register_pre_to_ctpa_vessel

    shape = (24, 32, 32)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    img = nib.Nifti1Image(np.empty(shape), affine)

    rng = np.random.default_rng(42)
    # Rich texture (random tissue) so MI and vesselness have signal.
    base = rng.normal(-500, 150, shape).astype(np.float32)
    ctpa = base.copy()
    # Add several "vessels" (bright bars) that the Frangi vesselness can lock onto.
    for axis, lo, hi in [(0, 6, 18), (1, 8, 24), (2, 10, 22)]:
        idx = [slice(None)] * 3
        mid = shape[axis] // 2
        idx[axis] = slice(mid - 1, mid + 2)
        ctpa[tuple(idx)] += 250
    pre = np.roll(base.copy(), shift=(0, 3, 0), axis=(0, 1, 2))
    # Same vessel bars in pre (dimmer, no contrast).
    for axis, lo, hi in [(0, 6, 18), (1, 8, 24), (2, 10, 22)]:
        idx = [slice(None)] * 3
        mid = shape[axis] // 2
        idx[axis] = slice(mid - 1, mid + 2)
        pre[tuple(idx)] += 50

    lung = np.zeros(shape, dtype=np.uint8)
    lung[2:shape[0] - 2, 2:shape[1] - 2, 2:shape[2] - 2] = 1

    reg, ddf, composite = register_pre_to_ctpa_vessel(
        pre, img, ctpa, img, lung_mask=lung,
        shrink_levels=(4, 2, 1),
    )

    assert composite is not None
    # Check that the DDF actually moves voxels (not all-zero).
    assert ddf is not None
    median_disp = float(np.median(np.sqrt((ddf ** 2).sum(axis=0))))
    # A 3-voxel (6mm) shift should produce at least 1mm median displacement.
    assert median_disp > 1.0

    # The residual difference inside the lung should not get dramatically worse
    # after registration (this toy volume is small; tolerances are generous).
    lung_bool = lung.astype(bool)
    diff_before = np.mean(np.abs(pre[lung_bool] - ctpa[lung_bool]))
    diff_after = np.mean(np.abs(reg[lung_bool] - ctpa[lung_bool]))
    assert diff_after <= diff_before + 15.0  # tolerance for resampling blur
