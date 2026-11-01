import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion, label
from scipy.ndimage import generate_binary_structure


def _mask_lung_region(data: np.ndarray, air_thresh: float = -500) -> np.ndarray:
    body_raw = data > air_thresh
    struct = generate_binary_structure(3, 2)
    labelled, n = label(body_raw, structure=struct)
    if n == 0:
        return np.zeros_like(data, dtype=bool)
    sizes = np.bincount(labelled.ravel())[1:]
    main = (labelled == (np.argmax(sizes) + 1))
    body = binary_dilation(main, structure=struct, iterations=2)
    return body


def _estimate_thoracic_range(data: np.ndarray) -> tuple[int, int]:
    body = data > -500
    axial_body = body.sum(axis=(0, 1)) > body.shape[0] * body.shape[1] * 0.05
    present = np.where(axial_body)[0]
    if len(present) < 5:
        return 0, data.shape[2] - 1
    top = max(0, present[0] - 5)
    bottom = min(data.shape[2] - 1, present[-1] + 5)
    return int(top), int(bottom)


def _touches_boundary(coords: tuple, shape: tuple) -> bool:
    """Return True if a connected component touches any of the 6 volume faces.

    Earlier code only checked axes 0 and 1, so a candidate reaching the
    superior/inferior (z) boundary slipped through. Now all three axes are
    considered so the edge-touch test is symmetric.
    """
    for axis, c in enumerate(coords):
        if c.min() == 0 or c.max() == shape[axis] - 1:
            return True
    return False


def segment_lungs(data: np.ndarray) -> np.ndarray:
    body = _mask_lung_region(data)
    lung_air = data < -400
    lung_candidate = lung_air & body

    struct = generate_binary_structure(3, 2)

    top, bottom = _estimate_thoracic_range(data)
    mid_z = (top + bottom) // 2

    labelled, n_labels = label(lung_candidate, structure=struct)
    if n_labels == 0:
        return np.zeros_like(data, dtype=np.uint8)

    sizes = np.bincount(labelled.ravel())[1:]
    sorted_idx = np.argsort(-sizes)
    max_size = sizes[sorted_idx[0]] if len(sorted_idx) > 0 else 1

    lung_mask = np.zeros_like(data, dtype=np.uint8)
    lungs_found = 0

    def _accept_component(idx, allow_boundary_touch):
        component = labelled == (idx + 1)
        if not allow_boundary_touch:
            coords = np.where(component)
            if _touches_boundary(coords, data.shape):
                return False
        component_mid = component[:, :, mid_z]
        if not np.any(component_mid):
            return False
        left = np.any(component_mid[:, : component_mid.shape[1] // 2])
        right = np.any(component_mid[:, component_mid.shape[1] // 2:])
        if not (left or right):
            return False
        size = sizes[idx]
        if size / max(1, max_size) <= 0.1:
            return False
        return True

    for idx in sorted_idx:
        if not _accept_component(idx, allow_boundary_touch=False):
            continue
        lung_mask |= (labelled == (idx + 1))
        lungs_found += 1
        if lungs_found >= 2:
            break

    if lungs_found == 0:
        # No interior lung component satisfied the tests. This happens when the
        # scan field is tight (lungs touch the FOV edge, common in targeted or
        # truncated-coverage studies). Retry permitting boundary-touching
        # components (still size- and mid-slice-gated) so the map isn't wiped.
        for idx in sorted_idx:
            if not _accept_component(idx, allow_boundary_touch=True):
                continue
            lung_mask |= (labelled == (idx + 1))
            lungs_found += 1
            if lungs_found >= 2:
                break

    if lungs_found == 0:
        return np.zeros_like(data, dtype=np.uint8)

    lung_mask = binary_fill_holes(lung_mask, structure=struct).astype(np.uint8)
    return lung_mask


def segment_airway(data: np.ndarray, lung_mask: np.ndarray, seed: tuple[int, int, int] | None = None) -> np.ndarray:
    """Mask the trachea + central bronchi, NOT the aerated lung.

    The gas track trachea->bronchi->segmentals->distal airspaces is a single
    connected < -700 HU component in a real thorax, so a plain region-grow from
    any seed inside it returns (almost) the whole lung and the subtraction
    masks the perfusion map to zero. To keep only the *wide* tubes we instead
    region-grow from a core-air mask (`< -950`) that is eroded by 3 voxels
    first (strips structures narrower than ~2*3+1 ~= 7-10mm at 1.5mm spacing:
    distal airspaces, subsegmental bronchi), grow from the seed, then dilate
    back by 4 to restore the vessel-wall halo.

    Failure mode is safe: if no suitable core-air exists the airway mask comes
    back empty and perfusion is left intact.
    """
    lung_region = data.copy()
    lung_region[lung_mask == 0] = 0

    if seed is None:
        mid = np.array(data.shape) // 2
        seed = (mid[0], mid[1], mid[2])

    struct = generate_binary_structure(3, 2)

    air_core = (data < -950) & lung_mask.astype(bool)
    core = binary_erosion(air_core, structure=struct, iterations=3)
    labelled, n_labels = label(core, structure=struct)

    seed_candidate = _find_first_air_seed(data, seed, radius=8, air_hu=-950)
    if seed_candidate is None:
        seed_candidate = _find_first_air_seed(data, seed, radius=8, air_hu=-850)
    if seed_candidate is None or n_labels == 0:
        return np.zeros_like(data, dtype=np.uint8)

    seed_val = labelled[seed_candidate]
    if seed_val == 0:
        return np.zeros_like(data, dtype=np.uint8)

    airway = (labelled == seed_val)
    airway = binary_dilation(airway, structure=struct, iterations=4)

    struct1 = generate_binary_structure(3, 1)
    airway = binary_erosion(airway, structure=struct1, iterations=1)
    airway = binary_dilation(airway, structure=struct1, iterations=2)

    return airway.astype(np.uint8)


def _find_first_air_seed(
    data: np.ndarray, seed: tuple[int, int, int], radius: int, air_hu: float
) -> tuple[int, int, int] | None:
    """Find the closest voxel below `air_hu` within a cube around `seed`.

    Candidates are visited in ascending Euclidean distance from `seed` and the
    first air voxel found is returned. The original triple-loop used row-major
    order and only broke out of the innermost iteration, so it kept overwriting
    the seed instead of stopping at the first (closest) hit.
    """
    sz, sy, sx = data.shape
    z0, y0, x0 = seed
    offsets = [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    offsets.sort(key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
    for dz, dy, dx in offsets:
        z, y, x = z0 + dz, y0 + dy, x0 + dx
        if 0 <= z < sz and 0 <= y < sy and 0 <= x < sx:
            if data[z, y, x] < air_hu:
                return (z, y, x)
    return None


def vesselness_map(
    data: np.ndarray, lung_mask: np.ndarray,
    sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Continuous Frangi vesselness response inside the lung.

    This is the shared primitive behind `segment_vessels` and the
    vesselness-driven registration backend: the vascular tree geometry is
    contrast-invariant (its brightness changes between pre/CTPA, but its
    *shape* doesn't), so matching vesselness maps is far more robust to
    contrast enhancement than matching raw intensities.

    Returns a float32 array in [0, 1], zeroed outside the lung.
    """
    from skimage.filters import frangi

    if sigmas is None:
        # 0.5–2.5 mm covers subsegmental → main PA at typical 1.5 mm spacing.
        sigmas = np.linspace(0.5, 2.5, 5)

    lung = data.copy()
    lung[lung_mask == 0] = 0

    # Normalize to [0, 1] within lung region for stable Hessian across HU scales
    lung_clip = np.clip(lung, -1000, 500)
    lung_vox = lung_clip[lung_mask.astype(bool)]
    if len(lung_vox) == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(lung_vox.min())
    hi = float(lung_vox.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    lung_norm = ((lung_clip - lo) / (hi - lo)).astype(np.float32)
    lung_norm[lung_mask == 0] = 0

    response = frangi(lung_norm, sigmas=sigmas, black_ridges=False, mode='constant')
    response = response.astype(np.float32)
    response[lung_mask == 0] = 0
    return response


def segment_vessels(data: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    response = vesselness_map(data, lung_mask)
    if response.max() == 0:
        return np.zeros_like(data, dtype=np.uint8)

    vessel = (response > 0.3) & lung_mask.astype(bool)

    # Remove speckle < 50 voxels
    struct = generate_binary_structure(3, 2)
    labelled, n_labels = label(vessel, structure=struct)
    if n_labels > 0:
        sizes = np.bincount(labelled.ravel())[1:]
        vessel = np.isin(labelled, np.where(sizes > 50)[0] + 1)

    return vessel.astype(np.uint8)
