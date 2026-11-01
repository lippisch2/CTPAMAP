import argparse
import sys
import gc
from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk

from ctpamap import __version__
from ctpamap.io import load_volume, save_volume
from ctpamap.preprocessing import (
    reorient_to_ras,
    resample_to_common_grid,
    clip_hu,
    light_denoise,
    _nib_to_sitk,
)
from ctpamap.segmentation import segment_lungs, segment_airway, segment_vessels
from ctpamap.registration import register_pre_to_ctpa
from ctpamap.subtraction import compute_iodine_map
from ctpamap.overlay import compute_iodine_overlay
from ctpamap.qc import compute_qc_metrics, save_qc_report


def _load_backend(name: str):
    """Lazily import a registration backend, raising a clean error if missing."""
    try:
        if name == "antspy":
            from ctpamap.registration_antspy import register_pre_to_ctpa_antspy
            return register_pre_to_ctpa_antspy
        if name == "stix":
            from ctpamap.registration_elastix import register_pre_to_ctpa_elastix
            return register_pre_to_ctpa_elastix
    except ImportError as e:
        pkg = {"antspy": "antspyx", "stix": "itk-elastix"}.get(name)
        if pkg:
            print(f"ERROR: --{name} requires {pkg}. Install with: pip install {pkg}")
            raise sys.exit(1) from e
        raise
    raise ValueError(f"Unknown backend: {name}")


def run_pipeline(pre_path: str, ctpa_path: str, out_dir: str,
                 target_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
                 skip_reg: bool = False, aggressive: bool = False,
                 method: str = "default", hd: bool = False,
                 lite: bool = False, no_seg: bool = False,
                 do_aer: bool = False, use_gpu: bool = False,
                 no_clip: bool = False, rigid_z_search: bool = False,
                 progress_callback=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading volumes...")
    if progress_callback:
        progress_callback("loading", 5)
    pre_data, pre_img = load_volume(pre_path)
    ctpa_data, ctpa_img = load_volume(ctpa_path)
    print(f"  pre shape:  {pre_data.shape}, ctpa shape: {ctpa_data.shape}")

    print("Reorienting to RAS...")
    if progress_callback:
        progress_callback("reorienting", 10)
    pre_data, pre_img = reorient_to_ras(pre_data, pre_img)
    ctpa_data, ctpa_img = reorient_to_ras(ctpa_data, ctpa_img)
    print(f"  pre shape:  {pre_data.shape}, ctpa shape: {ctpa_data.shape}")

    if hd:
        pre_aff = pre_img.affine
        ctpa_aff = ctpa_img.affine
        pre_sp = np.sqrt((pre_aff[:3, :3] ** 2).sum(axis=0))
        ctpa_sp = np.sqrt((ctpa_aff[:3, :3] ** 2).sum(axis=0))
        target_spacing = tuple(min(pre_sp[i], ctpa_sp[i]) for i in range(3))
        print(f"  pre spacing: {tuple(pre_sp.tolist())}")
        print(f"  ctpa spacing: {tuple(ctpa_sp.tolist())}")
        print(f"  HD target spacing: {target_spacing}")

    print("Resampling to common grid...")
    pre_data, pre_img, ctpa_data, ctpa_img = resample_to_common_grid(
        pre_data, pre_img, ctpa_data, ctpa_img, target_spacing
    )
    print(f"  pre shape:  {pre_data.shape}, ctpa shape: {ctpa_data.shape}")

    # Keep the axial slice spacing for QC (diaphragm displacement in mm).
    z_spacing_mm = float(target_spacing[2])

    # Raw (unclipped) CTPA keeps the full iodinated dynamic range for the
    # subtraction and is unaffected by the optional aeration correction.
    ctpa_data_raw = ctpa_data.copy()

    if no_clip:
        print("Skipping HU clipping (--noclip)...")
    else:
        print("Clipping HU range...")
        pre_data = clip_hu(pre_data)
        ctpa_data = clip_hu(ctpa_data)

    print("Light denoising...")
    pre_data = light_denoise(pre_data)
    ctpa_data = light_denoise(ctpa_data)

    print("Segmenting lungs...")
    lung_mask_pre = segment_lungs(pre_data)
    lung_mask_ctpa = segment_lungs(ctpa_data)
    lung_mask = (lung_mask_pre | lung_mask_ctpa).astype(np.uint8)

    if progress_callback:
        progress_callback("segmenting", 35)

    if not no_seg:
        print("Segmenting airways...")
        airway_mask = segment_airway(pre_data, lung_mask)

        print("Segmenting vessels (CTPA)...")
        vessel_mask_ctpa = segment_vessels(ctpa_data, lung_mask)
        print("Segmenting vessels (pre-contrast)...")
        vessel_mask_pre = segment_vessels(pre_data, lung_mask_pre)
    else:
        airway_mask = None
        vessel_mask_ctpa = None
        vessel_mask_pre = None

    print("Saving post-contrast CTPA on common grid...")
    save_volume(ctpa_data, ctpa_img, out / "ctpa_regrid.nii.gz")

    ref_img = ctpa_img
    ddf = None
    composite = None
    if skip_reg:
        print("Skipping registration (--no-reg), using common-grid pre directly...")
        pre_registered = pre_data
    else:
        print("Registering pre -> CTPA...")
        if progress_callback:
            progress_callback("registering", 55)
        reg_lung_mask = (lung_mask_pre & lung_mask_ctpa).astype(np.uint8)
        if reg_lung_mask.sum() < 100:
            print("WARNING: lung intersection too small, using ctpa-only lung mask for registration")
            reg_lung_mask = lung_mask_ctpa
        mode_str = method
        if aggressive and lite:
            mode_str += ", aggressive (lite)"
        elif aggressive:
            mode_str += ", aggressive"
        if no_seg:
            mode_str += ", no seg"
        print(f"  method: {mode_str}")

        if use_gpu:
            try:
                from ctpamap.registration_fireants import register_pre_to_ctpa_fireants
                print("  backend: FireANTs (GPU)")
                pre_registered, ddf = register_pre_to_ctpa_fireants(
                    pre_data, pre_img, ctpa_data, ctpa_img,
                    lung_mask=reg_lung_mask,
                    pre_lung_mask=lung_mask_pre,
                    aggressive=aggressive,
                )
                method = "fireants"
            except ImportError as e:
                print(f"  WARNING: FireANTs unavailable ({e}); using default CPU backend")
                print("           Install with: pip install ctpamap[fireants]")
                method = "default"

        if method == "vessel":
            from ctpamap.registration_vessel import register_pre_to_ctpa_vessel
            print("  backend: vesselness-driven MI + B-spline")
            pre_registered, ddf, composite = register_pre_to_ctpa_vessel(
                pre_data, pre_img, ctpa_data, ctpa_img,
                lung_mask=reg_lung_mask,
                pre_lung_mask=lung_mask_pre,
                aggressive=aggressive,
                airway_mask=airway_mask,
            )
        elif method in ("antspy", "stix"):
            backend_fn = _load_backend(method)
            print(f"  backend: {method}")
            pre_registered, ddf = backend_fn(
                pre_data, pre_img, ctpa_data, ctpa_img,
                lung_mask=reg_lung_mask,
                pre_lung_mask=lung_mask_pre,
                aggressive=aggressive,
            )
            composite = None
        elif method == "default":
            print("  backend: sitk.NormalizedCorrelation + demons")
            pre_registered, _, _, composite = register_pre_to_ctpa(
                pre_data, pre_img, ctpa_data, ctpa_img,
                lung_mask=reg_lung_mask,
                pre_lung_mask=lung_mask_pre,
                aggressive=aggressive,
                lite=lite,
                rigid_z_search=rigid_z_search,
            )
            if do_aer:
                try:
                    from ctpamap.registration import extract_displacement_field_sitk
                    ctpa_sitk = _nib_to_sitk(ctpa_data, ctpa_img)
                    ddf = extract_displacement_field_sitk(composite, ctpa_sitk)
                except Exception as e:
                    print(f"  WARNING: could not extract displacement field ({e})")

    save_volume(pre_registered, ref_img, out / "registered_pre.nii.gz")

    if do_aer and not skip_reg:
        from ctpamap.jacobian import displacement_to_jacobian, correct_aeration
        if ddf is not None:
            print("  Computing aeration correction...")
            # DDF arrays are (3, Z, Y, X) in sitk voxel order, so spacings must
            # be given as (s_z, s_y, s_x) = (target_spacing[2], [1], [0]).
            jac = displacement_to_jacobian(
                ddf, (target_spacing[2], target_spacing[1], target_spacing[0])
            )
            lung_vox = lung_mask.astype(bool)
            if lung_vox.sum() > 0:
                j_lung = jac[lung_vox]
                print(f"    J: mean={j_lung.mean():.4f}, min={j_lung.min():.4f}, max={j_lung.max():.4f},"
                      f" {((j_lung < 0.9).sum() / j_lung.size * 100):.0f}%<0.9,"
                      f" {((j_lung > 1.1).sum() / j_lung.size * 100):.0f}%>1.1")
            ctpa_data = correct_aeration(ctpa_data, jac)
        else:
            print("  WARNING: DDF not available from this backend, skipping aeration correction")

    if no_seg:
        vessel_mask = None
        airway_mask = None
    else:
        # Combine vessel masks: warp the pre vessel mask to CTPA space via the
        # backend's SimpleITK composite (default and vessel backends expose one).
        if not skip_reg and composite is not None:
            ctpa_sitk = _nib_to_sitk(ctpa_data, ctpa_img)
            pre_vessel_sitk = _nib_to_sitk(vessel_mask_pre.astype(np.float32), pre_img)
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(ctpa_sitk)
            r.SetTransform(composite)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            pre_mask_ctpa_sitk = r.Execute(sitk.Cast(pre_vessel_sitk, sitk.sitkFloat32))
            pre_mask_ctpa = np.transpose(sitk.GetArrayFromImage(pre_mask_ctpa_sitk).astype(bool), (2, 1, 0))
            vessel_mask = (vessel_mask_ctpa.astype(bool) | pre_mask_ctpa).astype(np.uint8)
            print("  vessel mask: CTPA + warped pre-contrast")
        else:
            vessel_mask = vessel_mask_ctpa
            print("  vessel mask: CTPA only (no composite from this backend)")

        print("Saving segmentation masks...")
        save_volume(lung_mask, pre_img, out / "lung_mask.nii.gz")
        save_volume(vessel_mask, pre_img, out / "vessel_mask.nii.gz")
        save_volume(airway_mask, pre_img, out / "airway_mask.nii.gz")

    print("Computing iodine map...")
    if progress_callback:
        progress_callback("iodine_map", 80)
    iodine_result = compute_iodine_map(
        ctpa_data_raw, pre_registered, lung_mask,
        vessel_mask=vessel_mask, airway_mask=airway_mask,
        no_seg=no_seg,
    )

    print("Saving iodine maps...")
    save_volume(iodine_result["raw"], ref_img, out / "iodine_map_raw.nii.gz")
    save_volume(iodine_result["clean"], ref_img, out / "iodine_map_clean.nii.gz")

    print("Computing iodine overlay...")
    if progress_callback:
        progress_callback("overlay", 90)
    overlay = compute_iodine_overlay(
        ctpa_data, iodine_result["normalized"], lung_mask=lung_mask,
        no_seg=no_seg, iodine_map_neg=iodine_result["normalized_neg"],
    )
    save_volume(overlay, ref_img, out / "iodine_overlay.nii.gz")

    print("Computing QC metrics...")
    if progress_callback:
        progress_callback("qc", 95)
    qc_metrics = compute_qc_metrics(
        ctpa_data, pre_registered, lung_mask, iodine_result["clean"],
        z_spacing_mm=z_spacing_mm, affine=ctpa_img.affine,
    )
    save_qc_report(qc_metrics, out / "qc_report.json")
    print("QC:", json.dumps(qc_metrics, indent=2))

    # Free large intermediates to ease memory pressure in HD mode.
    del iodine_result, overlay
    gc.collect()

    print(f"\nAll outputs written to {out.resolve()}")
    if progress_callback:
        progress_callback("done", 100)


def main():
    parser = argparse.ArgumentParser(
        description="CTPA Lung Subtraction Perfusion Mapper",
        prog="ctpamap",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("pre", help="Non-contrast chest CT (pre) NIfTI path")
    parser.add_argument("ctpa", help="Contrast-enhanced pulmonary CTA NIfTI path")
    parser.add_argument("-o", "--out", default=".", help="Output directory")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="Target spacing in mm (x y z)")
    parser.add_argument("--no-reg", action="store_true",
                        help="Skip registration; use common-grid pre volume directly")
    parser.add_argument("--agg", action="store_true",
                        help="Aggressive deformable registration for perfect lung contour match")
    parser.add_argument("--agglite", action="store_true",
                        help="Aggressive with downsampled rigid search (faster)")
    parser.add_argument("--hd", action="store_true",
                        help="High-definition: use finest native voxel spacing instead of --spacing")
    parser.add_argument("--noseg", action="store_true",
                        help="Skip vessel/airway segmentation; clip subtraction outliers instead")
    parser.add_argument("--aer", action="store_true",
                        help="Apply aeration correction via Jacobian determinant")
    parser.add_argument("--no-aer", action="store_true",
                        help="Disable aeration correction (overrides --aer)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use FireANTs GPU registration (pip install ctpamap[fireants])")
    parser.add_argument("--noclip", action="store_true",
                        help="Skip HU clipping (keep full iodinated dynamic range)")
    parser.add_argument("--rigid-z", action="store_true",
                        help="Add a coarse z-translation pre-scan before rigid alignment "
                             "(use when pre/CTPA differ by whole slices)")
    parser.add_argument("-v", "--view", action="store_true",
                        help="Launch SCT viewer on results after pipeline completes")

    method = parser.add_mutually_exclusive_group()
    method.add_argument("--antspy", action="store_true",
                        help="Use ANTsPy registration (pip install antspyx)")
    method.add_argument("--stix", action="store_true",
                        help="Use ITK-Elastix registration (pip install itk-elastix)")
    method.add_argument("--vessel", action="store_true",
                        help="Use vesselness-driven MI+B-spline registration (recommended for "
                             "subtraction CTPA; contrast-invariant, and supports --aer)")
    parser.add_argument("--view-n", action="store_true",
                        help="Launch the napari SCT viewer (requires: pip install ctpamap[napari])")

    args = parser.parse_args()

    method_name = "default"
    aggressive = args.agg or args.agglite
    lite = args.agglite
    if args.antspy:
        method_name = "antspy"
    elif args.stix:
        method_name = "stix"
    elif args.vessel:
        method_name = "vessel"

    # `--noseg` no longer implicitly enables aeration correction; aeration is
    # only applied when explicitly requested with `--aer` (and not cancelled by
    # `--no-aer`).
    do_aer = bool(args.aer) and not args.no_aer

    run_pipeline(args.pre, args.ctpa, args.out, tuple(args.spacing),
                 skip_reg=args.no_reg, aggressive=aggressive, method=method_name,
                 hd=args.hd, lite=lite, no_seg=args.noseg, do_aer=do_aer,
                 use_gpu=args.gpu, no_clip=args.noclip,
                 rigid_z_search=args.rigid_z)

    if args.view:
        out = Path(args.out)
        registered_pre = str(out / "registered_pre.nii.gz")
        iodine_clean = str(out / "iodine_map_clean.nii.gz")
        print("Launching SCT viewer...")
        from ctpamap.sctviewer import SCTViewer
        viewer = SCTViewer(registered_pre, iodine_clean)
        viewer.launch()

    if args.view_n:
        out = Path(args.out)
        print("Launching napari SCT viewer...")
        from ctpamap.sctviewern import SCTViewerN
        SCTViewerN(ctpa_path=str(out / "ctpa_regrid.nii.gz"),
                   iodine=str(out / "iodine_map_clean.nii.gz")).launch()


if __name__ == "__main__":
    main()