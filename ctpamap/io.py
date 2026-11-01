import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional


def load_volume(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    return data, img


def save_volume(
    data: np.ndarray, img: nib.Nifti1Image, path: str | Path, like: Optional[nib.Nifti1Image] = None
) -> None:
    ref = like or img
    out = nib.Nifti1Image(data, ref.affine)
    out.set_sform(ref.affine, code=1)
    out.set_qform(ref.affine, code=1)
    nib.save(out, str(path))


def load_dicom_metadata(path: str | Path, sample: int = 20) -> dict:
    """Read DICOM metadata tags.

    Samples up to `sample` files from a series (or directory) so that tags
    which differ between slices are still captured, instead of only reading
    the first file. Tags already seen are kept (first occurrence wins).
    """
    import SimpleITK as sitk

    p = Path(path)
    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    except Exception:
        dicom_names = []

    if dicom_names:
        files = list(dicom_names)
    elif p.is_dir():
        files = [str(f) for f in sorted(p.glob("*")) if f.is_file()]
    else:
        files = [str(p)]

    if not files:
        raise FileNotFoundError(f"No DICOM files found at {path}")

    meta: dict = {}
    for fname in files[:sample]:
        fr = sitk.ImageFileReader()
        fr.SetFileName(fname)
        fr.LoadPrivateTagsOn()
        try:
            fr.ReadImageInformation()
        except RuntimeError:
            # Some files in a folder may not be valid DICOM slices; skip them.
            continue
        for key in fr.GetMetaDataKeys():
            meta.setdefault(key, fr.GetMetaData(key))
    return meta
