"""
sctviewern.py — Fast GPU-accelerated CTPA iodine map viewer using napari.

Usage:
    python sctviewern.py ctpa.nii.gz iodine_clean.nii.gz [--threshold 1.0]
"""

import argparse
import ctypes
from pathlib import Path

try:
    import PyQt6
    _qt_lib = Path(PyQt6.__file__).parent / "Qt6" / "lib"
    _lib = _qt_lib / "libxcb-cursor.so.0"
    if _lib.exists():
        ctypes.CDLL(str(_lib))
except Exception:
    pass

import numpy as np
import nibabel as nib

IODINE_CMAP_RGB = [
    (32, 0, 72), (33, 0, 73), (34, 0, 75), (35, 0, 76), (36, 0, 78),
    (37, 0, 79), (38, 0, 80), (39, 0, 82), (40, 0, 83), (41, 0, 85),
    (42, 0, 86), (43, 0, 88), (44, 0, 89), (44, 0, 90), (45, 0, 92),
    (46, 0, 93), (47, 0, 95), (48, 0, 96), (49, 0, 98), (50, 0, 99),
    (51, 0, 100), (52, 0, 102), (53, 0, 103), (54, 0, 105), (55, 0, 106),
    (56, 0, 107), (57, 0, 109), (58, 0, 110), (59, 0, 112), (60, 0, 113),
    (61, 0, 114), (62, 0, 116), (63, 0, 117), (64, 0, 119), (65, 0, 120),
    (66, 0, 122), (66, 0, 123), (67, 0, 124), (68, 0, 126), (69, 0, 127),
    (70, 0, 129), (71, 0, 130), (72, 0, 132), (73, 0, 133), (74, 0, 134),
    (75, 0, 136), (76, 0, 137), (77, 0, 139), (78, 0, 140), (80, 1, 141),
    (81, 1, 141), (83, 2, 142), (84, 3, 142), (86, 4, 143), (88, 4, 144),
    (89, 5, 144), (91, 6, 145), (93, 6, 146), (94, 7, 146), (96, 8, 147),
    (98, 8, 148), (99, 9, 148), (101, 10, 149), (102, 11, 149), (104, 11, 150),
    (106, 12, 151), (107, 13, 151), (109, 13, 152), (110, 14, 152), (112, 15, 153),
    (114, 16, 154), (115, 16, 154), (117, 17, 155), (119, 18, 156), (120, 18, 156),
    (122, 19, 157), (124, 20, 158), (125, 21, 158), (127, 21, 159), (128, 22, 159),
    (130, 23, 160), (132, 23, 161), (133, 24, 161), (135, 25, 162), (136, 26, 162),
    (138, 26, 163), (140, 27, 164), (141, 28, 164), (143, 28, 165), (145, 29, 166),
    (146, 30, 166), (148, 30, 167), (150, 31, 168), (151, 32, 168), (153, 33, 169),
    (154, 33, 169), (156, 34, 170), (157, 34, 168), (158, 35, 167), (160, 36, 165),
    (161, 36, 163), (162, 36, 162), (164, 37, 160), (165, 38, 158), (166, 38, 157),
    (167, 38, 155), (168, 39, 153), (170, 40, 152), (171, 40, 150), (172, 40, 148),
    (174, 41, 147), (175, 42, 145), (176, 42, 143), (177, 42, 142), (178, 43, 140),
    (180, 44, 138), (181, 44, 137), (182, 44, 135), (184, 45, 133), (185, 46, 132),
    (186, 46, 130), (187, 46, 128), (188, 47, 127), (190, 48, 125), (191, 48, 123),
    (192, 48, 122), (194, 49, 120), (195, 50, 118), (196, 50, 117), (197, 50, 115),
    (198, 51, 113), (200, 52, 112), (201, 52, 110), (202, 52, 108), (204, 53, 107),
    (205, 54, 105), (206, 54, 103), (207, 54, 102), (208, 55, 100), (210, 56, 98),
    (211, 56, 97), (212, 56, 95), (214, 57, 93), (215, 58, 92), (216, 58, 90),
    (217, 59, 89), (217, 60, 87), (218, 61, 86), (218, 62, 85), (219, 63, 84),
    (220, 64, 82), (220, 66, 81), (221, 67, 80), (221, 68, 78), (222, 69, 77),
    (222, 70, 76), (223, 71, 74), (224, 72, 73), (224, 73, 72), (225, 74, 71),
    (225, 75, 69), (226, 76, 68), (226, 78, 67), (227, 79, 65), (228, 80, 64),
    (228, 81, 63), (229, 82, 62), (229, 83, 60), (230, 84, 59), (231, 85, 58),
    (231, 86, 56), (232, 87, 55), (232, 88, 54), (233, 89, 53), (234, 90, 51),
    (234, 92, 50), (235, 93, 49), (235, 94, 47), (236, 95, 46), (236, 96, 45),
    (237, 97, 44), (238, 98, 42), (238, 99, 41), (239, 100, 40), (239, 101, 38),
    (240, 102, 37), (240, 104, 36), (241, 105, 34), (242, 106, 33), (242, 107, 32),
    (243, 108, 31), (243, 109, 29), (244, 110, 28), (244, 112, 29), (244, 114, 30),
    (245, 116, 31), (245, 118, 32), (245, 120, 33), (245, 122, 34), (245, 124, 35),
    (245, 125, 36), (246, 127, 37), (246, 129, 38), (246, 131, 39), (246, 133, 40),
    (246, 135, 41), (246, 137, 42), (247, 139, 43), (247, 141, 44), (247, 143, 45),
    (247, 145, 46), (247, 147, 47), (247, 149, 48), (248, 151, 49), (248, 153, 50),
    (248, 155, 51), (248, 156, 52), (248, 158, 53), (249, 160, 54), (249, 162, 55),
    (249, 164, 56), (249, 166, 57), (249, 168, 58), (249, 170, 59), (250, 172, 61),
    (250, 174, 62), (250, 176, 63), (250, 178, 64), (250, 180, 65), (250, 182, 66),
    (251, 184, 67), (251, 186, 68), (251, 187, 69), (251, 189, 70), (251, 191, 71),
    (252, 193, 72), (252, 195, 73), (252, 197, 74), (252, 199, 75), (252, 201, 76),
    (252, 203, 77), (253, 205, 78), (253, 207, 79), (253, 209, 80), (253, 211, 81),
    (253, 213, 82), (253, 215, 83), (254, 217, 84), (254, 218, 85), (254, 220, 86),
    (254, 222, 87), (254, 224, 88), (254, 226, 89), (255, 228, 90), (255, 230, 91),
    (255, 232, 92),
]


def _make_iodine_cmap():
    from vispy.color import Colormap
    rgb = np.array(IODINE_CMAP_RGB, dtype=np.float32) / 255.0
    return Colormap(rgb)


def _make_iodine_cmap_bipolar():
    from vispy.color import Colormap
    pos_rgb = np.array(IODINE_CMAP_RGB, dtype=np.float32) / 255.0
    neg = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = (255 - i) / 255.0
        neg[i] = [t * 0.12, t * 0.31, t * 0.78]
    rgb = np.vstack([neg, pos_rgb])
    return Colormap(rgb)


def _load_nifti(path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.transpose(data, (2, 1, 0))
    data = data[::-1, ::-1, :]
    pixdim = img.header.get_zooms()[:3]
    scale = (pixdim[2], pixdim[1], pixdim[0])
    return data, scale


class SCTViewerN:
    def __init__(self, ctpa_path, iodine_path, threshold=0.0):
        import napari

        ctpa_data, ctpa_scale = _load_nifti(ctpa_path)
        iodine_data, _ = _load_nifti(iodine_path)

        if threshold > 0:
            iodine_data = iodine_data.copy()
            mask = np.abs(iodine_data) < threshold if np.any(iodine_data < -1) else iodine_data < threshold
            iodine_data[mask] = np.nan

        bipolar = bool(np.any(iodine_data < -1))
        cmap = _make_iodine_cmap_bipolar() if bipolar else _make_iodine_cmap()
        cmap_name = ("iodine_heat_bipolar" if bipolar else "iodine_heat", cmap)
        c_limits = [-150, 150] if bipolar else [1, 200]

        self.viewer = napari.Viewer(title="CTPA Iodine Map Viewer")

        self.viewer.add_image(
            ctpa_data,
            name="CTPA",
            colormap="gray",
            contrast_limits=[0, 100],
            scale=ctpa_scale,
        )

        self.viewer.add_image(
            iodine_data,
            name="Iodine",
            colormap=cmap_name,
            opacity=0.6,
            blending="additive",
            contrast_limits=c_limits,
            scale=ctpa_scale,
        )

    def launch(self):
        import napari
        napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Subtraction CTPA Iodine Map Viewer (napari)"
    )
    parser.add_argument("ctpa", help="Post-contrast CTPA NIfTI")
    parser.add_argument("iodine", help="Iodine map clean NIfTI")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Threshold: values below this are made transparent (default: 1.0)")
    args = parser.parse_args()

    viewer = SCTViewerN(args.ctpa, args.iodine, threshold=args.threshold)
    viewer.launch()


if __name__ == "__main__":
    main()
