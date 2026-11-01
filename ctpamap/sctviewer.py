"""
sctviewer.py — Dual-volume viewer for subtraction CTPA iodine maps.

Shows post-contrast CTPA in grayscale (abdominal window) with
iodine-map-clean overlay in a custom heat colormap.
Toggles between single axial view and 3-view MPR (axial + sagittal + coronal).

Usage:
    python sctviewer.py ctpa.nii.gz iodine_clean.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib

# The previous module set matplotlib.use("TkAgg") unconditionally at import
# time, which breaks on headless machines and overrides the user's chosen
# backend. Instead, only force an interactive backend lazily inside launch()
# and fall back gracefully.
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib import colors, cm

CT_WW = 100.0
CT_WL = 50.0

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


def _select_interactive_backend():
    """Try to enable an interactive backend without hard-failing on headless hosts."""
    for backend in (matplotlib.get_backend(), "TkAgg", "Qt5Agg", "Qt4Agg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


class SCTViewer:
    def __init__(self, ctpa_path, iodine_path):
        self.ctpa_data = None
        self.iodine_data = None
        self.shape = None
        self.view_mode = 'axial'
        self.flip_lr = False

        self.i_min = 1.0
        self.i_max = 200.0
        self.i_ww = 100.0
        self.i_wl = 50.0

        self.alpha = 0.6

        self.interpolate = True

        self.bipolar = False

        self.cmap_name = "IodineHeat"
        rgb = np.array(IODINE_CMAP_RGB, dtype=np.float32) / 255.0
        self.cmap = colors.ListedColormap(rgb, name="IodineHeat")

        neg_part = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = (255 - i) / 255.0
            neg_part[i] = [t * 0.12, t * 0.31, t * 0.78]
        pos_part = np.array(IODINE_CMAP_RGB, dtype=np.float32) / 255.0
        combined = np.vstack([neg_part, pos_part])
        self.cmap_bipolar = colors.ListedColormap(combined, name="IodineBipolar")

        self.fig = None
        self.ax = None
        self.im = None
        self.slider = None
        self.cax = None
        self.cbar = None
        self.i_min_text = None
        self.i_max_text = None
        self.i_ww_text = None
        self.i_wl_text = None

        self.mpr_mode = False
        self.pos = [0, 0, 0]
        self.mpr_btn = None
        self.interp_btn = None
        self.sag_btn = None
        self.cor_btn = None
        self.flip_btn = None
        self._mpr_axes = []
        self._mpr_ims = []
        self._mpr_cross = []

        self.load_files(ctpa_path, iodine_path)

    def load_files(self, ctpa_path, iodine_path):
        ctpa_img = nib.load(str(ctpa_path))
        self.ctpa_data = np.asarray(ctpa_img.get_fdata(), dtype=np.float32)

        iodine_img = nib.load(str(iodine_path))
        self.iodine_data = np.asarray(iodine_img.get_fdata(), dtype=np.float32)

        if self.iodine_data.ndim == 4:
            self.iodine_data = self.iodine_data[..., 0]

        self.bipolar = bool(np.any(self.iodine_data < -1))
        if self.bipolar:
            self.i_min = -150.0
            self.i_max = 150.0
            self.i_ww = 300.0
            self.i_wl = 0.0

        self.shape = self.ctpa_data.shape[:3]
        self.pos = [s // 2 for s in self.shape]

    def _composite_slice(self, ctpa_sl, iodine_sl):
        ct_low = CT_WL - CT_WW / 2
        ct_high = CT_WL + CT_WW / 2
        ct_norm = np.clip((ctpa_sl - ct_low) / (ct_high - ct_low), 0, 1)

        i_low = self.i_wl - self.i_ww / 2
        i_high = self.i_wl + self.i_ww / 2
        if i_high <= i_low:
            i_high = i_low + 1
        iodine_clipped = np.clip(iodine_sl, self.i_min, self.i_max)
        iodine_norm = np.clip((iodine_clipped - i_low) / (i_high - i_low), 0, 1)

        cmap = self.cmap_bipolar if self.bipolar else self.cmap
        iodine_rgb = cmap(iodine_norm)[..., :3]
        mask = np.abs(iodine_sl) > 5 if self.bipolar else iodine_sl >= self.i_min
        bg = ct_norm[..., None]
        blended = (1 - self.alpha) * bg + self.alpha * iodine_rgb
        result = np.where(mask[..., None], blended, bg)
        return np.clip(result, 0, 1)

    @property
    def _view_axis(self):
        return {'axial': 2, 'sagittal': 0, 'coronal': 1}[self.view_mode]

    @property
    def _view_size(self):
        return self.shape[self._view_axis]

    def _get_view(self, axis, idx):
        if axis == 'axial':
            return self._composite_slice(
                self.ctpa_data[:, :, idx].T,
                self.iodine_data[:, :, idx].T), self.shape[0], self.shape[1]
        elif axis == 'sagittal':
            return self._composite_slice(
                self.ctpa_data[idx, :, :].T,
                self.iodine_data[idx, :, :].T), self.shape[2], self.shape[1]
        elif axis == 'coronal':
            return self._composite_slice(
                self.ctpa_data[:, idx, :].T,
                self.iodine_data[:, idx, :].T), self.shape[2], self.shape[0]

    def _build_single_ui(self):
        for a in self._mpr_axes:
            a.remove()
        self._mpr_axes.clear()
        self._mpr_ims.clear()
        self._mpr_cross.clear()

        if self.ax is not None:
            self.ax.remove()
            self.ax = None
            self.im = None
        if self._slider_ax is not None:
            self.slider.ax.remove()
            self._slider_ax = None
            self.slider = None

        self.ax = self.fig.add_axes([0.15, 0.02, 0.70, 0.94], facecolor="#000000")
        idx = self.pos[self._view_axis]
        comp, w, h = self._get_view(self.view_mode, idx)
        interp = 'bilinear' if self.interpolate else 'nearest'
        self.im = self.ax.imshow(comp, origin="lower", interpolation=interp)
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors="white")

        self._slider_ax = self.fig.add_axes([0.90, 0.02, 0.03, 0.94], facecolor="#000000")
        self.slider = Slider(
            self._slider_ax, "Slice", 0, self._view_size - 1,
            valinit=idx, valstep=1, color="#4c8eda",
            orientation="vertical",
        )
        self.slider.on_changed(self._on_slider_single)

        self.mpr_btn.label.set_text("MPR")
        self.fig.canvas.draw_idle()

    def _build_mpr_ui(self):
        self.ax.remove()
        self.ax = None
        self.im = None
        self.slider.ax.remove()
        self._slider_ax = None
        self.slider = None

        x0, y0, w, h = 0.15, 0.02, 0.70, 0.96
        gap = 0.008
        left_w = w * 0.5 - gap * 0.5
        right_w = w * 0.5 - gap * 0.5
        right_h = h * 0.5 - gap * 0.5

        ax_a = self.fig.add_axes([x0, y0, left_w, h], facecolor="#000000")
        ax_s = self.fig.add_axes([x0 + left_w + gap, y0 + h - right_h, right_w, right_h], facecolor="#000000")
        ax_c = self.fig.add_axes([x0 + left_w + gap, y0, right_w, right_h], facecolor="#000000")

        interp = 'bilinear' if self.interpolate else 'nearest'
        comp_a, _, _ = self._get_view('axial', self.pos[2])
        comp_s, ws, hs = self._get_view('sagittal', self.pos[0])
        comp_c, wc, hc = self._get_view('coronal', self.pos[1])
        im_a = ax_a.imshow(comp_a, origin="lower", interpolation=interp)
        im_s = ax_s.imshow(comp_s, origin="lower", interpolation=interp)
        im_c = ax_c.imshow(comp_c, origin="lower", interpolation=interp)

        for ax_ in (ax_a, ax_s, ax_c):
            ax_.set_aspect("equal")
            ax_.tick_params(colors="white")

        lh_a = ax_a.axhline(0, color="red", lw=0.6, alpha=0.7)
        lv_a = ax_a.axvline(0, color="red", lw=0.6, alpha=0.7)
        lh_s = ax_s.axhline(0, color="red", lw=0.6, alpha=0.7)
        lv_s = ax_s.axvline(0, color="red", lw=0.6, alpha=0.7)
        lh_c = ax_c.axhline(0, color="red", lw=0.6, alpha=0.7)
        lv_c = ax_c.axvline(0, color="red", lw=0.6, alpha=0.7)

        self._mpr_axes = [ax_a, ax_s, ax_c]
        self._mpr_ims = [im_a, im_s, im_c]
        self._mpr_cross = [
            (lh_a, lv_a),
            (lh_s, lv_s),
            (lh_c, lv_c),
        ]

        self.mpr_btn.label.set_text("Axial")
        self.fig.canvas.draw_idle()

    def _update_single(self):
        idx = self.pos[self._view_axis]
        comp, w, h = self._get_view(self.view_mode, idx)
        self.im.set_data(comp)
        self.ax.set_title(
            f"{self.view_mode.capitalize()} {idx + 1}/{self._view_size} | "
            f"I [{self.i_min:.0f}–{self.i_max:.0f}]  "
            f"WW={self.i_ww:.0f} WL={self.i_wl:.0f}",
            color="white", fontsize=10,
        )

    def _update_mpr(self):
        nx, ny, nz = self.shape
        px, py, pz = self.pos

        comp_a, _, _ = self._get_view('axial', min(pz, nz - 1))
        comp_s, ws, hs = self._get_view('sagittal', min(px, nx - 1))
        comp_c, wc, hc = self._get_view('coronal', min(py, ny - 1))

        self._mpr_ims[0].set_data(comp_a)
        self._mpr_ims[1].set_data(comp_s)
        self._mpr_ims[2].set_data(comp_c)

        self._mpr_axes[0].set_xlim(0, nx)
        self._mpr_axes[0].set_ylim(0, ny)
        self._mpr_axes[1].set_xlim(0, hs)
        self._mpr_axes[1].set_ylim(0, ws)
        self._mpr_axes[2].set_xlim(0, hc)
        self._mpr_axes[2].set_ylim(0, wc)

        ch_a_h, ch_a_v = self._mpr_cross[0]
        ch_a_h.set_ydata([py, py])
        ch_a_v.set_xdata([px, px])

        ch_s_h, ch_s_v = self._mpr_cross[1]
        ch_s_h.set_ydata([pz, pz])
        ch_s_v.set_xdata([py, py])

        ch_c_h, ch_c_v = self._mpr_cross[2]
        ch_c_h.set_ydata([pz, pz])
        ch_c_v.set_xdata([px, px])

        self._mpr_axes[0].set_title(
            f"Axial  z={pz + 1}/{nz}", color="white", fontsize=9)
        self._mpr_axes[1].set_title(
            f"Sagittal  x={px + 1}/{nx}", color="white", fontsize=9)
        self._mpr_axes[2].set_title(
            f"Coronal  y={py + 1}/{ny}", color="white", fontsize=9)

    def update(self, draw=True):
        if not self.mpr_mode and self.im is None:
            return
        if self.mpr_mode:
            self._update_mpr()
        else:
            self._update_single()

        self._refresh_cbar()

        if draw:
            self.fig.canvas.draw_idle()

    def _refresh_cbar(self):
        cmap = self.cmap_bipolar if self.bipolar else self.cmap
        norm = colors.Normalize(vmin=self.i_min, vmax=self.i_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if self.cbar is None:
            self.cbar = self.fig.colorbar(sm, cax=self.cax)
        else:
            self.cbar.mappable = sm
            self.cbar.update_normal(sm)
        self.cbar.set_label("Iodine (HU)", color="white")
        self.cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"{x:.0f}")
        )
        for t in self.cbar.ax.get_yticklabels():
            t.set_color("white")
        self.cbar.ax.tick_params(colors="white")
        self.cbar.outline.set_edgecolor("white")

    def _on_scroll_single(self, event):
        if event.inaxes != self.ax:
            return
        step = 1 if event.button == "up" else -1
        axis = self._view_axis
        self.pos[axis] = max(0, min(self._view_size - 1, self.pos[axis] + step))
        self.slider.set_val(self.pos[axis])

    def _on_scroll_mpr(self, event):
        for i, ax in enumerate(self._mpr_axes):
            if event.inaxes == ax:
                step = 1 if event.button == "up" else -1
                axis_name = ['axial', 'sagittal', 'coronal'][i]
                axis_idx = {'axial': 2, 'sagittal': 0, 'coronal': 1}[axis_name]
                self.pos[axis_idx] = max(0, min(self.shape[axis_idx] - 1,
                                               self.pos[axis_idx] - step))
                self.update()
                return

    def _on_click_mpr(self, event):
        for i, ax in enumerate(self._mpr_axes):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                x, y = int(round(event.xdata)), int(round(event.ydata))
                if i == 0:
                    self.pos[0] = max(0, min(self.shape[0] - 1, x))
                    self.pos[1] = max(0, min(self.shape[1] - 1, y))
                elif i == 1:
                    self.pos[1] = max(0, min(self.shape[1] - 1, x))
                    self.pos[2] = max(0, min(self.shape[2] - 1, y))
                else:
                    self.pos[0] = max(0, min(self.shape[0] - 1, x))
                    self.pos[2] = max(0, min(self.shape[2] - 1, y))
                self.update()
                return

    def _on_slider_single(self, val):
        self.pos[self._view_axis] = int(val)
        self.update()

    def toggle_mpr(self, event=None):
        self.mpr_mode = not self.mpr_mode
        if self.mpr_mode:
            self._build_mpr_ui()
            self.fig.canvas.mpl_disconnect(self._scroll_cid)
            self._scroll_cid = self.fig.canvas.mpl_connect(
                "scroll_event", self._on_scroll_mpr)
            self._click_cid = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_click_mpr)
        else:
            self.fig.canvas.mpl_disconnect(self._scroll_cid)
            if hasattr(self, '_click_cid'):
                self.fig.canvas.mpl_disconnect(self._click_cid)
            self.view_mode = 'axial'
            self._build_single_ui()
            self._scroll_cid = self.fig.canvas.mpl_connect(
                "scroll_event", self._on_scroll_single)
        self.update()

    def _set_view(self, mode):
        self.view_mode = mode
        if self.mpr_mode:
            self.toggle_mpr()
        elif self.fig is not None:
            self._build_single_ui()
        if self.fig is not None:
            self.update()

    def set_view_sagittal(self, event=None):
        self._set_view('sagittal')

    def set_view_coronal(self, event=None):
        self._set_view('coronal')

    def toggle_flip(self, event=None):
        self.flip_lr = not self.flip_lr
        if self.flip_btn is not None:
            label = "Flip LR" if self.flip_lr else "Normal"
            self.flip_btn.label.set_text(label)
        self.ctpa_data = self.ctpa_data[::-1, :, :].copy()
        self.iodine_data = self.iodine_data[::-1, :, :].copy()
        self.pos[0] = self.shape[0] - 1 - self.pos[0]
        if self.fig is not None:
            self.update()

    def on_i_min_submit(self, text):
        try:
            self.i_min = float(text)
            self.update()
        except ValueError:
            pass

    def on_i_max_submit(self, text):
        try:
            self.i_max = float(text)
            self.update()
        except ValueError:
            pass

    def on_i_ww_submit(self, text):
        try:
            self.i_ww = float(text)
            self.update()
        except ValueError:
            pass

    def on_i_wl_submit(self, text):
        try:
            self.i_wl = float(text)
            self.update()
        except ValueError:
            pass

    def on_alpha_change(self, val):
        self.alpha = val
        self.update()

    def _apply_interp(self):
        interp = 'bilinear' if self.interpolate else 'nearest'
        if self.mpr_mode:
            for im in self._mpr_ims:
                im.set_interpolation(interp)
        elif self.im is not None:
            self.im.set_interpolation(interp)

    def toggle_interp(self, event=None):
        self.interpolate = not self.interpolate
        label = "Smooth" if self.interpolate else "Pixel"
        self.interp_btn.label.set_text(label)
        self._apply_interp()
        self.fig.canvas.draw_idle()

    def auto_ww_wl(self, event=None):
        if self.bipolar:
            values = self.iodine_data[np.abs(self.iodine_data) > 5]
            if values.size == 0:
                return
            p98 = float(np.percentile(np.abs(values), 98))
            self.i_min = -p98
            self.i_max = p98
            self.i_ww = 2 * p98
            self.i_wl = 0.0
        else:
            values = self.iodine_data[self.iodine_data > 1.0]
            if values.size == 0:
                return
            p2, p98 = np.percentile(values, [2, 98])
            self.i_min = float(p2)
            self.i_max = float(p98)
            self.i_ww = p98 - p2
            self.i_wl = (p98 + p2) / 2.0
        self.i_min_text.set_val(f"{self.i_min:.0f}")
        self.i_max_text.set_val(f"{self.i_max:.0f}")
        self.i_ww_text.set_val(f"{self.i_ww:.0f}")
        self.i_wl_text.set_val(f"{self.i_wl:.0f}")
        self.update()

    def launch(self):
        _select_interactive_backend()

        # Probe early so headless/CI hosts get a clear hint instead of a
        # confusing TclError deep inside widget construction.
        try:
            probe = plt.figure()
            plt.close(probe)
        except Exception as e:
            print("ERROR: no interactive GUI backend available (headless session?).")
            print("  The pipeline outputs are saved as NIfTI files; you can view them with")
            print("  ITK-SNAP / ImageJ or `pip install ctpamap[napari]` + `ctpamap-viewern`.")
            raise SystemExit(1) from e

        plt.rcParams.update({
            "figure.facecolor": "#000000",
            "axes.facecolor": "#000000",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "font.family": "sans-serif",
        })

        self.fig = plt.figure(figsize=(12, 8), facecolor="#000000")

        self.cax = self.fig.add_axes([0.87, 0.02, 0.02, 0.94], facecolor="#000000")

        idx = self.pos[self._view_axis]
        comp, w, h = self._get_view(self.view_mode, idx)
        self.ax = self.fig.add_axes([0.15, 0.02, 0.70, 0.94], facecolor="#000000")
        self.im = self.ax.imshow(comp, origin="lower",
                                 interpolation='bilinear' if self.interpolate else 'nearest')
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors="white")

        self._slider_ax = self.fig.add_axes([0.90, 0.02, 0.03, 0.94], facecolor="#000000")
        self.slider = Slider(
            self._slider_ax, "Slice", 0, self._view_size - 1,
            valinit=idx, valstep=1, color="#4c8eda",
            orientation="vertical",
        )
        self.slider.on_changed(self._on_slider_single)

        ax_imin = self.fig.add_axes([0.02, 0.86, 0.10, 0.04], facecolor="#444444")
        ax_imax = self.fig.add_axes([0.02, 0.80, 0.10, 0.04], facecolor="#444444")
        ax_iww = self.fig.add_axes([0.02, 0.72, 0.10, 0.04], facecolor="#444444")
        ax_iwl = self.fig.add_axes([0.02, 0.66, 0.10, 0.04], facecolor="#444444")
        ax_alpha = self.fig.add_axes([0.02, 0.56, 0.10, 0.04], facecolor="#000000")
        ax_mpr = self.fig.add_axes([0.02, 0.48, 0.10, 0.04], facecolor="#000000")

        self.i_min_text = TextBox(
            ax_imin, "I min", initial=f"{self.i_min:.0f}",
            color="#444444", hovercolor="#555555",
        )
        self.i_min_text.on_submit(self.on_i_min_submit)

        self.i_max_text = TextBox(
            ax_imax, "I max", initial=f"{self.i_max:.0f}",
            color="#444444", hovercolor="#555555",
        )
        self.i_max_text.on_submit(self.on_i_max_submit)

        self.i_ww_text = TextBox(
            ax_iww, "I WW", initial=f"{self.i_ww:.0f}",
            color="#444444", hovercolor="#555555",
        )
        self.i_ww_text.on_submit(self.on_i_ww_submit)

        self.i_wl_text = TextBox(
            ax_iwl, "I WL", initial=f"{self.i_wl:.0f}",
            color="#444444", hovercolor="#555555",
        )
        self.i_wl_text.on_submit(self.on_i_wl_submit)

        alpha_slider = Slider(
            ax_alpha, "α", 0.0, 1.0,
            valinit=self.alpha, valstep=0.05, color="#4c8eda",
        )
        alpha_slider.on_changed(self.on_alpha_change)

        self.mpr_btn = Button(ax_mpr, "MPR", color="#444444", hovercolor="#666666")
        self.mpr_btn.on_clicked(self.toggle_mpr)

        ax_interp = self.fig.add_axes([0.02, 0.40, 0.10, 0.04], facecolor="#000000")
        self.interp_btn = Button(ax_interp, "Smooth", color="#444444", hovercolor="#666666")
        self.interp_btn.on_clicked(self.toggle_interp)

        ax_auto = self.fig.add_axes([0.02, 0.32, 0.10, 0.04], facecolor="#000000")
        self.auto_btn = Button(ax_auto, "Auto", color="#444444", hovercolor="#666666")
        self.auto_btn.on_clicked(self.auto_ww_wl)

        ax_sag = self.fig.add_axes([0.02, 0.24, 0.10, 0.04], facecolor="#000000")
        self.sag_btn = Button(ax_sag, "Sag", color="#444444", hovercolor="#666666")
        self.sag_btn.on_clicked(self.set_view_sagittal)

        ax_cor = self.fig.add_axes([0.02, 0.16, 0.10, 0.04], facecolor="#000000")
        self.cor_btn = Button(ax_cor, "Cor", color="#444444", hovercolor="#666666")
        self.cor_btn.on_clicked(self.set_view_coronal)

        ax_flip = self.fig.add_axes([0.02, 0.08, 0.10, 0.04], facecolor="#000000")
        self.flip_btn = Button(ax_flip, "Flip LR", color="#444444", hovercolor="#666666")
        self.flip_btn.on_clicked(self.toggle_flip)

        self._scroll_cid = self.fig.canvas.mpl_connect(
            "scroll_event", self._on_scroll_single)

        self.update()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Subtraction CTPA Iodine Map Viewer"
    )
    parser.add_argument("ctpa", help="Post-contrast CTPA NIfTI")
    parser.add_argument("iodine", help="Iodine map clean NIfTI")
    args = parser.parse_args()

    viewer = SCTViewer(args.ctpa, args.iodine)
    viewer.launch()


if __name__ == "__main__":
    main()
