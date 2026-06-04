from math import exp
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from pylinac.core.profile import SingleProfile, Interpolation, Normalization

from .wtscans import XyProfile


LINE_KWARGS = dict(colors='C4', linestyles='dashed', linewidth=1)


def plot_tangent(a: list, b: list, forward: int, back: int, modality: str = None, axis: plt.Axes = None) -> None:
    """
    Plot a tangent line between two points and extend it.
    """

    slope = (a[1] - b[1]) / (a[0] - b[0])

    x_range = np.linspace(a[0] - back, b[0] + forward, 100)
    y = slope * (x_range - a[0]) + a[1]

    axis.plot(x_range, y, 'C4--', linewidth=1)

    if modality != 'EL':
        axis.plot(a[0], a[1], 'bx')
        axis.plot(b[0], b[1], 'bx')


class ArrayScanPlotting:

    def profile_plot(self, data: dict, plane: str, axis: plt.Axes = None) -> None:

        d = data[plane]

        axis.plot(d['Profile_pos'], d['Profile_val'])

        axis.scatter(
            d['CaxDev'],
            d['norm_value'],
            marker='x',
            color='red'
        )

        axis.grid(True, which="both", ls="-")

        axis.set_ylim(0, np.max(d['Profile_val']) * 1.10)

        axis.set_title(
            f"{plane.replace('_', ' ').title()} "
            f"{d['Energy']}{d['Modality']}"
        )

        axis.set_xlabel('Distance (mm)')
        axis.set_ylabel('Response (arb.)')

        self.plot_penumbra(
            penumbra=d['Penumbra'],
            axis=axis
        )

        self.plot_symmetry(
            d['FWHM'],
            d['norm_value'],
            d['Filter'],
            axis=axis
        )

        self.plot_fwhm(
            d['FWHM'],
            d['norm_value'],
            axis=axis
        )

        if d['Filter'] == 'FF':
            self.plot_flatness(profile=d['Profile_val'], axis=axis)
        else:
            if d['Nominal_field_size'] == 100:
                plot_tangent(a=d['slopes']['Left 30%'], b=d['slopes']['Left 60%'], forward=20, back=20, axis=axis)
                plot_tangent(a=d['slopes']['Right 30%'], b=d['slopes']['Right 60%'], forward=20, back=20, axis=axis)

            else:
                plot_tangent(a=d['slopes']['Left 30%'], b=d['slopes']['Left 60%'], forward=75, back=40, axis=axis)
                plot_tangent(a=d['slopes']['Right 30%'], b=d['slopes']['Right 60%'], forward=40, back=75, axis=axis)

    @staticmethod
    def plot_flatness(profile: np.ndarray, axis: plt.Axes) -> None:
        """
        Plot flatness parameters.
        """
        prof = SingleProfile(profile, None, Interpolation.NONE, False, 0.1, 10, Normalization.BEAM_CENTER)
        data = prof.field_data(in_field_ratio=0.8)
        field_vals = data['field values']

        fmax = field_vals.max() * 100
        fmin = field_vals.min() * 100

        axis.axhline(fmax, color='g', linestyle='dashed', linewidth=1)
        axis.axhline(fmin, color='g', linestyle='dashed', linewidth=1)

    @staticmethod
    def plot_symmetry(fwhm: dict, peak: float, filter_type: str, axis: plt.Axes) -> None:
        """
        Plot symmetry parameters.
        """

        if filter_type == 'FFF':
            axis.vlines(fwhm['left_80%'][0], fwhm['left_80%'][1], peak, **LINE_KWARGS)
            axis.vlines(fwhm['right_80%'][0], fwhm['right_80%'][1], peak, **LINE_KWARGS)
            axis.hlines((peak - fwhm['left_80%'][1]) * 0.5 + fwhm['left_80%'][1], fwhm['left_80%'][0],
                        fwhm['right_80%'][0], **LINE_KWARGS)
        else:
            axis.scatter(fwhm['left_80%'][0], fwhm['left_80%'][1], marker='x', color='red')
            axis.scatter(fwhm['right_80%'][0], fwhm['right_80%'][1], marker='x', color='red')

    @staticmethod
    def plot_fwhm(fwhm: dict, peak: float, axis: plt.Axes) -> None:
        """
        Plot FWHM parameters.
        """

        ymax = peak * 0.75

        axis.vlines(x=fwhm['left_index'], ymin=0, ymax=ymax, **LINE_KWARGS)
        axis.vlines(x=fwhm['right_index'], ymin=0, ymax=ymax, **LINE_KWARGS)
        axis.hlines(y=50, xmin=fwhm['left_index'], xmax=fwhm['right_index'], **LINE_KWARGS)

    @staticmethod
    def plot_penumbra(penumbra: dict, axis: plt.Axes = None) -> None:
        """
        Plot penumbra regions.
        """

        p = penumbra[2]
        axis.axvspan(p['left 20% index (exact)'], p['left 80% index (exact)'], alpha=0.5, color='pink')
        axis.axvspan(p['right 20% index (exact)'], p['right 80% index (exact)'], alpha=0.5, color='pink')


class DetectorArray:

    def __init__(self, meta_data, array_data, **kwargs):

        normalise_profile = kwargs.pop('normalise_profile')
        self.scale = kwargs.pop('scale')

        self.meta_data = meta_data

        scans = self.get_data(array_data)

        self.grid = self.create_grid(scans)

        rows = self.grid.shape[0]

        size = self.scale * rows

        if self.scale > 1:
            if size % 2 == 0:
                size += 1

            mag = size / rows
            self.grid = transform.rescale(self.grid, mag, order=1, preserve_range=True, anti_aliasing=False)

        position = np.linspace(-130, 130, size)

        crossplane = self.get_profile('CROSSPLANE_PROFILE')
        inplane = self.get_profile('INPLANE_PROFILE')

        meta = meta_data[1]

        def calc_profiles(values, _position, scan_type):
            data = [
                scan_type,
                {
                    "Type": scan_type,
                    "MODALITY": meta["MODALITY"],
                    "ENERGY": meta["ENERGY"],
                    "FILTER": meta["FILTER"],
                    "ISOCENTER": 0.0,
                    "SSD": meta["SSD"],
                    "SCAN_DEPTH": meta["SCAN_DEPTH"],
                    "SCAN_OFFAXIS_INPLANE": meta["SCAN_OFFAXIS_INPLANE"],
                    "FIELD_INPLANE": meta['FIELD_INPLANE'],
                    "FIELD_CROSSPLANE": meta['FIELD_CROSSPLANE'],
                },
                {
                    "Values": values,
                    "Position": _position
                }
            ]

            return XyProfile(data, normalise_profile)

        self.crossplane = calc_profiles(crossplane, position, 'CROSSPLANE_PROFILE').results
        self.inplane = calc_profiles(inplane, position, 'INPLANE_PROFILE').results

    def get_profile(self, profile: str) -> np.ndarray:
        """
        Return a 1D profile through the center.
        """

        if profile == 'CROSSPLANE_PROFILE':
            hor_mid = self.grid.shape[0] // 2
            return self.grid[hor_mid, :]
        vert_mid = self.grid.shape[1] // 2

        return self.grid[:, vert_mid]

    @staticmethod
    def get_data(data: list) -> list:
        """
        Convert detector rows into interpolation format.
        """

        return [[profile[:, 0], profile[:, 1]] for profile in data]

    @staticmethod
    def create_grid(data: list) -> np.ndarray:
        """
        Create interpolated detector grid.
        """

        dist = np.linspace(-130, 130, 53)

        return np.array([np.interp(dist, item[0], item[1]) for item in data])

    @staticmethod
    def gaussian_lp(d0: int, img_shape: tuple) -> np.ndarray:
        """
        Vectorized 2D Gaussian low-pass filter.
        """

        rows, cols = img_shape[:2]

        y, x = np.ogrid[:rows, :cols]

        cy = rows / 2
        cx = cols / 2

        dist_sq = (x - cx) ** 2 + (y - cy) ** 2

        return np.exp(-dist_sq / (2 * d0 ** 2))

    def fourier_filter(self, array: np.ndarray, d0: int) -> np.ndarray:
        """
        Apply Gaussian low-pass Fourier filter.        """

        fft = np.fft.fftshift(np.fft.fft2(array))
        fft *= self.gaussian_lp(d0, array.shape)
        return np.fft.ifft2(np.fft.ifftshift(fft)).real