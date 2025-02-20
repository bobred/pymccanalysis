from math import sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from scipy import signal
from pylinac.core.profile import SingleProfile, Interpolation, Normalization
from .wtscans import XyProfile


def plot_tangent(a: list, b: list, forward: int, back: int, modality: str = None, axis: plt.Axes = None) -> None:
    """
    Plats a line between two given points and extends the line past these points.

    Parameters
    ----------
    a : list
        First point of line.
    b : list
        Second point of line.
    forward: int
        How far to extend the line forward.
    back: int
        How far to extend the line backward.
    modality: str
        Photon or electron.
    axis : Matplotlib.Axes
        Plot line to a given axis

    Returns
    -------
    None.

    """
    slope = (a[1] - b[1]) / (a[0] - b[0])
    xrange = np.linspace(a[0] - back, b[0] + forward, 100)
    y = slope * (xrange - a[0]) + a[1]
    axis.plot(xrange, y, 'C4--', linewidth=1)
    if modality != 'EL':
        axis.plot(a[0], a[1], 'bx')
        axis.plot(b[0], b[1], 'bx')
    pass


class ArrayScanPlotting:

    def __init__(self):
        pass

    def profile_plot(self, data: dict, plane: str, axis: plt.Axes = None) -> None:
        axis.plot(data[plane]['Profile_pos'], data[plane]['Profile_val'])
        axis.scatter(data[plane]['CaxDev'], data[plane]['norm_value'], marker='x', color='red')
        axis.grid(True, which="both", ls="-")
        axis.set_ylim(0, np.max(data[plane]['Profile_val']) * 1.10)
        axis.set_title(plane.replace('_', ' ').title() + ' ' + str(data[plane]['Energy']) + data[plane]['Modality'])
        axis.set_xlabel('Distance (mm)')
        axis.set_ylabel('Response (arb.)')

        self.plot_penumbra(penumbra=data[plane]['Penumbra'], axis=axis)
        self.plot_symmetry(data[plane]['FWHM'], data[plane]['norm_value'], data[plane]['Filter'], axis=axis)
        self.plot_fwhm(data[plane]['FWHM'], data[plane]['norm_value'], axis=axis)

        if data[plane]['Filter'] == 'FF':
            self.plot_flatness(profile=data[plane]['Profile_val'], axis=axis)
        else:
            if data[plane]['Nominal_field_size'] == 100:
                plot_tangent(data[plane]['slopes']['Left 30%'], data[plane]['slopes']['Left 60%'], 20, 20, axis=axis)
                plot_tangent(data[plane]['slopes']['Right 30%'], data[plane]['slopes']['Right 60%'], 20, 20, axis=axis)
            else:
                plot_tangent(data[plane]['slopes']['Left 30%'], data[plane]['slopes']['Left 60%'], 75, 40, axis=axis)
                plot_tangent(data[plane]['slopes']['Right 30%'], data[plane]['slopes']['Right 60%'], 40, 75, axis=axis)
            pass

    @staticmethod
    def plot_flatness(profile: np.array, axis: plt.Axes) -> None:
        """Plot flatness parameters. Applies to both flatness dose ratio and dose difference."""
        _profile = SingleProfile(profile, None, Interpolation.NONE, False, 0.1, 10, Normalization.BEAM_CENTER)
        data = _profile.field_data(in_field_ratio=0.8)
        axis.axhline(np.max(data['field values'])*100, color='g', linestyle='dashed', linewidth=1)
        axis.axhline(np.min(data['field values'])*100, color='g', linestyle='dashed', linewidth=1)

    @staticmethod
    def plot_symmetry(fwhm: dict, peak: float, filter_type: str, axis: plt.Axes) -> None:
        """Plot symmetry parameters."""
        if filter_type == 'FFF':
            axis.vlines(x=fwhm['left_80%'][0], ymin=fwhm['left_80%'][1], ymax=peak, colors='C4', linestyles='dashed',
                        linewidth=1)
            axis.vlines(x=fwhm['right_80%'][0], ymin=fwhm['right_80%'][1], ymax=peak, colors='C4', linestyles='dashed',
                        linewidth=1)
            axis.hlines(y=(peak - fwhm['left_80%'][1]) * 0.5 + fwhm['left_80%'][1], xmin=fwhm['left_80%'][0],
                        xmax=fwhm['right_80%'][0], colors='C4', linestyles='dashed', linewidth=1)
        else:
            axis.scatter(fwhm['left_80%'][0], fwhm['left_80%'][1], marker='x', color='red')
            axis.scatter(fwhm['right_80%'][0], fwhm['right_80%'][1], marker='x', color='red')

    @staticmethod
    def plot_fwhm(fwhm: dict, peak: float, axis: plt.Axes) -> None:
        """Plot FWHM parameters."""
        axis.vlines(x=fwhm['left_index'], ymin=0, ymax=peak*0.75, colors='C4', linestyles='dashed', linewidth=1)
        axis.vlines(x=fwhm['right_index'], ymin=0, ymax=peak*0.75, colors='C4', linestyles='dashed', linewidth=1)
        axis.hlines(y=50, xmin=fwhm['left_index'], xmax=fwhm['right_index'], colors='C4', linestyles='dashed',
                    linewidth=1)

    @staticmethod
    def plot_penumbra(penumbra: dict, axis: plt.Axes = None) -> None:
        """Plot the non-linear regression fit against the profile"""
        axis.axvspan(penumbra[2]['left 20% index (exact)'],
                     penumbra[2]['left 80% index (exact)'], alpha=0.5, color='pink')
        axis.axvspan(penumbra[2]['right 20% index (exact)'],
                     penumbra[2]['right 80% index (exact)'], alpha=0.5, color='pink')


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
            if (size % 2) == 0:  # Ensures the rescaled grid is odd so that zero is an actual value.
                size = size + 1
                mag = size / rows
            else:
                mag = size / rows
            self.grid = transform.rescale(self.grid, mag)

        #self.grid = self.fourier_filter(self.grid, 10)
        position = np.linspace(-130, 130, size)
        crossplane = self.get_profile('CROSSPLANE_PROFILE')
        inplane = self.get_profile('INPLANE_PROFILE')

        def calc_profiles(values, _position, scan_type):
            _data = [scan_type, {"Type": scan_type, "MODALITY": meta_data[1]["MODALITY"],
                                 "ENERGY": meta_data[1]["ENERGY"], "FILTER": meta_data[1]["FILTER"],
                                 "ISOCENTER": 0.0, "SSD": meta_data[1]["SSD"],
                                 "SCAN_DEPTH": meta_data[1]["SCAN_DEPTH"],
                                 "SCAN_OFFAXIS_INPLANE": meta_data[1]["SCAN_OFFAXIS_INPLANE"],
                                 'FIELD_INPLANE': meta_data[1]['FIELD_INPLANE'],
                                 'FIELD_CROSSPLANE': meta_data[1]['FIELD_CROSSPLANE'],
                                 'ARRAY': self.array_profile},
                     {"Values": values, "Position": _position}]
            return XyProfile(_data, normalise_profile)

        self.crossplane = calc_profiles(crossplane, position, 'CROSSPLANE_PROFILE').results
        self.inplane = calc_profiles(inplane, position, 'INPLANE_PROFILE').results

        pass

    def get_profile(self, profile: str) -> np.array:
        """
        Returns a 1D profile through the center, either horizontal or vertical.

        Parameters
        ----------
        profile : str
            Profile type, PDD, Cross plane or inplane.

        Returns
        -------
        numpy array.

        """
        if profile == 'CROSSPLANE_PROFILE':
            hor_mid = int(round(self.grid.shape[0] * 0.5))
            return self.grid[hor_mid, :]
        else:
            vert_mid = int(round(self.grid.shape[1] * 0.5))
            return self.grid[:, vert_mid]

    @staticmethod
    def get_data(data: list) -> list:
        """
        .

        Parameters
        ----------
        data : list
            Profile type, PDD, Cross plane or inplane.

        Returns
        -------
        list.

        """
        scans = []
        for profile in data:
            scans.append([profile[:, 0], profile[:, 1]])
        return scans

    @staticmethod
    def create_grid(data: list) -> np.array:
        """
        Take a list of each row of the detector, where each row is a list of position and data.
        Returns a numpy 2D grid of data points.

        Parameters
        ----------
        data : list
            .list of each row of the detector, where each row is a list of position and data.

        Returns
        -------
        numpy array.

        """
        dist = np.linspace(-130, 130, 53)
        arr = []
        for item in data:
            arr.append(np.interp(dist, item[0], item[1]))
        _grid = np.asarray(arr)
        return _grid

    def distance(self, point1: tuple, point2: tuple) -> float:
        """
            Function to calculate the Euclidian distance between two points.
            Parameters
            ----------
            point1 : tuple
                Sigma value for the Gaussian.
            point2 : tuple
                Dimensions of the array.
            Returns
            -------
            float
                Distance.
        """
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def gaussian_lp(self, d0: int, img_shape: tuple) -> np.array:
        """
            2D Gaussinan function that matches the array size and with a spread governed by d0.

            Parameters
            ----------
            d0 : int
                Sigma value for the Gaussian.
            img_shape : tuple
                Dimensions of the array.
            Returns
            -------
            np.array
                2D Gaussian array.
        """
        base = np.zeros(img_shape[:2])
        rows, cols = img_shape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                base[y, x] = exp(((-self.distance((y, x), center) ** 2) / (2 * (d0 ** 2))))
        return base

    def fourier_filter(self, array: np.array, d0: int) -> np.array:
        """
            Function calculates the 2D Fourier transform of the array, applies a Gaussian low pass filter, takes the
            inverse Fourier transform and returns the array.
            The FT needs to be shifted so that the DC, or zero frequency value is at the center. The Gaussian is then
            applied to the center of the FT and then the FT needs to be shifted before the inverse is calculated.

            Parameters
            ----------
            array : np.array
                2D array of the profile.
            d0 : int
                Sigma value for the Gaussian.

            Returns
            -------
            np.array
                Filtered array.
        """
        # Compute the 2d FFT of the image
        original = np.fft.fft2(array)
        # Shift the zero frequency to the centre of the 2d FFT
        center = np.fft.fftshift(original)
        # Multiply the centered spectrum by a 2D Gaussian of sigma D0
        low_pass_center = center * self.gaussian_lp(d0, array.shape)
        # Shift the spectrum from the centre
        low_pass = np.fft.ifftshift(low_pass_center)
        # Calculate the inverse of the spectrum, smoothed image.
        inverse_low_pass = np.fft.ifft2(low_pass).real
        return inverse_low_pass
