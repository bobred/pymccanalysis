import numpy as np
from scipy.stats import linregress
from pylinac import field_analysis
from pylinac.core.profile import SingleProfile, Interpolation, Normalization, Edge
import matplotlib.pyplot as plt
import pandas as pd


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


class WTScanPlotting:

    def __init__(self):
        pass

    @staticmethod
    def pdd_plot(data: dict, axis: plt.Axes = None) -> None:
        if data['PDD']['Modality'] == 'EL':
            axis.plot(data['PDD']['PDD_pos'], data['PDD']['PDD_val'])
            axis.scatter(data['PDD']['R100'], 100, marker='x', color='red')
            axis.scatter(data['PDD']['R90'], 90, marker='x', color='red')
            axis.scatter(data['PDD']['R80'], 80, marker='x', color='red')
            axis.scatter(data['PDD']['R50'], 50, marker='x', color='red')
            axis.scatter(data['PDD']['R30'], 30, marker='x', color='red')
            axis.grid(True, which="both", ls="-")
            axis.set_ylim(0, np.max(data['PDD']['PDD_val']) * 1.1)
            axis.set_xlim(xmin=0)
            axis.set_xlabel('Distance (mm)')
            axis.set_ylabel('Response (%)')
            axis.set_title('PDD ' + str(data['PDD']['Energy']) + data['PDD']['Modality'])
            plot_tangent([data['PDD']['R60'], 60], [data['PDD']['R40'], 40], 20, 100, 'EL', axis=axis)
            plot_tangent([data['PDD']['PDD_pos'][-100], data['PDD']['PDD_val'][-100]], [data['PDD']['PDD_pos'][-10],
                         data['PDD']['PDD_val'][-10]], 10, 200, 'EL', axis=axis)
            axis.vlines(x=data['PDD']['Rp'][0], ymin=0, ymax=20, colors='C4', linestyles='dashed', linewidth=1)
        else:
            axis.plot(data['PDD']['PDD_pos'], data['PDD']['PDD_val'])
            axis.scatter(data['PDD']['R100'], 100, marker='x', color='red')
            axis.scatter(100, data['PDD']['D100'], marker='x', color='red')
            axis.scatter(200, data['PDD']['D200'], marker='x', color='red')
            axis.grid(True, which="both", ls="-")
            axis.set_xlabel('Distance (mm)')
            axis.set_ylabel('Response (%)')
            axis.set_ylim(0, np.max(data['PDD']['PDD_val']) * 1.1)
            axis.set_xlim(xmin=0)
            axis.set_title('PDD ' + str(data['PDD']['Energy']) + data['PDD']['Modality'])

    def profile_plot(self, data: dict, plane: str, axis: plt.Axes = None) -> None:
        axis.plot(data['Profile_pos'], data['Profile_val'])
        axis.scatter(data['CaxDev'], data['norm_value'], marker='x', color='red')
        axis.grid(True, which="both", ls="-")
        axis.set_ylim(0, np.max(data['Profile_val']) * 1.10)
        axis.set_title(plane.replace('_', ' ').title() + ' ' + str(data['Energy']) + data['Modality'])
        axis.set_xlabel('Distance (mm)')
        axis.set_ylabel('Response (arb.)')

        self.plot_penumbra(penumbra=data['Penumbra'], axis=axis)
        self.plot_symmetry(data['FWHM'], data['norm_value'], data['Filter'], axis=axis)
        self.plot_fwhm(data['FWHM'], data['norm_value'], axis=axis)

        if data['Filter'] == 'FF':
            self.plot_flatness(profile=data['Profile_val'], axis=axis)
        else:
            if data['Nominal Field Size'] == 100:
                plot_tangent(data['slopes']['Left 30%'], data['slopes']['Left 60%'], 20, 20, axis=axis)
                plot_tangent(data['slopes']['Right 30%'], data['slopes']['Right 60%'], 20, 20, axis=axis)
            else:
                plot_tangent(data['slopes']['Left 30%'], data['slopes']['Left 60%'], 75, 40, axis=axis)
                plot_tangent(data['slopes']['Right 30%'], data['slopes']['Right 60%'], 40, 75, axis=axis)
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


class PDD:

    def __init__(self, data, normalise: bool = False, ion_to_dose: bool = False):
        self.results = {}
        self.curve_type = data[0]
        self.modality = data[1]['MODALITY']
        self.energy = data[1]['ENERGY']
        if 'FILTER' in data[1]:
            self.filter = data[1]['FILTER']
        self.ion_to_dose = ion_to_dose

        self.meas_values = data[2]['Values'].to_numpy()
        self.position = data[2]['Position'].to_numpy()
        if normalise:
            self.normalise()

        if self.modality == 'X':
            self.results["Type"] = self.curve_type
            self.results["Modality"] = self.modality
            self.results["Energy"] = self.energy
            self.results["Filter"] = self.filter
            self.results["Nominal Field Size"] = data[1]['FIELD_CROSSPLANE']
            self.results["D100"] = self.dose_x(100.0)
            self.results["D200"] = self.dose_x(200)
            self.results["R100"] = self.depth_max()
            self.results["R80"] = self.depth_x(80)
            self.results["R50"] = self.depth_x(50)
            self.results["Q Index"] = self.calc_q_index()
            self.results["Surface Dose"] = self.calc_surface_dose()
            self.results["PDD_pos"] = self.position
            self.results["PDD_val"] = self.meas_values
        elif self.modality == "EL":
            rp = self.calc_rp()
            self.results["Type"] = self.curve_type
            self.results["Modality"] = self.modality
            self.results["Energy"] = self.energy
            self.results["R100"] = self.depth_max()
            self.results["R50"] = self.depth_x(50.0)
            self.results["R30"] = self.depth_x(30.0)
            self.results["R40"] = self.depth_x(40.0)
            self.results["R60"] = self.depth_x(60.0)
            self.results["R80"] = self.depth_x(80.0)
            self.results["R90"] = self.depth_x(90.0)
            self.results["Rp"] = [rp, self.dose_x(rp) * 100]
            self.results["Ds"] = self.dose_x(0.5) * 100
            self.results["PDD_pos"] = self.position
            self.results["PDD_val"] = self.meas_values
        pass

    @staticmethod
    def interp_value(point_a: tuple, point_b: tuple, interp_y: float) -> float:
        """
        Calculate an interpolated "x" value between two points.

        Parameters
        ----------
        point_a : float
            First point for interpolation.
        point_b : float
            Second point for interpolation.
        interp_y : float
            Position at which to interpolate the points to.

        Returns
        -------
        interp_x : float
            interpolated "x value at .

        """
        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        intersect = point_a[1] - slope * point_a[0]
        # y value at x
        interp_x = (interp_y - intersect) / slope

        return interp_x

    def normalise(self):
        _max = np.max(self.meas_values)
        self.meas_values = 100 * (self.meas_values / _max)

    def depth_max(self) -> float:
        """
        function that returns the position of the maximum of the pdd in mm

        Parameters
        ----------

        Returns
        -------
        max_depth : float
            depth of maximum dose (smoothed if interp = True).

        """
        idx_max = np.argmax(self.meas_values)
        max_depth = self.position[idx_max]

        return max_depth

    def depth_x(self, x: float) -> float:
        """
        function that returns the position of x% of the max of the pdd with
        linear interpolation between data points (in case of a steep gradient)
        in mm.

        Parameters
        ----------
        x : float
            percentage value from 0 to 100%.

        Returns
        -------
        x_depth : float
            depth where the dose dropped to x percent from maximum.

        """
        max_dose = self.meas_values.max()
        x_val = max_dose * x / 100
        i = np.flip(self.meas_values)
        idx_x_depth = np.searchsorted(i, x_val)

        b_1 = (np.flip(self.position)[idx_x_depth - 1], np.flip(self.meas_values)[idx_x_depth - 1])
        b_2 = (np.flip(self.position)[idx_x_depth], np.flip(self.meas_values)[idx_x_depth])

        x_depth = self.interp_value(b_1, b_2, x_val)

        return x_depth

    def dose_x(self, depth: float) -> float:
        """
        returns the percent dose value at x mm depth

        Parameters
        ----------
        depth : float
            depth in mm where the percent value gets returned.

        Returns
        -------
        dx : float
            dose in percent at x mm depth.

        """

        idx = np.searchsorted(self.position, depth)
        dx = self.meas_values[idx]

        return dx

    def calc_surface_dose(self) -> float:
        """
        function that returns the relative surface dose at 0.5 mm depth (in %).
        Takes advantage of the already interpolated values from the dataframe

        Returns
        -------
        rel_surface_dose : float
            relative dose at 0.5 mm depth.

        """

        # only works if values are already interpolated with step size 0.1 mm
        surface_dose = self.meas_values[5]
        max_dose = self.meas_values.max()

        rel_surface_dose = surface_dose / max_dose * 100

        return rel_surface_dose

    def calc_q_index(self) -> float:
        """
        calculate Q_index following IAEA TRS 398

        Returns
        -------
        qi : float
            calculated Q index.

        """

        qi = (1.2661 * self.results["D200"] / self.results["D100"]) - 0.0595

        return qi

    def calc_rp(self) -> float:
        """
        Calculate practical Range of an electron beam. The practical range is
        defined as the intersection of the tangent at 50% dose and the linear
        regression line of the Bremsstrahlungs tail.

        Returns
        -------
        Rp : float
            practical range of the electron beam.

        """
        max_dose = self.meas_values.max()

        # get 40/60% data points (x,y) for slope calculation
        a1 = (self.depth_x(60.0), 0.6 * max_dose)
        a2 = (self.depth_x(40.0), 0.4 * max_dose)

        # calculate slope:
        slope_50 = (a1[1] - a2[1]) / (a1[0] - a2[0])

        # intercept
        a50 = (self.depth_x(50.0), 0.5 * max_dose)
        inter_50 = a50[1] - (a50[0] * slope_50)

        # define data range for linear regression, start with an estimated
        # practical Range (formula from PTW data analyze handbook)
        e0_mean = self.calc_e0_mean()
        rp_est = (0.11 + 0.505 * e0_mean - 3E-4 * e0_mean ** 2) * 100
        # index equals steps of 0.1 mm

        # add safety distance behind Rp (slope should be shallow here)
        lin_start = int(rp_est + 100)

        # for debugging
        # print(lin_start)
        # print(self.dataframe.position.size - 2)

        # if there are enough data points do a linear regression
        if lin_start < (self.position.size - 2):
            # self.dataframe[lin_start:].meas_values.plot()

            # do linear regression on Dataframe Series
            lr = linregress(self.position[lin_start:], self.meas_values[lin_start:])
            inter_bs = lr.intercept
            slope_bs = lr.slope
        # else, use the last point as an approximation for the intercept
        # with 0 slope
        else:
            inter_bs = self.meas_values[-1]
            slope_bs = 0

        # Rp is the depth of the point of intersection
        rp = (inter_bs - inter_50) / (slope_50 - slope_bs)

        return rp

    def calc_e0_mean(self) -> float:
        """
        Estimate of the mean Electron energy at the phantom surface using the
        R50 dose to water.

        Returns
        -------
        E0_mean : float
            estimated mean electron energy.

        """
        e0_mean = 2.33 * self.calc_r50d_ipem() / 10

        return e0_mean

    def calc_r50d_ipem(self) -> float:
        """
        function that returns the R50 value (dose to water) according to
        Varian

        Returns
        -------
        R50 : float
            Dose to water at 50% maximum dose.

        """
        # half maximum
        d50ion = self.meas_values.max() / 2

        # searchsorted is a binary search and expects a sorted series
        # hence we need invert the curve.
        idx_d50ion = np.searchsorted(np.flip(self.meas_values), d50ion)

        b_1 = (np.flip(self.position)[idx_d50ion - 1],
               np.flip(self.meas_values)[idx_d50ion - 1])
        b_2 = (np.flip(self.position)[idx_d50ion],
               np.flip(self.meas_values)[idx_d50ion])

        r50ion = self.interp_value(b_1, b_2, d50ion)

        r50d = 1.029 * r50ion - 0.063

        return r50d


class XyProfile:

    def __init__(self, data, normalisation: Normalization = Normalization.BEAM_CENTER):
        self.results = {}
        self.re_norm_percent = None
        if isinstance(data[2]['Values'], list) or isinstance(data[2], pd.DataFrame):
            self.meas_values = data[2]['Values'].to_numpy()
        else:
            self.meas_values = data[2]['Values']
        if isinstance(data[2]['Position'], list) or isinstance(data[2], pd.DataFrame):
            self.position = data[2]['Position'].to_numpy()
        else:
            self.position = data[2]['Position']
        self.curve_type = data[0]
        self.modality = data[1]['MODALITY']
        self.energy = data[1]['ENERGY']
        if 'ARRAY' in data[1]:
            self.detector_array = data[1]['ARRAY']
        else:
            self.detector_array = False
        self.filter = data[1]['FILTER']
        self.isocenter = data[1]['ISOCENTER']
        self.ssd = data[1]['SSD']
        self.scan_depth = data[1]['SCAN_DEPTH']
        self.normalisation = normalisation
        if self.curve_type == 'INPLANE_PROFILE':
            self.nominal_field_size = data[1]['FIELD_INPLANE']
        if self.curve_type == 'CROSSPLANE_PROFILE':
            self.nominal_field_size = data[1]['FIELD_CROSSPLANE']
        self.offset = 0
        if self.filter == "FFF" and self.energy == 6:
            self.norm_value = self.calc_fff_renormalisation_6x() * 100
        elif self.filter == "FFF" and self.energy == 10:
            self.norm_value = self.calc_fff_renormalisation_10x() * 100
        else:
            self.norm_value = 100
        self.field_width = self.calc_fwhm()

        if self.filter == "FF":
            self.results["Type"] = self.curve_type
            self.results["Modality"] = self.modality
            self.results["Energy"] = self.energy
            self.results["Filter"] = self.filter
            self.results["CaxDev"] = self.calc_cax_deviation()
            self.results["Scan depth"] = self.scan_depth
            self.results["FWHM"] = self.calc_fwhm()
            self.results["Flatness"] = self.calc_varian_flat()
            self.results["Symmetry"] = self.calc_sym()
            self.results["Penumbra"] = self.calc_penumbra()  # self.calc_penumbra_pylinac(),
            self.results["Profile_pos"] = self.position
            self.results["norm_value"] = 100
            self.results["Nominal Field Size"] = self.nominal_field_size
            self.results["Profile_val"] = self.normalise()
            self.results["Varian Acceptance Data"] = self.calc_varian_acceptance_data()
            self.results["Inplane_offaxis"] = data[1]['SCAN_OFFAXIS_INPLANE']
        elif self.filter == "FFF":
            slopes = self.calc_fff_slopes_peak()
            self.results["Type"] = self.curve_type
            self.results["Modality"] = self.modality
            self.results["Energy"] = self.energy
            self.results["Filter"] = self.filter
            self.results["CaxDev"] = self.calc_cax_deviation()
            self.results["Scan depth"] = self.scan_depth
            self.results["FWHM"] = self.calc_fwhm(max_type='max')
            self.results["Symmetry"] = self.calc_sym()
            self.results["Unflatness"] = self.calc_fff_unflatness()
            self.results["Peak"] = slopes["Slope peak"]
            self.results["Profile_pos"] = self.position
            self.results["Penumbra"] = self.calc_penumbra()  # self.calc_penumbra_pylinac(),
            self.results["norm_value"] = self.norm_value
            self.results["slopes"] = slopes
            self.results["Nominal Field Size"] = self.nominal_field_size
            self.results["Profile_val"] = self.normalise()
            self.results["Varian Acceptance Data"] = self.calc_varian_acceptance_data()
            self.results["Inplane_offaxis"] = data[1]['SCAN_OFFAXIS_INPLANE']
        pass

    @staticmethod
    def interp_value(point_a: tuple, point_b: tuple, interp_y: np.float64) -> np.float64:
        """
        Calculate an interpolated "x" value between two points.

        Parameters
        ----------
        point_a : np.float64
            First point for interpolation.
        point_b : np.float64
            Second point for interpolation.
        interp_y : np.float64
            Position at which to interpolate the points to.

        Returns
        -------
        interp_x : np.float64
            interpolated "x value at .

        """

        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])

        intersect = point_a[1] - slope * point_a[0]

        # x value at y
        interp_x = (interp_y - intersect) / slope
        return interp_x

    def calc_cax_deviation(self, max_type: str = 'cax') -> float:
        """
        Calculate the distance between the CAX and the center of the
        field that has been calculated.

        Parameters
        ----------
        max_type : str, optional
            Normalize to CAX Value or to absolute max. The default is 'cax'.

        Returns
        -------
        cax_dev : np.float64
            Cax deviation in mm.

        """
        half_max = self.calc_half_max(max_type=max_type)

        # find the index where half max should be inserted before to keep the
        # series sorted. Hence, value at _pos is always higher.
        left_pos = np.searchsorted(self.meas_values, half_max)

        a_1 = (self.position[left_pos - 1], self.meas_values[left_pos - 1])
        a_2 = (self.position[left_pos], self.meas_values[left_pos])

        right_pos = np.searchsorted(np.flip(self.meas_values), half_max)
        b_1 = (np.flip(self.position)[right_pos - 1], np.flip(self.meas_values)[right_pos - 1])
        b_2 = (np.flip(self.position)[right_pos], np.flip(self.meas_values)[right_pos])

        cax_dev = (self.interp_value(b_1, b_2, half_max) + self.interp_value(a_1, a_2, half_max)) * 0.5
        return cax_dev - self.offset

    def calc_half_max(self, max_type: str = 'cax') -> np.float64:
        """
        Return half of the max value or half the cax value depending on
        the given max_type. Default is to return half the CAX value. For FFF
        fields the value gets re-normalized to return the "correct" value.

        Parameters
        ----------
        max_type : str, optional
            Normalize to CAX Value or to absolute max. The default is 'cax'.

        Returns
        -------
        half_max: np.float64
            Half of the maximum. Either half the CAX or half of the absolute
            maximum.

        """

        if max_type == 'cax':
            max_val = self.meas_values[(self.position == self.offset)]
        else:
            max_val = self.meas_values.max()

        half_max = (max_val * 0.5)

        if self.filter == "FFF" and self.energy == 6:
            return half_max / self.calc_fff_renormalisation_6x()
        elif self.filter == "FFF" and self.energy == 10:
            return half_max / self.calc_fff_renormalisation_10x()
        return half_max

    def calc_fwhm(self, max_type: str = 'cax') -> dict:
        """
        Calculate the FWHM from data in the DataFrame and return the
        Fieldsize nominal and at isocenter distance.

        Parameters
        ----------
        max_type : str, optional
            normalize to cax value or to the absolute maximum.
            The default is 'cax'.

        Returns
        -------
        dict
            The 'fwhm (nominal)' result is the actual measurement length
            (distance the detector travelled), 'fhwm' is the field size at the
            isocenter.

        """

        half_max = self.calc_half_max(max_type=max_type)

        # find the index where half max should be inserted before to keep the
        # series sorted. Hence, value at _pos is always higher.
        left_pos = np.searchsorted(self.meas_values, half_max)
        a_1 = (self.position[left_pos - 1], self.meas_values[left_pos - 1])
        a_2 = (self.position[left_pos], self.meas_values[left_pos])

        right_pos = np.searchsorted(np.flip(self.meas_values), half_max)
        b_1 = (np.flip(self.position)[right_pos - 1],
               np.flip(self.meas_values)[right_pos - 1])
        b_2 = (np.flip(self.position)[right_pos],
               np.flip(self.meas_values)[right_pos])

        fwhm = self.interp_value(b_1, b_2, half_max) - self.interp_value(a_1, a_2, half_max)

        # correct for measurement depth so that Fieldsize is returned at iso
        iso_corr = self.isocenter / (self.ssd + self.scan_depth)

        _left_pos = self.interp_value(a_1, a_2, half_max)
        _right_pos = self.interp_value(b_1, b_2, half_max)

        idx_80_left = np.searchsorted(self.position, (0.0 - 0.8 * fwhm / 2))
        idx_80_right = np.searchsorted(self.position, (0.8 * fwhm / 2))

        profile = SingleProfile(self.meas_values, None, Interpolation.NONE, False, 0.1, 10,
                                self.normalisation)
        m = profile.values * self.norm_value

        left_80 = self.position[idx_80_left], m[idx_80_left]
        right_80 = self.position[idx_80_right], m[idx_80_right]

        fwhm_results = {'fwhm (nominal)': fwhm, 'fwhm': fwhm * iso_corr, 'left_index': _left_pos,
                        'right_index': _right_pos, 'left_80%': left_80, 'right_80%': right_80,
                        }
        return fwhm_results

    def calc_penumbra(self):
        meas_values = self.meas_values
        if self.filter == 'FF':
            profile = SingleProfile(meas_values, None, Interpolation.NONE, False, 0.1, 10, self.normalisation)
        else:
            profile = SingleProfile(meas_values, None, Interpolation.NONE, False, 0.1, 10, self.normalisation,
                                    edge_detection_method=Edge.INFLECTION_DERIVATIVE)

        ind = round(profile.values.shape[0] * 0.5)
        left = profile.values[0:ind] * self.norm_value
        right = profile.values[ind:-1] * self.norm_value
        s = np.interp(20, left, self.position[0:ind])
        t = np.interp(80, left, self.position[0:ind])
        u = np.interp(80, np.flip(right), np.flip(self.position[ind:-1]))
        v = np.interp(20, np.flip(right), np.flip(self.position[ind:-1]))
        pen = {"left 20% index (exact)": s,
               "left 80% index (exact)": t,
               "right 80% index (exact)": u,
               "right 20% index (exact)": v,
               }
        return t-s, v-u, pen

    def calc_fff_slopes_peak(self, center: bool = True) -> dict:
        """
        Calculate the left and right slopes of the fff profiles

        Parameters
        ----------
        center : bool, optional
            if True the positions are shifted by the cax deviation.
            The default is True.

        Returns
        -------
        slope_left : np.float64
            left slope of the field.
        slope_right : np.float64
            right slope of the field.
        peak_pos : np.float64
            Peak Position in mm.

        """

        # fff field with needed to find 1/3 and 2/3 points on slopes

        # re-normalize slope to CAX (%)
        idx_center = np.searchsorted(self.position, 0.0)
        if self.energy == 6:
            self.re_norm_percent = (self.calc_fff_renormalisation_6x() * 100 / self.meas_values[idx_center])
        else:
            self.re_norm_percent = (self.calc_fff_renormalisation_10x() * 100 / self.meas_values[idx_center])
        if center:
            offset = self.calc_cax_deviation()
        else:
            offset = 0.0

        # find point positions (index) on left side
        idx_a1 = np.searchsorted(self.position, offset - self.field_width['fwhm (nominal)'] / 3)
        idx_a2 = np.searchsorted(self.position, offset - self.field_width['fwhm (nominal)'] / 6)

        # find point positions (index) on right side
        idx_b1 = np.searchsorted(self.position, offset + self.field_width['fwhm (nominal)'] / 6)
        idx_b2 = np.searchsorted(self.position, offset + self.field_width['fwhm (nominal)'] / 3)

        # print("CAX-Index: ", idx_center)

        # get data points for slope calculation (no interpolation)
        a1 = (self.position[idx_a1], self.meas_values[idx_a1] * self.re_norm_percent)
        a2 = (self.position[idx_a2], self.meas_values[idx_a2] * self.re_norm_percent)

        b1 = (self.position[idx_b1], self.meas_values[idx_b1] * self.re_norm_percent)
        b2 = (self.position[idx_b2], self.meas_values[idx_b2] * self.re_norm_percent)

        # calculate slopes:
        slope_left = (a1[1] - a2[1]) / (a1[0] - a2[0])
        slope_right = (b1[1] - b2[1]) / (b1[0] - b2[0])

        # Intercepts, Peak Position
        i_left = a1[1] - (a1[0] * slope_left)
        i_right = b2[1] - (b2[0] * slope_right)

        peak_pos = (i_left - i_right) / (slope_right - slope_left)

        return {"Slope Left": slope_left, "Slope Right": slope_right, "Slope peak": peak_pos,
                "Left 30%": [a1[0], a1[1]],
                "Left 60%": [a2[0], a2[1]],
                "Right 30%": [b1[0], b1[1]],
                "Right 60%": [b2[0], b2[1]],
                }

    def calc_varian_flat(self) -> float:
        """
        Calculate the flatness according to Varian using pylinac.

        Returns
        -------
        din_flat : np.float64
            Flatness of the profile.

        """

        # extract np.array from Dataframe and create Pylinac profile
        meas_val = self.meas_values
        profil = SingleProfile(meas_val, None, Interpolation.NONE, False, 0.1, 10, self.normalisation)

        profil_max = profil.field_calculation(0.8, 'max')
        profil_min = profil.field_calculation(0.8, 'min')

        # flatness like PTW Data Analyse Varian Protocol
        varian_flat = (profil_max - profil_min) / (profil_max + profil_min) * 100

        return varian_flat

    def calc_sym(self) -> float:
        """
        Calculates the symmetry of the field plane with the point difference
        method using pylinac single profile class.

        Returns
        -------
        symmetry : np.float64
            Calculated symmetry value in %.

        """

        meas_val = self.meas_values
        if not self.detector_array:
            profile = SingleProfile(meas_val, None, Interpolation.NONE, False, 0.1, 10, self.normalisation)
        else:
            profile = SingleProfile(meas_val, None, Interpolation.NONE, False, 0.1, 10, self.normalisation,
                                    Edge.INFLECTION_HILL)
        symmetry = field_analysis.symmetry_point_difference(profile, 0.8)

        return np.abs(symmetry)

    def calc_fff_unflatness(self) -> np.float64:
        """
        Calculate the FFF un-flatness according to Fogliata

        Returns
        -------
        un-flatness : np.float64
            calculated un-flatness value.

        """

        idx_center = np.searchsorted(self.position, 0.0)

        # Unflattness
        idx_80_left = np.searchsorted(self.position, 0.0 - 0.8 * self.field_width['fwhm (nominal)'] / 2)
        idx_80_right = np.searchsorted(self.position, 0.8 * self.field_width['fwhm (nominal)'] / 2)

        unflattness = 1.0
        for x in [idx_80_left, idx_80_right]:
            tmp = (self.meas_values[idx_center] / self.meas_values[x])
            if tmp > unflattness:
                unflattness = tmp

        return unflattness

    def calc_fff_renormalisation_10x(self) -> float:
        """
        Return the field size correction factor for FFF beams
        Formula is taken from Folgiata et al.
        https://www.postersessiononline.eu/173580348_eu/congresos/ESTRO2016/aula/-PO_809_ESTRO2016.pdf

        Returns
        -------
        renormalisation_factor: np.float64
            Re-normalisation value to "stretch" the FFF profile percent values.

        """

        renormalisation_factor = (84.4 + (3.10 * (self.nominal_field_size / 10)) + (1.37 * (self.scan_depth / 10))) / \
                                 (1 - (0.0063 * (self.nominal_field_size / 10)) + (0.013 * (self.scan_depth / 10)))

        return renormalisation_factor / 100

    def calc_fff_renormalisation_6x(self) -> float:
        """
        Return the field size correction factor for FFF beams
        Formula is taken from Folgiata et al.
        https://www.postersessiononline.eu/173580348_eu/congresos/ESTRO2016/aula/-PO_809_ESTRO2016.pdf

        Returns
        -------
        renormalisation_factor: np.float64
            Re-normalisation value to "stretch" the FFF profile percent values.

        """

        renormalisation_factor = (91.3 + 1.2 * self.nominal_field_size / 10 + 0.138 * self.scan_depth / 10) / \
                                 (1 - 0.0075 * self.nominal_field_size / 10 + 0.0014 * self.scan_depth / 10)

        return renormalisation_factor / 100

    def normalise(self, varian_acceptance: bool = False):
        if self.filter == "FFF" and not varian_acceptance:
            return self.meas_values * self.re_norm_percent
        else:
            idx = np.searchsorted(self.position, 0.0)
            cax_val = self.meas_values[idx]
            return 100 * self.meas_values / cax_val

    def calc_varian_acceptance_data(self):
        meas_values = self.normalise(varian_acceptance=True)

        if self.nominal_field_size == 100.0:
            point_a = 40
            point_b = 20
        else:
            point_a = 180
            point_b = 60

        # find point positions (index) on left side
        idx_a1 = np.searchsorted(self.position, -point_a)
        idx_a2 = np.searchsorted(self.position, -point_b)

        # find point positions (index) on right side
        idx_b1 = np.searchsorted(self.position, point_b)
        idx_b2 = np.searchsorted(self.position, point_a)

        if self.nominal_field_size == 100.0:
            return {'Left 4cm': meas_values[idx_a1], 'Left 2cm': meas_values[idx_a2],
                    'Right 2cm': meas_values[idx_b1], 'Right 4cm': meas_values[idx_b2]}
        elif self.nominal_field_size == 400.0:
            return {'Left 18cm': meas_values[idx_a1], 'Left 6cm': meas_values[idx_a2],
                    'Right 6cm': meas_values[idx_b1], 'Right 18cm': meas_values[idx_b2]}
