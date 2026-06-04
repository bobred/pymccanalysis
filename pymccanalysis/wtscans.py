import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from pylinac import field_analysis
from pylinac.core.profile import (
    SingleProfile,
    Interpolation,
    Normalization,
    Edge
)


# =============================================================================
# UTILITIES
# =============================================================================

def interp_x(x1, y1, x2, y2, target_y):
    """
    Fast linear interpolation for x at target_y.
    """
    return x1 + ((target_y - y1) * (x2 - x1)) / (y2 - y1)


def plot_tangent(a, b, forward, back, modality=None, axis=None):
    """
    Plot tangent between two points.
    """

    slope = (a[1] - b[1]) / (a[0] - b[0])

    x = np.linspace(a[0] - back, b[0] + forward, 100)

    y = slope * (x - a[0]) + a[1]

    axis.plot(x, y, 'C4--', linewidth=1)

    if modality != 'EL':
        axis.plot(a[0], a[1], 'bx')
        axis.plot(b[0], b[1], 'bx')


# =============================================================================
# PLOTTING
# =============================================================================

class WTScanPlotting:

    @staticmethod
    def configure_axis(axis, title):
        axis.grid(True, which="both", ls="-")
        axis.set_xlabel("Distance (mm)")
        axis.set_ylabel("Response (%)")
        axis.set_title(title)

    @staticmethod
    def pdd_plot(data, axis=None):

        pdd = data['PDD']

        axis.plot(pdd['PDD_pos'], pdd['PDD_val'])

        if pdd['Modality'] == 'EL':
            for key in ('R100', 'R90', 'R80', 'R50', 'R30'):
                axis.scatter(
                    pdd[key],
                    float(key[1:]),
                    marker='x',
                    color='red'
                )

            plot_tangent(
                [pdd['R60'], 60],
                [pdd['R40'], 40],
                20,
                100,
                'EL',
                axis
            )

            plot_tangent(
                [
                    pdd['PDD_pos'][-100],
                    pdd['PDD_val'][-100]
                ],
                [
                    pdd['PDD_pos'][-10],
                    pdd['PDD_val'][-10]
                ],
                10,
                200,
                'EL',
                axis
            )

            axis.vlines(
                x=pdd['Rp'][0],
                ymin=0,
                ymax=20,
                colors='C4',
                linestyles='dashed',
                linewidth=1
            )

        else:

            axis.scatter(
                pdd['R100'],
                100,
                marker='x',
                color='red'
            )

            axis.scatter(
                100,
                pdd['D100'],
                marker='x',
                color='red'
            )

            axis.scatter(
                200,
                pdd['D200'],
                marker='x',
                color='red'
            )

        axis.set_ylim(
            0,
            np.max(pdd['PDD_val']) * 1.1
        )

        axis.set_xlim(xmin=0)

        WTScanPlotting.configure_axis(
            axis,
            f"PDD {pdd['Energy']}{pdd['Modality']}"
        )

    def profile_plot(self, data, plane, axis=None):
        axis.plot(data['Profile_pos'], data['Profile_val'])
        axis.scatter(data['CaxDev'], data['norm_value'], marker='x', color='red')
        axis.set_ylim(0, np.max(data['Profile_val']) * 1.10)

        self.configure_axis(axis, f"{plane.replace('_', ' ').title()} "
            f"{data['Energy']}{data['Modality']}")

        self.plot_penumbra(data['Penumbra'], axis)

        self.plot_symmetry(data['FWHM'], data['norm_value'], data['Filter'], axis)

        self.plot_fwhm(data['FWHM'], data['norm_value'], axis)

        if data['Filter'] == 'FF':
            self.plot_flatness(data['Profile_val'], axis)
        else:
            if data['Nominal Field Size'] == 100:
                left_forward, left_back = 20, 20
                right_forward, right_back = 20, 20
            else:
                left_forward, left_back = 75, 40
                right_forward, right_back = 40, 75

            plot_tangent(data['slopes']['Left 30%'], data['slopes']['Left 60%'], left_forward, left_back, axis=axis)
            plot_tangent(data['slopes']['Right 30%'], data['slopes']['Right 60%'], right_forward, right_back, axis=axis)

    @staticmethod
    def plot_flatness(profile, axis):

        sp = SingleProfile(
            profile,
            None,
            Interpolation.NONE,
            False,
            0.1,
            10,
            Normalization.BEAM_CENTER
        )

        data = sp.field_data(in_field_ratio=0.8)

        axis.axhline(
            np.max(data['field values']) * 100,
            color='g',
            linestyle='dashed',
            linewidth=1
        )

        axis.axhline(
            np.min(data['field values']) * 100,
            color='g',
            linestyle='dashed',
            linewidth=1
        )

    @staticmethod
    def plot_symmetry(fwhm, peak, filter_type, axis):
        if filter_type == 'FFF':
            axis.vlines(x=fwhm['left_80%'][0], ymin=fwhm['left_80%'][1], ymax=peak, colors='C4', linestyles='dashed',
                linewidth=1)
            axis.vlines(x=fwhm['right_80%'][0], ymin=fwhm['right_80%'][1], ymax=peak, colors='C4', linestyles='dashed',
                linewidth=1)
        else:
            axis.scatter(*fwhm['left_80%'], marker='x', color='red')
            axis.scatter(*fwhm['right_80%'], marker='x', color='red')

    @staticmethod
    def plot_fwhm(fwhm, peak, axis):
        axis.vlines(x=fwhm['left_index'],
            ymin=0,
            ymax=peak * 0.75,
            colors='C4',
            linestyles='dashed',
            linewidth=1
        )

        axis.vlines(
            x=fwhm['right_index'],
            ymin=0,
            ymax=peak * 0.75,
            colors='C4',
            linestyles='dashed',
            linewidth=1
        )

        axis.hlines(
            y=50,
            xmin=fwhm['left_index'],
            xmax=fwhm['right_index'],
            colors='C4',
            linestyles='dashed',
            linewidth=1
        )

    @staticmethod
    def plot_penumbra(
        penumbra,
        axis=None
    ):

        p = penumbra[2]

        axis.axvspan(
            p['left 20% index (exact)'],
            p['left 80% index (exact)'],
            alpha=0.5,
            color='pink'
        )

        axis.axvspan(
            p['right 20% index (exact)'],
            p['right 80% index (exact)'],
            alpha=0.5,
            color='pink'
        )


# =============================================================================
# PDD
# =============================================================================

class PDD:

    def __init__(
            self,
            data,
            normalise=False,
            ion_to_dose=False
    ):

        meta = data[1]
        dataset = data[2]

        self.results = {}

        self.curve_type = data[0]
        self.modality = meta['MODALITY']
        self.energy = meta['ENERGY']
        self.filter = meta.get('FILTER')

        self.nominal_field_size = meta['FIELD_CROSSPLANE']

        self.ion_to_dose = ion_to_dose

        self.meas_values = np.asarray(
            dataset['Values'],
            dtype=np.float32
        ).copy()

        self.position = np.asarray(
            dataset['Position'],
            dtype=np.float32
        )

        self.rev_values = self.meas_values[::-1]
        self.rev_position = self.position[::-1]

        if normalise:
            self.normalise()

        self.max_dose = np.nanmax(self.meas_values)
        self.idx_max = np.argmax(self.meas_values)

        self.results.update({
            "Type": self.curve_type,
            "Modality": self.modality,
            "Energy": self.energy,
            "Nominal Field Size": self.nominal_field_size,
            "PDD_pos": self.position,
            "PDD_val": self.meas_values
        })

        if self.filter:
            self.results["Filter"] = self.filter

        self.calculate_results()

    # -------------------------------------------------------------------------

    def calculate_results(self):

        if self.modality == 'X':
            d100 = self.dose_x(100.0)
            d200 = self.dose_x(200.0)

            self.results.update({
                "Type": self.curve_type,
                "Modality": self.modality,
                "Energy": self.energy,
                "Filter": self.filter,
                "Nominal Field Size": self.nominal_field_size,
                "D100": d100,
                "D200": d200,
                "R100": self.depth_max(),
                "R80": self.depth_x(80),
                "R50": self.depth_x(50),
                "Q Index": (1.2661 * d200 / d100) - 0.0595,
                "Surface Dose": self.calc_surface_dose(),
                "PDD_pos": self.position,
                "PDD_val": self.meas_values,
            })

        elif self.modality == 'EL':

            rp = self.calc_rp()

            self.results.update({
                "R100": self.depth_max(),
                "R90": self.depth_x(90),
                "R80": self.depth_x(80),
                "R60": self.depth_x(60),
                "R50": self.depth_x(50),
                "R40": self.depth_x(40),
                "R30": self.depth_x(30),
                "Rp": [rp, self.dose_x(rp) * 100],
                "Ds": self.dose_x(0.5) * 100
            })

    # -------------------------------------------------------------------------

    def normalise(self) -> None:
        self.meas_values = np.array(
            self.meas_values,
            dtype=np.float32,
            copy=True
        )

        max_val = np.nanmax(self.meas_values)

        if max_val:
            self.meas_values *= 100.0 / max_val

    # -------------------------------------------------------------------------

    def depth_max(self):

        return self.position[self.idx_max]

    # -------------------------------------------------------------------------

    def depth_x(self, x: float) -> float:

        target = self.max_dose * x * 0.01

        rev_values = self.meas_values[::-1]
        rev_position = self.position[::-1]

        idx = np.searchsorted(
            rev_values,
            target
        )

        return interp_x(
            rev_position[idx - 1],
            rev_values[idx - 1],
            rev_position[idx],
            rev_values[idx],
            target
        )

    # -------------------------------------------------------------------------

    def dose_x(self, depth):

        return np.interp(
            depth,
            self.position,
            self.meas_values
        )

    # -------------------------------------------------------------------------

    def calc_surface_dose(self):

        return (
            self.meas_values[5] / self.max_dose
        ) * 100

    # -------------------------------------------------------------------------

    def calc_rp(self):

        a1x = self.depth_x(60)
        a2x = self.depth_x(40)
        a50x = self.depth_x(50)

        slope_50 = (
            (0.6 - 0.4) *
            self.max_dose
        ) / (
            a1x - a2x
        )

        inter_50 = (
            0.5 * self.max_dose
        ) - (
            a50x * slope_50
        )

        e0_mean = self.calc_e0_mean()

        rp_est = (
            0.11 +
            0.505 * e0_mean -
            3e-4 * e0_mean ** 2
        ) * 100

        lin_start = int(rp_est + 100)

        if lin_start < (self.position.size - 2):

            slope_bs, inter_bs = np.polyfit(
                self.position[lin_start:],
                self.meas_values[lin_start:],
                1
            )

        else:

            inter_bs = self.meas_values[-1]
            slope_bs = 0.0

        return (
            inter_bs - inter_50
        ) / (
            slope_50 - slope_bs
        )

    # -------------------------------------------------------------------------

    def calc_e0_mean(self):

        return (
            2.33 *
            self.calc_r50d_ipem() /
            10
        )

    # -------------------------------------------------------------------------

    def calc_r50d_ipem(self):

        d50ion = self.rev_values.max() / 2

        rev_vals = np.flip(self.rev_values)
        rev_pos = np.flip(self.rev_position)

        idx = np.searchsorted(rev_vals, d50ion)

        r50ion = np.interp(d50ion, self.rev_values[::-1], self.rev_position[::-1])

        return (1.029 * r50ion) - 0.063

# =============================================================================
# PROFILE
# =============================================================================

class XyProfile:

    def __init__(
        self,
        data,
        normalisation=Normalization.BEAM_CENTER
    ):

        meta = data[1]
        dataset = data[2]

        self.results = {}

        self.meas_values = np.asarray(dataset['Values'])

        self.position = np.asarray(dataset['Position'])

        self.rev_values = self.meas_values[::-1]
        self.rev_position = self.position[::-1]

        self.curve_type = data[0]
        self.modality = meta['MODALITY']
        self.energy = meta['ENERGY']

        self.filter = meta['FILTER']

        self.isocenter = meta['ISOCENTER']
        self.ssd = meta['SSD']
        self.scan_depth = meta['SCAN_DEPTH']

        self.normalisation = normalisation

        self.nominal_field_size = (meta['FIELD_INPLANE'] if self.curve_type == 'INPLANE_PROFILE'
            else meta['FIELD_CROSSPLANE'])

        self.offset = 0.0

        if self.filter == "FFF" and self.energy == 6:
            self.norm_value = self.calc_fff_renormalisation_6x() * 100
        elif self.filter == "FFF" and self.energy == 10:
            self.norm_value = self.calc_fff_renormalisation_10x() * 100
        else:
            self.norm_value = 100

        self.max_value = np.max(self.meas_values)

        edge_method = (Edge.INFLECTION_DERIVATIVE
            if self.filter == 'FFF'
            else Edge.FWHM
        )

        self.profile = SingleProfile(self.meas_values, None, Interpolation.NONE, False, 0.1, 10, self.normalisation, edge_detection_method=edge_method)

        self.norm_factor = self.get_norm_factor()

        self.field_width = self.calc_fwhm()

        self.build_results()

    # -------------------------------------------------------------------------

    def get_norm_factor(self):

        if self.filter != 'FFF':
            return 1.0

        if self.energy == 6:
            return self.calc_fff_renormalisation_6x()

        return self.calc_fff_renormalisation_10x()

    # -------------------------------------------------------------------------

    def normalise(self):

        idx = np.searchsorted(
            self.position,
            0.0
        )

        cax = self.meas_values[idx]

        if self.filter == 'FFF':
            return (
                self.meas_values *
                self.norm_factor *
                100
            )

        return (
            self.meas_values /
            cax
        ) * 100

    # -------------------------------------------------------------------------

    def calc_half_max(
        self,
        max_type='cax'
    ):

        if max_type == 'cax':

            idx = np.searchsorted(
                self.position,
                self.offset
            )

            max_val = self.meas_values[idx]

        else:

            max_val = self.max_value

        half_max = 0.5 * max_val

        if self.filter == 'FFF':
            half_max /= self.norm_factor

        return half_max

    # -------------------------------------------------------------------------

    def calc_fwhm(self, max_type: str = 'cax') -> dict:

        half_max = self.calc_half_max(max_type)

        meas = self.meas_values
        pos = self.position

        rev_meas = meas[::-1]
        rev_pos = pos[::-1]

        left_idx = np.searchsorted(meas, half_max)
        right_idx = np.searchsorted(rev_meas, half_max)

        ax1 = pos[left_idx - 1]
        ay1 = meas[left_idx - 1]
        ax2 = pos[left_idx]
        ay2 = meas[left_idx]

        bx1 = rev_pos[right_idx - 1]
        by1 = rev_meas[right_idx - 1]
        bx2 = rev_pos[right_idx]
        by2 = rev_meas[right_idx]

        left_pos = interp_x(ax1, ay1, ax2, ay2, half_max)
        right_pos = interp_x(bx1, by1, bx2, by2, half_max)

        fwhm_nominal = right_pos - left_pos

        iso_corr = self.isocenter / (self.ssd + self.scan_depth)

        idx_80_left = np.searchsorted(
            pos,
            -0.4 * fwhm_nominal
        )

        idx_80_right = np.searchsorted(
            pos,
            0.4 * fwhm_nominal
        )

        profile = SingleProfile(
            meas,
            None,
            Interpolation.NONE,
            False,
            0.1,
            10,
            self.normalisation
        )

        profile_values = profile.values * self.norm_value

        return {
            'fwhm (nominal)': fwhm_nominal,
            'fwhm': fwhm_nominal * iso_corr,
            'left_index': left_pos,
            'right_index': right_pos,
            'left_80%': (
                pos[idx_80_left],
                profile_values[idx_80_left]
            ),
            'right_80%': (
                pos[idx_80_right],
                profile_values[idx_80_right]
            )
        }

    # -------------------------------------------------------------------------

    def calc_cax_deviation(self):

        return (
            self.field_width['left_index'] +
            self.field_width['right_index']
        ) * 0.5

    # -------------------------------------------------------------------------

    def calc_varian_flat(self):

        pmax = self.profile.field_calculation(
            0.8,
            'max'
        )

        pmin = self.profile.field_calculation(
            0.8,
            'min'
        )

        return (
            (pmax - pmin) /
            (pmax + pmin)
        ) * 100

    # -------------------------------------------------------------------------

    def calc_sym(self):

        symmetry = field_analysis.symmetry_point_difference(
            self.profile,
            0.8
        )

        return np.abs(symmetry)

    # -------------------------------------------------------------------------

    def calc_penumbra(self):

        values = self.profile.values * (
            self.norm_factor * 100
        )

        mid = values.size // 2

        left = values[:mid]
        right = values[mid:]

        left_pos = self.position[:mid]
        right_pos = self.position[mid:]

        s = np.interp(20, left, left_pos)
        t = np.interp(80, left, left_pos)

        u = np.interp(
            80,
            right[::-1],
            right_pos[::-1]
        )

        v = np.interp(
            20,
            right[::-1],
            right_pos[::-1]
        )

        return (
            t - s,
            v - u,
            {
                "left 20% index (exact)": s,
                "left 80% index (exact)": t,
                "right 80% index (exact)": u,
                "right 20% index (exact)": v,
            }
        )

    # -------------------------------------------------------------------------

    def calc_fff_slopes_peak(self, center: bool = True) -> dict:
        """
        Calculate the left and right slopes of the FFF profiles.

        Parameters
        ----------
        center : bool, optional
            If True, positions are shifted by the CAX deviation.

        Returns
        -------
        dict
            Left/right slopes, peak position, and slope points.
        """

        idx_center = np.searchsorted(
            self.position,
            0.0
        )

        renorm = (
            self.calc_fff_renormalisation_6x()
            if self.energy == 6
            else self.calc_fff_renormalisation_10x()
        )

        self.re_norm_percent = (renorm * 100 / self.meas_values[idx_center])

        width = self.field_width['fwhm (nominal)']

        offset = (
            self.calc_cax_deviation()
            if center else 0.0
        )

        pos = self.position
        vals = self.meas_values
        scale = self.re_norm_percent

        idx_a1 = np.searchsorted(
            pos,
            offset - width / 3
        )

        idx_a2 = np.searchsorted(
            pos,
            offset - width / 6
        )

        idx_b1 = np.searchsorted(
            pos,
            offset + width / 6
        )

        idx_b2 = np.searchsorted(
            pos,
            offset + width / 3
        )

        a1 = (pos[idx_a1], vals[idx_a1] * scale)

        a2 = (
            pos[idx_a2],
            vals[idx_a2] * scale
        )

        b1 = (
            pos[idx_b1],
            vals[idx_b1] * scale
        )

        b2 = (
            pos[idx_b2],
            vals[idx_b2] * scale
        )

        slope_left = (
                (a1[1] - a2[1]) /
                (a1[0] - a2[0])
        )

        slope_right = (
                (b1[1] - b2[1]) /
                (b1[0] - b2[0])
        )

        i_left = a1[1] - (a1[0] * slope_left)

        i_right = b2[1] - (b2[0] * slope_right)

        peak_pos = (
                (i_left - i_right) /
                (slope_right - slope_left)
        )

        return {
            "Slope Left": slope_left,
            "Slope Right": slope_right,
            "Slope peak": peak_pos,
            "Left 30%": list(a1),
            "Left 60%": list(a2),
            "Right 30%": list(b1),
            "Right 60%": list(b2),
        }

    # -------------------------------------------------------------------------

    def calc_fff_unflatness(self):

        idx_center = np.searchsorted(
            self.position,
            0.0
        )

        width = self.field_width['fwhm (nominal)']

        idx_left = np.searchsorted(
            self.position,
            -0.4 * width
        )

        idx_right = np.searchsorted(
            self.position,
            0.4 * width
        )

        vals = self.meas_values[
            [idx_left, idx_right]
        ]

        return np.max(
            self.meas_values[idx_center] / vals
        )

    # -------------------------------------------------------------------------

    def normalise(self):
        idx = np.searchsorted(self.position, 0.0)

        cax = self.meas_values[idx]

        if self.filter == 'FFF':
            return (self.meas_values * self.norm_factor * 100)

        return (self.meas_values / cax) * 100

    def normalise(self, varian_acceptance: bool = False):
        if self.filter == "FFF" and not varian_acceptance:
            return self.meas_values * self.re_norm_percent
        else:
            idx = np.searchsorted(self.position, 0.0)
            cax_val = self.meas_values[idx]
            return 100 * self.meas_values / cax_val

    # -------------------------------------------------------------------------

    def calc_fff_renormalisation_10x(self):

        fs = self.nominal_field_size / 10
        depth = self.scan_depth / 10

        val = (
            84.4 +
            (3.10 * fs) +
            (1.37 * depth)
        ) / (
            1 -
            (0.0063 * fs) +
            (0.013 * depth)
        )

        return val / 100

    # -------------------------------------------------------------------------

    def calc_fff_renormalisation_6x(self):
        fs = self.nominal_field_size / 10
        depth = self.scan_depth / 10

        val = (
            91.3 +
            1.2 * fs +
            0.138 * depth
        ) / (
            1 -
            0.0075 * fs +
            0.0014 * depth
        )

        return val / 100

    # -------------------------------------------------------------------------

    def build_results(self):

        self.results.update({
            "Type": self.curve_type,
            "Modality": self.modality,
            "Energy": self.energy,
            "Filter": self.filter,
            "CaxDev": self.calc_cax_deviation(),
            "Scan depth": self.scan_depth,
            "FWHM": self.field_width,
            "Symmetry": self.calc_sym(),
            "Penumbra": self.calc_penumbra(),
            "Profile_pos": self.position,
            "Nominal Field Size": self.nominal_field_size
        })

        if self.filter == 'FF':
            self.results.update({
                "Flatness": self.calc_varian_flat(),
                "norm_value": 100,
                "Profile_val": self.normalise(),
            })

        elif self.filter == 'FFF':
            slopes = self.calc_fff_slopes_peak()
            self.results.update({
                "Unflatness": self.calc_fff_unflatness(),
                "norm_value": self.norm_factor * 100,
                "slopes": slopes,
                "Peak": slopes["Slope peak"],
                "Profile_val": self.normalise()
            })
            pass