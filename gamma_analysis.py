# -*- coding: utf-8 -*-
"""
gamma_analysis.py
Created on Fri 8th July 2022
@author: James Murphy
Modified by James Murphy 24th May 2024
"""

from typing import Union
from pymedphys import gamma
from pylinac.core.profile import SingleProfile, Interpolation, Normalization
import numpy as np
import matplotlib.pyplot as plt

NumberLike = Union[int, float]


def gamma_hist(instance: gamma, curve_type: str, axis: plt.Axes = None) -> None:
    """
        Produce gamma histogram of the evaluation scan and the reference scan.

        Parameters
        ----------
        instance : Gamma
            Instance of the Gamma class.
        curve_type : str
            Scan type.
        axis : Matplotlib.Axes
            Plot line to a given axis.

        Returns
        -------
        None.
    """

    if instance.local_gamma:
        gamma_norm_condition = 'local gamma'
    else:
        gamma_norm_condition = 'global gamma'

    if curve_type != 'PDD':
        curve_type = curve_type.replace('_', ' ').title()

    if instance.modality == 'EL':
        instance.modality = ' MeV'

    axis.set_title(curve_type + ' ' + str(instance.energy) + instance.modality +
                   f" Dose cut: {instance.lower_percent_dose_cutoff}% | {gamma_norm_condition} "
                   f"({instance.dose_percent_threshold}%/{instance.distance_mm_threshold}mm) | "
                   f"Pass Rate(\u03B3<=1): {instance.pass_ratio * 100:.2f}% \n ref pts: {len(instance.dose_reference)} "
                   f"| " f"valid \u03B3 pts: {len(instance.valid_gamma)}")

    if len(instance.dose_reference) == len(instance.dose_evaluation):
        p = axis.hist(instance.valid_gamma, instance.bins, density=True)  # y value is probability density in each bin
    else:
        p = axis.hist(instance.gamma_result, instance.bins, density=True)

    max_gamma = max(p[0])
    axis.set_xlim(0, instance.max_gamma)
    axis.set_xlabel('gamma index of reference point')  # FG
    axis.set_ylabel('probability density')
    axis.set_ylim(0, 1.1 * max_gamma)
    axis.vlines(x=[1], ymin=0, ymax=1.1 * max_gamma, colors='purple', ls='-', lw=1, label='target')


def gamma_curve(instance: gamma, curve_type: str, ax: plt.Axes = None) -> None:
    """
        Produce plots of the evaluation scan and the reference scan along with the gamma value at each position.

        Parameters
        ----------
        instance : Gamma
            Instance of the Gamma class.
        curve_type : str
            Scan type.
        ax : Matplotlib.Axes
            Plot line to a given axis.

        Returns
        -------
        None.

        """
    if instance.local_gamma:
        gamma_norm_condition = 'local gamma'
    else:
        gamma_norm_condition = 'global gamma'

    if curve_type != 'PDD':
        curve_type = curve_type.replace('_', ' ').title()

    if instance.modality == 'EL':
        instance.modality = ' MeV'

    _dose_reference = instance.dose_reference * 100
    _dose_evaluation = instance.dose_evaluation * 100

    if isinstance(instance.dose_evaluation, SingleProfile):
        _dose_evaluation = instance.dose_evaluation.values * 100
    if isinstance(instance.dose_reference, SingleProfile):
        _dose_reference = instance.dose_reference.values * 100

    max_ref_dose = np.max(_dose_reference)  # max reference dose
    max_eva_dose = np.max(_dose_evaluation)  # max evaluation dose
    #  lower_dose_cutoff = instance.lower_percent_dose_cutoff / 100 * max_ref_dose

    ax.set_title('{0} {1}{2} reference and evaluation dose curves for {3} index {4} cm field.'
                 .format(curve_type, instance.energy, instance.modality, gamma_norm_condition,
                         instance.filed_size), fontsize=12)

    ax.tick_params(direction='in')
    ax.tick_params(axis='x', bottom='on', top='on')
    ax.tick_params(labeltop='on')
    ax.minorticks_on()
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Response (%)')
    ax.set_ylim(0, max(max_ref_dose, max_eva_dose) * 1.1)

    ax2 = ax.twinx()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('gamma index')
    ax2.set_ylim([0, instance.max_gamma * 2.0])

    curve_0 = ax.plot(instance.axis_reference, _dose_reference, 'r-', label='reference dose')
    curve_1 = ax.plot(instance.axis_evaluation, _dose_evaluation, 'b-', mfc='none', markersize=5,
                      label='evaluation dose')

    curve_2 = ax2.plot(instance.axis_reference, instance.gamma_result,
                       label=f"gamma ({instance.dose_percent_threshold}%/" f"{instance.distance_mm_threshold}mm)")
    curves = curve_0 + curve_1 + curve_2

    labels = [label.get_label() for label in curves]
    ax.legend(curves, labels, loc='upper right', fontsize=10)
    ax.grid(True)


class Gamma:
    """
    Calculate the 1D Gamma for PDDs and Profiles.
    """

    dose_percent_threshold = 1
    distance_mm_threshold = 1
    lower_percent_dose_cutoff = 10
    interp_fraction = 20  # Should be 10 or more for more accurate results
    random_subset = None
    max_gamma = 2  # int
    local_gamma = True  # False indicates global gamma is calculated
    quiet = True
    num_bins = 20
    modality = None
    energy = None

    def __init__(self, reference_data: list, evaluation_data: list, **gamma_options) -> None:
        self.gamma_result = None
        self.pass_ratio = 0
        self.bins = None
        reference_type = reference_data[0]
        evaluation_type = evaluation_data[0]
        reference = reference_data[1]
        evaluation = evaluation_data[1]
        self.modality = reference['Modality']
        self.energy = reference['Energy']
        self.normalization = Normalization.BEAM_CENTER
        self.filed_size = reference['Nominal Field Size']

        if reference_type == evaluation_type:
            for k, v in gamma_options.items():
                setattr(self, k, v)

            if reference_type == 'PDD':
                self.axis_reference = reference['PDD_pos']
                self.dose_reference = SingleProfile(reference['PDD_val'], None, Interpolation.NONE, False,
                                                    0.1, 10, Normalization.MAX).values
                self.axis_evaluation = evaluation['PDD_pos']
                self.dose_evaluation = SingleProfile(evaluation['PDD_val'], None, Interpolation.NONE, False,
                                                     0.1, 10, Normalization.MAX).values
            else:
                self.axis_reference = reference['Profile_pos']
                self.dose_reference = SingleProfile(reference['Profile_val'], None, Interpolation.NONE, False,
                                                    0.1, 10, self.normalization).values
                self.axis_evaluation = evaluation['Profile_pos']
                self.dose_evaluation = SingleProfile(evaluation['Profile_val'], None, Interpolation.NONE, False,
                                                     0.1, 10, self.normalization).values

            self.gamma_result = gamma(self.axis_reference, self.dose_reference, self.axis_evaluation,
                                      self.dose_evaluation, self.dose_percent_threshold, self.distance_mm_threshold,
                                      self.lower_percent_dose_cutoff, self.interp_fraction, self.max_gamma,
                                      self.local_gamma, self.random_subset, self.quiet)
            self.valid_gamma = self.gamma_result[~np.isnan(self.gamma_result)]
            self.gamma_ratio()
            pass

    def gamma_ratio(self):
        self.bins = np.linspace(0, self.max_gamma, self.num_bins + 1)
        self.pass_ratio = np.sum(self.valid_gamma <= 1) / len(self.valid_gamma)
