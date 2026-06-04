# -*- coding: utf-8 -*-
"""
gamma_analysis.py
Created on Fri 8th July 2022
@author: James Murphy
Modified by James Murphy 24th May 2024
"""

from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymedphys import gamma as pymedphys_gamma
from pylinac import field_analysis
from pylinac.core.profile import (
    Interpolation,
    Normalization,
    SingleProfile,
)

NumberLike = Union[int, float]


# -----------------------------------------------------------------------------
# Plotting Utilities
# -----------------------------------------------------------------------------

def gamma_hist(instance: "Gamma", curve_type: str,
               axis: Optional[plt.Axes] = None) -> None:
    """
    Produce gamma histogram of the evaluation scan and reference scan.
    """

    if axis is None:
        axis = plt.gca()

    gamma_norm = "local gamma" if instance.local_gamma else "global gamma"

    display_curve_type = (
        curve_type if curve_type == "PDD"
        else curve_type.replace("_", " ").title()
    )

    modality = " MeV" if instance.modality == "EL" else instance.modality

    axis.set_title(
        f"{display_curve_type} {instance.energy}{modality} | "
        f"Dose cut: {instance.lower_percent_dose_cutoff}% | "
        f"{gamma_norm} "
        f"({instance.dose_percent_threshold}%/"
        f"{instance.distance_mm_threshold}mm) | "
        f"Pass Rate(γ<=1): {instance.pass_ratio * 100:.2f}%\n"
        f"ref pts: {len(instance.dose_reference)} | "
        f"valid γ pts: {len(instance.valid_gamma)}"
    )

    gamma_values = (
        instance.valid_gamma
        if len(instance.dose_reference) == len(instance.dose_evaluation)
        else instance.gamma_result
    )

    hist, _, _ = axis.hist(
        gamma_values,
        bins=instance.bins,
        density=True
    )

    max_gamma_density = np.max(hist)

    axis.set(
        xlim=(0, instance.max_gamma),
        ylim=(0, 1.1 * max_gamma_density),
        xlabel="gamma index of reference point",
        ylabel="probability density"
    )

    axis.axvline(
        x=1,
        color="purple",
        linestyle="-",
        linewidth=1,
        label="target"
    )


def gamma_curve(instance: "Gamma", curve_type: str,
                ax: Optional[plt.Axes] = None) -> None:
    """
    Plot reference/evaluation curves and gamma values.
    """

    if ax is None:
        ax = plt.gca()

    gamma_norm = "local gamma" if instance.local_gamma else "global gamma"

    display_curve_type = (
        curve_type if curve_type == "PDD"
        else curve_type.replace("_", " ").title()
    )

    modality = " MeV" if instance.modality == "EL" else instance.modality

    dose_reference = (
        instance.dose_reference.values * 100
        if isinstance(instance.dose_reference, SingleProfile)
        else instance.dose_reference * 100
    )

    dose_evaluation = (
        instance.dose_evaluation.values * 100
        if isinstance(instance.dose_evaluation, SingleProfile)
        else instance.dose_evaluation * 100
    )

    ax.set_title(
        f"{display_curve_type} {instance.energy}{modality} "
        f"reference and evaluation dose curves for "
        f"{gamma_norm} index {instance.field_size} cm field.",
        fontsize=12
    )

    ax.set(
        xlabel="Distance (mm)",
        ylabel="Response (%)",
        ylim=(0, max(np.max(dose_reference),
                      np.max(dose_evaluation)) * 1.1)
    )

    ax.tick_params(direction="in", axis="x",
                   bottom=True, top=True, labeltop=True)

    ax.minorticks_on()
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel("gamma index")
    ax2.set_ylim(0, instance.max_gamma * 2.0)

    curves = []

    curves += ax.plot(
        instance.axis_reference,
        dose_reference,
        "r-",
        label="reference dose"
    )

    curves += ax.plot(
        instance.axis_evaluation,
        dose_evaluation,
        "b-",
        mfc="none",
        markersize=5,
        label="evaluation dose"
    )

    curves += ax2.plot(
        instance.axis_reference,
        instance.gamma_result,
        label=(
            f"gamma "
            f"({instance.dose_percent_threshold}%/"
            f"{instance.distance_mm_threshold}mm)"
        )
    )

    ax.legend(
        curves,
        [curve.get_label() for curve in curves],
        loc="upper right",
        fontsize=10
    )


# -----------------------------------------------------------------------------
# Gamma Analysis
# -----------------------------------------------------------------------------

class Gamma:
    """
    Calculate 1D Gamma for PDDs and Profiles.
    """

    # Defaults
    dose_percent_threshold = 1
    distance_mm_threshold = 1
    lower_percent_dose_cutoff = 10
    interp_fraction = 20
    random_subset = None
    max_gamma = 2
    local_gamma = True
    quiet = True
    num_bins = 20

    def __init__(
        self,
        reference_data: list,
        evaluation_data: list,
        **gamma_options
    ) -> None:

        self.gamma_result = None
        self.gamma_results = None
        self.pass_ratio = 0.0
        self.bins = None

        reference_type, reference = reference_data
        evaluation_type, evaluation = evaluation_data

        if reference_type != evaluation_type:
            raise ValueError(
                "Reference and evaluation curve types do not match."
            )

        # Apply custom gamma settings
        for key, value in gamma_options.items():
            setattr(self, key, value)

        self.modality = reference["Modality"]
        self.energy = reference["Energy"]
        self.field_size = reference["Nominal Field Size"]

        self.normalization = Normalization.BEAM_CENTER

        # Load profiles
        if reference_type == "PDD":
            self.axis_reference = reference["PDD_pos"]
            self.axis_evaluation = evaluation["PDD_pos"]

            self.dose_reference = self._normalise_profile(
                reference["PDD_val"],
                Normalization.MAX
            )

            self.dose_evaluation = self._normalise_profile(
                evaluation["PDD_val"],
                Normalization.MAX
            )

        else:
            self.axis_reference = reference["Profile_pos"]
            self.axis_evaluation = evaluation["Profile_pos"]

            self.dose_reference = self._normalise_profile(
                reference["Profile_val"],
                self.normalization
            )

            self.dose_evaluation = self._normalise_profile(
                evaluation["Profile_val"],
                self.normalization
            )

        self.calculate_gamma()

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalise_profile(values, normalization):
        return SingleProfile(
            values,
            dpmm=None,
            interpolation=Interpolation.NONE,
            ground=False,
            interpolation_resolution_mm=0.1,
            edge_smoothing_ratio=10,
            normalization_method=normalization,
        ).values

    # -------------------------------------------------------------------------
    # Core Gamma Calculation
    # -------------------------------------------------------------------------

    def calculate_gamma(self) -> None:

        self.gamma_result = pymedphys_gamma(
            self.axis_reference,
            self.dose_reference,
            self.axis_evaluation,
            self.dose_evaluation,
            self.dose_percent_threshold,
            self.distance_mm_threshold,
            self.lower_percent_dose_cutoff,
            self.interp_fraction,
            self.max_gamma,
            self.local_gamma,
            self.random_subset,
            self.quiet,
        )

        self.valid_gamma = self.gamma_result[
            ~np.isnan(self.gamma_result)
        ]

        self.bins = np.linspace(
            0,
            self.max_gamma,
            self.num_bins + 1
        )

        self.pass_ratio = self.gamma_ratio(self.valid_gamma)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def gamma_ratio(self, valid_gamma: np.ndarray) -> float:
        """
        Calculate gamma pass ratio.
        """

        if len(valid_gamma) == 0:
            return 0.0

        return np.mean(valid_gamma <= 1)

    def gamma_values(self) -> None:
        """
        Calculate gamma pass ratios for combinations of
        DTA and dose thresholds.
        """

        dta_values = [0.5, 1, 1.5, 2, 2.5, 3]
        dose_values = [0.5, 1, 1.5, 2, 2.5, 3]

        results = []

        for dta in dta_values:

            row = []

            for dose in dose_values:

                gamma_result = pymedphys_gamma(
                    self.axis_reference,
                    self.dose_reference,
                    self.axis_evaluation,
                    self.dose_evaluation,
                    dose,
                    dta,
                    self.lower_percent_dose_cutoff,
                    self.interp_fraction,
                    self.max_gamma,
                    self.local_gamma,
                    self.random_subset,
                    self.quiet,
                )

                valid_gamma = gamma_result[
                    ~np.isnan(gamma_result)
                ]

                row.append(self.gamma_ratio(valid_gamma))

            results.append(row)

        self.gamma_results = pd.DataFrame(
            results,
            index=dta_values,
            columns=dose_values
        )