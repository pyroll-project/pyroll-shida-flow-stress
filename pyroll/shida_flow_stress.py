import numpy as np

from pyroll.core import DeformationUnit
from pyroll.interface_material import ChemicalComposition
import logging

VERSION = "2.0"


@DeformationUnit.Profile.flow_stress
def shida_flow_stress(self: DeformationUnit.Profile):
    if hasattr(self, "chemical_composition"):
        return flow_stress(
            self.chemical_composition,
            self.strain,
            self.unit.strain_rate,
            self.temperature
        )


@DeformationUnit.Profile.flow_stress_function
def shida_flow_stress_function(self: DeformationUnit.Profile):
    if hasattr(self, "chemical_composition"):
        def f(strain: float, strain_rate: float, temperature: float) -> float:
            return flow_stress(self.chemical_composition, strain, strain_rate, temperature)

        return f


def flow_stress(chemical_composition: ChemicalComposition, strain: float, strain_rate: float, temperature: float):
    """
    Calculates the flow stress according to the constitutive equation from S. Shida for the provided
    material composition, strain, strain rate and temperature.

    :param chemical_composition: the chemical composition of the material
    :param strain: the equivalent strain experienced
    :param strain_rate: the equivalent strain rate experienced
    :param temperature: the absolute temperature of the material (K)
    """

    if chemical_composition.weight_percent_carbon >= 1.2:
        logging.getLogger(__name__).warning("Carbon content to great, Shida's flow-stress model is extrapolated!")
    elif strain > 0.7:
        logging.getLogger(__name__).warning("Strain to great, Shida's flow-stress model is extrapolated!")
    elif strain_rate < 0.1:
        logging.getLogger(__name__).warning("Strain rate to small, Shida's flow-stress model is extrapolated!")
    elif strain_rate > 100:
        logging.getLogger(__name__).warning("Strain rate to great, Shida's flow-stress model is extrapolated!")
    elif temperature <= 923.15:
        logging.getLogger(__name__).warning("Temperature to low, Shida's flow-stress model is extrapolated!")
    elif temperature >= 1423.15:
        logging.getLogger(__name__).warning("Temperature to high, Shida's flow-stress model is extrapolated!")

    transformation_temperature = 0.95 * (chemical_composition.weight_percent_carbon + 0.41) / (
            chemical_composition.weight_percent_carbon + 0.32)

    normalized_temperature = temperature / 1000

    if normalized_temperature < transformation_temperature:
        temperature_correction = 30 * (chemical_composition.weight_percent_carbon + 0.9) * (
                normalized_temperature - 0.95 * (chemical_composition.weight_percent_carbon + 0.49) / (
                chemical_composition.weight_percent_carbon + 0.42)) ** 2 + (
                                         chemical_composition.weight_percent_carbon + 0.06) / (
                                         chemical_composition.weight_percent_carbon + 0.09)
        strain_rate_sensitivity = (
                                          0.081 * chemical_composition.weight_percent_carbon - 0.154) * normalized_temperature + (
                                          -0.019 * chemical_composition.weight_percent_carbon + 0.207) + 0.027 / (
                                          chemical_composition.weight_percent_carbon + 0.32)

    else:
        temperature_correction = 1
        strain_rate_sensitivity = (
                                          -0.019 * chemical_composition.weight_percent_carbon - 0.126) * normalized_temperature + (
                                          0.075 * chemical_composition.weight_percent_carbon - 0.05)

    strain_hardening_factor = 0.41 - 0.07 * chemical_composition.weight_percent_carbon
    deformation_resistance_contribution = 0.28 * temperature_correction * np.exp(
        chemical_composition.weight_percent_carbon / 5 - 0.01 / (chemical_composition.weight_percent_carbon + 0.05))
    strain_hardening_contribution = 1.3 * (strain / 0.2) ** strain_hardening_factor - 0.3 * (strain / 0.2)
    strain_rate_hardening_contribution = (strain_rate / 10) ** strain_rate_sensitivity

    return deformation_resistance_contribution * strain_hardening_contribution * strain_rate_hardening_contribution
