import numpy as np

from pyroll.core import DeformationUnit

VERSION = "2.0.0"


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


def flow_stress(chemical_composition: dict[str, float], strain: float, strain_rate: float, temperature: float):
    """
    Calculates the flow stress according to the constitutive equation from S. Shida for the provided
    material composition, strain, strain rate and temperature.

    :param chemical_composition: the chemical composition of the material
    :param strain: the equivalent strain experienced
    :param strain_rate: the equivalent strain rate experienced
    :param temperature: the absolute temperature of the material (K)
    """

    strain = strain + 0.1
    strain_rate = strain_rate + 0.1

    transformation_temperature = 0.95 * (chemical_composition["carbon"] + 0.41) / (
            chemical_composition["carbon"] + 0.32)
    normalized_temperature = temperature / 1000

    if normalized_temperature < transformation_temperature:
        temperature_correction = 30 * (chemical_composition["carbon"] + 0.9) * (
                normalized_temperature - 0.95 * (chemical_composition["carbon"] + 0.49) / (
                chemical_composition["carbon"] + 0.42)) ** 2 + (
                                         chemical_composition["carbon"] + 0.06) / (
                                         chemical_composition["carbon"] + 0.09)
        strain_rate_sensitivity = (
                                          0.081 * chemical_composition["carbon"] - 0.154) * normalized_temperature + (
                                          -0.019 * chemical_composition["carbon"] + 0.207) + 0.027 / (
                                          chemical_composition["carbon"] + 0.32)

    else:
        temperature_correction = 1
        strain_rate_sensitivity = (
                                          -0.019 * chemical_composition["carbon"] - 0.126) * normalized_temperature + (
                                          0.075 * chemical_composition["carbon"] - 0.05)

    strain_hardening_factor = 0.41 - 0.07 * chemical_composition["carbon"]
    deformation_resistance_contribution = 0.28 * temperature_correction * np.exp(
        5 / normalized_temperature - 0.01 / (chemical_composition["carbon"] + 0.05))
    strain_contribution = 1.3 * (strain / 0.2) ** strain_hardening_factor - 0.3 * (strain / 0.2)
    strain_rate_contribution = (strain_rate / 10) ** strain_rate_sensitivity

    conversion_to_si_units_from_kgf_per_mm_squared = 9806650

    return deformation_resistance_contribution * strain_contribution * strain_rate_contribution * conversion_to_si_units_from_kgf_per_mm_squared
