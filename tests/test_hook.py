import weakref

import numpy as np

from pyroll.interface_material import ChemicalComposition
from pyroll.shida_flow_stress import flow_stress
from pyroll.shida_flow_stress import shida_flow_stress as hook

strain = 1
strain_rate = 1
temperature = 1200
composition = ChemicalComposition(
    weight_percent_carbon=0.2,
    weight_percent_silicium=0.2,
    weight_percent_manganese=0.5,
    weight_percent_phosphor=0.02,
    weight_percent_sulfur=0.017,
    weight_percent_copper=0.2
)


class DummyProfile:
    def __init__(self):
        self.strain = strain
        self.temperature = temperature
        self.material = "C45"
        self.chemical_composition = composition


class DummyRollPass:
    def __init__(self):
        self.strain_rate = strain_rate


def test_hook():
    rp = DummyRollPass()
    p = DummyProfile()
    p.unit = rp
    print()

    fs = hook(p)
    print(fs)
    assert np.isfinite(fs)
    assert fs == flow_stress(composition, strain, strain_rate, temperature)

    rp.strain_rate = 0
    fs = hook(p)
    print(fs)
    assert np.isfinite(fs)
    assert fs == flow_stress(composition, strain, 0, temperature)

    p.strain = 0
    fs = hook(p)
    print(fs)
    assert np.isfinite(fs)
    assert fs == flow_stress(composition, 0, 0, temperature)
