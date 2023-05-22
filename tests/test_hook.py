import numpy as np

from pyroll.shida_flow_stress import flow_stress
from pyroll.shida_flow_stress import shida_flow_stress as hook

strain = 1
strain_rate = 1
temperature = 1200
composition = {"carbon": 0.1,
               "silicium": 0.25,
               "manganese": 0.45}


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
