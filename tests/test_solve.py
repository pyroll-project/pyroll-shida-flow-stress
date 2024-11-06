import logging
import webbrowser
from pathlib import Path
from pyroll.core import Profile, Roll, RollPass, Transport, RoundGroove, CircularOvalGroove, PassSequence


def test_solve(tmp_path: Path, caplog):
    caplog.set_level(logging.DEBUG, logger="pyroll")

    import pyroll.shida_flow_stress

    in_profile = Profile.round(
        diameter=30e-3,
        temperature=1200 + 273.15,
        strain=0,
        material=["Test Steel", "steel"],
        chemical_composition={"carbon": 0.1,
                              "silicium": 0.25,
                              "manganese": 0.45}

    )

    sequence = PassSequence(
        [
            RollPass(
                label="Oval I",
                roll=Roll(
                    groove=CircularOvalGroove(
                        depth=8e-3,
                        r1=6e-3,
                        r2=40e-3
                    ),
                    nominal_radius=160e-3,
                    rotational_frequency=1
                ),
                gap=2e-3,
            ),
            Transport(
                label="I => II",
                duration=1
            ),
            RollPass(
                label="Round II",
                roll=Roll(
                    groove=RoundGroove(
                        r1=1e-3,
                        r2=12.5e-3,
                        depth=11.5e-3
                    ),
                    nominal_radius=160e-3,
                    rotational_frequency=1
                ),
                gap=2e-3,
            ),
        ]
    )

    try:
        sequence.solve(in_profile)
    finally:
        print("\nLog:")
        print(caplog.text)

    try:
        from pyroll.report import report
        report = report(sequence)
        f = tmp_path / "report.html"
        f.write_text(report, encoding="utf-8")
        webbrowser.open(f.as_uri())
    except ImportError:
        pass
