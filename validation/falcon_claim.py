"""Compare DroneBench's flown energy per kilometre with the Titan Falcon V2 manufacturer figure.

The manual states 1.6 Wh/km at 3.5 kg, cruising between 45 and 65 km/h (fixtures/README.md). This is a manufacturer
claim, not an independent flight test, but it is the only published number for this aircraft, and DroneBench was not
tuned to it: `cd0_factor` stays at 1.0.

The plain Falcon (falcon_v2) weighs 3.15 kg in the model; the manual's 3.5 kg includes glue, paint, wiring and fittings
the model does not list. falcon_v2_padded (3.43 kg) is the closer match in mass, so both are run.

    .venv/bin/python validation/falcon_claim.py [fixture ...]
"""
import sys

from dronebench import flight as Fl
from dronebench import mission as M
from dronebench.ingest import load_design
from dronebench.pipeline import apply_plausibility, core

CLAIM = 1.6
calm = M.Condition(0.0, 0, 0.0, 1, "calm")

for name in sys.argv[1:] or ["falcon_v2", "falcon_v2_padded"]:
    d = load_design(f"fixtures/{name}")
    apply_plausibility(d)
    veh = Fl.Vehicle.from_bits(core(d, "endurance"), d)
    print(f"{name}: model mass {veh.mass:.2f} kg (manual: 3.5 kg), manufacturer claim: {CLAIM} Wh/km at 45-65 km/h\n")
    print("km/h   trim Wh/km   flown calm Wh/km   vs claim")
    flown = []
    for kmh in (45, 50, 55, 60, 65):
        V = kmh / 3.6
        r = M.power_at(veh, V, soc=0.8)
        f = M.fly_legs(veh, calm, V)
        flown.append(f.wh / f.ground_km)
        print(f"{kmh:4d}   {r['P'] / kmh:9.3f}   {flown[-1]:12.3f}   {100 * (flown[-1] / CLAIM - 1):+7.0f}%")
    mean = sum(flown) / len(flown)
    print(f"mean of five speeds {mean:.3f} Wh/km, {100 * (mean / CLAIM - 1):+.0f}% against the claim\n")
