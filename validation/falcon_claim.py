"""Compare DroneBench's flown energy per kilometre with the Titan Falcon V2 manufacturer figure.

The manual states 1.6 Wh/km at 3.5 kg, cruising between 45 and 65 km/h (fixtures/README.md). This is a manufacturer
claim, not an independent flight test, but it is the only published number for this aircraft, and DroneBench was not
tuned to it: `cd0_factor` stays at 1.0.

    .venv/bin/python validation/falcon_claim.py
"""
from dronebench import flight as Fl
from dronebench import mission as M
from dronebench.ingest import load_design
from dronebench.pipeline import apply_plausibility, core

d = load_design("fixtures/falcon_v2")
apply_plausibility(d)
veh = Fl.Vehicle.from_bits(core(d, "endurance"), d)
print(f"model mass {veh.mass:.2f} kg (manual: 3.5 kg), manufacturer claim: 1.6 Wh/km at 45-65 km/h\n")
print("km/h   trim Wh/km   flown calm Wh/km")
calm = M.Condition(0.0, 0, 0.0, 1, "calm")
for kmh in (45, 50, 55, 60, 65):
    V = kmh / 3.6
    r = M.power_at(veh, V, soc=0.8)
    f = M.fly_legs(veh, calm, V)
    print(f"{kmh:4d}   {r['P'] / kmh:9.3f}   {f.wh / f.ground_km:12.3f}")
