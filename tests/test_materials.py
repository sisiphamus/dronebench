import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

import meshkit as mk  # noqa: E402
from dronebench.components import lookup  # noqa: E402
from dronebench.materials import MATERIALS, get_material, part_mass_g, resolve_material  # noqa: E402
from dronebench.model import PART_TYPES, Material, Part  # noqa: E402


@pytest.mark.parametrize("ptype", PART_TYPES)
def test_material_never_empty(ptype):
    m = resolve_material(Part(id="x", type=ptype))
    assert m.name and m.kind and m.density_kg_m3 > 0
    assert m.source == "assumed"


def test_table_values():
    assert MATERIALS["PLA"].density_kg_m3 == 1240
    assert MATERIALS["CF tube"].tensile_mpa == 600
    assert MATERIALS["Al 7075"].density_kg_m3 == 2810
    assert all(m.density_kg_m3 > 0 and m.kind for m in MATERIALS.values())


@pytest.mark.parametrize("alias,name", [("carbon", "CF tube"), ("CF", "CF tube"), ("carbon fiber tube", "CF tube"),
                                        ("Aluminium", "Al 6061"), ("aluminum", "Al 6061"), ("lwpla", "LW-PLA"),
                                        ("LW PLA", "LW-PLA"), ("petg", "PETG")])
def test_aliases(alias, name):
    assert get_material(alias).name == name


def test_bom_material_merged():
    p = Part(id="w", type="wing", material=Material(name="lwpla", density_kg_m3=0, kind="", infill=0.0, walls=1,
                                                   orientation="vase"))
    m = resolve_material(p)
    assert (m.name, m.kind, m.density_kg_m3, m.walls, m.infill, m.source) == ("LW-PLA", "printed", 1240, 1, 0.0, "bom")


def test_inferred_materials():
    assert resolve_material(Part(id="s", type="spar")).name == "CF tube"
    motor = resolve_material(Part(id="m", type="motor"))
    assert motor.sub["magnets"] == "NdFeB" and motor.sub["bell"] == "Al 6061"
    assert resolve_material(Part(id="p", type="prop")).name == "Nylon glass-filled"
    assert resolve_material(Part(id="s", type="servo", part_number="Emax ES08MA II")).sub["gears"] != "Nylon PA66"
    assert resolve_material(Part(id="s", type="servo", part_number="TowerPro SG90 plastic")).sub["gears"] == "Nylon PA66"
    assert resolve_material(Part(id="b", type="battery", part_number="6S2P P42A")).name == "Li-ion cell"
    assert resolve_material(Part(id="f", type="fastener", part_number="M5 nylon bolt")).name == "Nylon PA66"
    cat = resolve_material(Part(id="m", type="motor", part_number="BrotherHobby Avenger 2816 620KV"))
    assert cat.source == "catalog"


def test_vase_wing_panel_mass():
    """1 m x 0.2 m vase-mode LW-PLA panel: 100-350 g."""
    st = [(0.0, 0.2, 0.0, 0.0, "2412"), (1.0, 0.2, 0.0, 0.0, "2412")]
    mesh = mk.loft_surface(st)
    p = Part(id="w", type="wing", mesh=mesh,
             material=Material(name="LW-PLA", density_kg_m3=0, kind="", infill=0.0, walls=1, orientation="vase"))
    g, src = part_mass_g(p)
    assert src == "cad" and 100 <= g <= 350, g


def test_solid_printed_mass_uses_infill():
    mesh = mk.box((0.1, 0.1, 0.1), (0, 0, 0))
    p = Part(id="b", type="mount", mesh=mesh,
             material=Material(name="PLA", density_kg_m3=0, kind="", infill=0.2, walls=2, orientation="flat"))
    g, _ = part_mass_g(p)
    solid = 1e-3 * 1240 * 1000
    assert 0.2 * solid < g < 0.3 * solid


def test_mass_priority():
    assert part_mass_g(Part(id="m", type="motor", specs={"mass_g": 99})) == (99.0, "bom")
    p = Part(id="m", type="motor", specs={"mass_g": 50})
    p.sources["mass_g"] = "agent_supplied"
    assert part_mass_g(p) == (50.0, "agent_supplied")
    assert part_mass_g(Part(id="m", type="motor", part_number="BrotherHobby Avenger 2816 620KV")) == (128.0, "catalog")
    g, src = part_mass_g(Part(id="x", type="gps"))
    assert src == "assumed" and g > 0


def test_catalog_lookup():
    assert lookup("avenger 2816 620kv")["specs"]["kv"] == 620
    assert lookup("Avenger 2816 900KV")["specs"]["kv"] == 900
    assert lookup("Emax ES08MA II")["specs"]["torque_kgcm"] == 2.0
    assert lookup("ES08A II")["name"] == "Emax ES08A II"
    assert lookup("APC 11x8E")["specs"]["diam_in"] == 11
    assert lookup("6S2P Molicel P42A Li-ion")["specs"]["capacity_mah"] == 8400
    assert lookup("Holybro Remote ID")["specs"]["has_remote_id"] is True
    assert lookup("Matek F405-WTE")["specs"]["has_remote_id"] is False
    assert lookup("Falcon V2 Wing1 L") is None
    assert lookup(None) is None
