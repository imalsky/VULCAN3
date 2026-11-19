"""Up-front refusal of configs master cannot run.

1. A const_mix key that is not a network species: master's ini_y calls
   species.index(sp) and fails (build_atm.py:200).
2. A condense_sp entry outside the supported set: master's op.conden
   leaves its rate at zero.
3. A conver_ignore or diff_esc name the network lacks: master's species.index(sp) fails (op.py:1045-1046, :999).
4. A fix_species entry with no saturation curve under a cold-trap pin: master raises a KeyError (op.py:881).
5. Condensation settings master accepts and then ignores: a condensate left out of non_gas_sp, a remove_list
   entry on a condensation row that the rate update rewrites, and a condense_sp entry nothing acts on.
6. use_relax on a gas whose condensate the network lacks: a bare KeyError at setup.
7. A boundary flux or pin with a finite element_budget_tol: the closed-column budget cannot be met.

H2S alone (saturation capping, no kinetics) is master-legal and must pass.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"ini_mix": "const_mix", "const_mix": {"H2": 0.9, "Ar": 0.1}}, r"const_mix key 'Ar'"),
        ({"ini_mix": "const_mix", "const_mix": {"H2": 0.9, "He": 0.1}}, None),
        ({"use_condense": True, "condense_sp": ["Fe"]}, r"condense_sp entry 'Fe'"),
        ({"use_condense": True, "condense_sp": ["H2S"]}, None),  # sat-only tier
        ({"conver_ignore": ["HC3N", "HC3M"]}, r"conver_ignore entries \['HC3M'\]"),
        ({"diff_esc": ["H", "Xe"]}, r"diff_esc entries \['Xe'\]"),
        ({"use_condense": True, "condense_sp": ["H2O"], "fix_species": ["H2O", "CO"]}, r"fix_species entries \['CO'\]"),
        (
            {
                "use_condense": True,
                "condense_sp": ["H2O"],
                "fix_species": ["H2O", "CO"],
                "fix_species_from_coldtrap_lev": False,
                "element_budget_tol": float("inf"),
            },
            None,  # a whole-column pin needs no saturation curve
        ),
        ({"use_botflux": True}, r"use_botflux open the column, so element_budget_tol=0.01"),
        ({"use_fix_sp_bot": {"H2O": 1e-3}, "element_budget_tol": float("inf")}, None),
        ({"use_fix_H2He": True}, r"use_fix_H2He open the column"),
    ],
    ids=[
        "inert_const_mix_key",
        "network_const_mix_keys",
        "unknown_condensate",
        "h2s_sat_only",
        "unknown_conver_ignore",
        "unknown_diff_esc",
        "fix_species_without_curve",
        "fix_species_whole_column",
        "open_column_finite_budget",
        "open_column_budget_off",
        "fix_h2he_finite_budget",
    ],
)
def test_up_front_validation(overrides, error):
    from vulcan3 import make_config
    from vulcan3.runtime_validation import validate_runtime_config

    cfg = make_config(**overrides)
    if error is None:
        validate_runtime_config(cfg)
    else:
        with pytest.raises(RuntimeError, match=error):
            validate_runtime_config(cfg)


@pytest.fixture(scope="module")
def sulfur_net():
    """Return a packaged network with condensation rows (H2O and S8); the import-locked default has none."""
    from vulcan3 import network
    from vulcan3._paths import resolve_data_path

    return network.parse_network(str(resolve_data_path("thermo/SNCHO_photo_network_2025.txt")))


_S8_CONDENSATION = {"use_condense": True, "condense_sp": ["S8"], "non_gas_sp": ["S8_l_s"]}


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({}, None),
        ({"non_gas_sp": []}, "condensates ['S8_l_s'] are not in non_gas_sp"),
        # The network has no NH3 condensation row and no NH3_l_s.
        ({"condense_sp": ["NH3"], "non_gas_sp": []}, "condense_sp entries ['NH3'] would do nothing"),
        (
            {"condense_sp": ["NH3"], "non_gas_sp": [], "use_relax": ["NH3"]},
            "condense_sp entries ['NH3'] would do nothing",
        ),
        ({"condense_sp": ["NH3"], "non_gas_sp": [], "use_ini_cold_trap": True}, None),
        # The cold trap acts on NH3, so only the missing NH3_l_s stands in the way of relaxing it.
        (
            {"condense_sp": ["NH3"], "non_gas_sp": [], "use_ini_cold_trap": True, "use_relax": ["NH3"]},
            "use_relax entries ['NH3'] need their condensates ['NH3_l_s']",
        ),
        ({"condense_sp": ["NH3"], "non_gas_sp": [], "fix_species": ["NH3"]}, None),
        ({"condense_sp": ["H2S"], "non_gas_sp": []}, None),  # saturation-only tier, as in master
    ],
    ids=[
        "valid",
        "condensate_as_gas",
        "inert_condensate",
        "relax_without_its_condensate",
        "cold_trap_acts",
        "relax_needs_its_condensate",
        "pin_acts",
        "h2s_sat_only",
    ],
)
def test_condensation_settings_the_network_contradicts(sulfur_net, overrides, error):
    from vulcan3 import make_config
    from vulcan3.runtime_validation import _validate_condensation_network

    errors = _validate_condensation_network(make_config(**{**_S8_CONDENSATION, **overrides}), sulfur_net)
    assert errors == [] if error is None else any(error in e for e in errors), errors


def test_remove_list_on_a_live_condensation_row_is_refused(sulfur_net):
    """Removing a row the condensation update rewrites is refused; a relaxed gas's row, held at zero, is not."""
    from vulcan3 import make_config
    from vulcan3.runtime_validation import _validate_condensation_network

    row = {rf.split(" -> ")[0].strip(): i for i, rf in sulfur_net.Rf.items() if sulfur_net.is_conden[i]}
    s8 = make_config(**_S8_CONDENSATION, remove_list=[row["S8"], row["S8"] + 1])
    assert any(
        "remove_list names the condensation reactions" in e for e in _validate_condensation_network(s8, sulfur_net)
    )

    h2o = {"use_condense": True, "condense_sp": ["H2O"], "non_gas_sp": ["H2O_l_s"], "use_relax": ["H2O"]}
    relaxed = make_config(**h2o, remove_list=[row["H2O"], row["H2O"] + 1])
    assert _validate_condensation_network(relaxed, sulfur_net) == []


def test_kinetics_set_matches_sat_set():
    """The kinetics tier stays a subset of the saturation tier, and the only sat-only species is H2S."""
    from vulcan3._saturation import _SUPPORTED_CONDENSABLES
    from vulcan3.conden import SUPPORTED_CONDEN_KINETICS

    assert set(SUPPORTED_CONDEN_KINETICS) <= set(_SUPPORTED_CONDENSABLES)
    assert set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS) == {"H2S"}
