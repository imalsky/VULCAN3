"""Up-front refusal of configs master cannot run.

1. A const_mix key that is not a network species: master's ini_y calls
   species.index(sp) and fails (build_atm.py:200).
2. A condense_sp entry outside the supported set: master's op.conden
   leaves its rate at zero.
3. A conver_ignore or diff_esc name the network lacks: master's species.index(sp) fails (op.py:1045-1046, :999).
4. A fix_species entry with no saturation curve under a cold-trap pin: master raises a KeyError (op.py:881).
5. Condensation settings master accepts and then ignores: a condensate left out of non_gas_sp.

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
            },
            None,  # a whole-column pin needs no saturation curve
        ),
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
    ],
    ids=["valid", "condensate_as_gas"],
)
def test_condensation_settings_the_network_contradicts(sulfur_net, overrides, error):
    from vulcan3 import make_config
    from vulcan3.runtime_validation import _validate_condensation_network

    errors = _validate_condensation_network(make_config(**{**_S8_CONDENSATION, **overrides}), sulfur_net)
    assert errors == [] if error is None else any(error in e for e in errors), errors


def test_kinetics_set_matches_sat_set():
    """The kinetics tier stays a subset of the saturation tier, and the only sat-only species is H2S."""
    from vulcan3._saturation import _SUPPORTED_CONDENSABLES
    from vulcan3.conden import SUPPORTED_CONDEN_KINETICS

    assert set(SUPPORTED_CONDEN_KINETICS) <= set(_SUPPORTED_CONDENSABLES)
    assert set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS) == {"H2S"}
