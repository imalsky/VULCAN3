"""Parse a VULCAN reaction-network text file into a typed Network.

Public names: `Network`, `parse_network`. The file format and index conventions are in docs/method.md.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Network", "parse_network"]

_SECTION_TWO_BODY = "two_body"
_SECTION_THREE_BODY_KINF = "three_body_kinf"
_SECTION_THREE_BODY_NO_KINF = "three_body_no_kinf"
_SECTION_SPECIAL = "special"
_SECTION_CONDEN = "conden"
_SECTION_RADIATIVE = "radiative"
_SECTION_PHOTO = "photo"
_SECTION_ION = "ion"

# Sections whose rows carry Arrhenius fits read from the file; only these
# have a meaningful documented temperature range in the trailing annotation.
_THERMAL_SECTIONS = (
    _SECTION_TWO_BODY,
    _SECTION_THREE_BODY_KINF,
    _SECTION_THREE_BODY_NO_KINF,
)


@dataclass(frozen=True)
class Network:
    """Parsed reaction network.

    All arrays use 1-based reaction indexing (entries [0] are unused).
    `nr` is the total number of slots including reverses; arrays have
    leading shape [nr+1, ...].
    """

    species: tuple[str, ...]
    species_idx: dict  # species -> int (0-based species index in `species`)
    ni: int  # number of species
    nr: int  # number of reaction slots (forward + reverse)

    # Stoichiometry, padded with `ni` (no-op slot pointing to y[ni]=1.0).
    # Shape [nr+1, max_terms]. Row 0 is unused (1-based indexing).
    reactant_idx: np.ndarray  # int64
    product_idx: np.ndarray  # int64
    reactant_stoich: np.ndarray  # float64
    product_stoich: np.ndarray  # float64

    # Per-reaction Arrhenius params (k = a * T^n * exp(-E/T)).
    # Shape [nr+1]. Set to 0 for slots with no Arrhenius (photo, conden, etc.)
    a: np.ndarray
    n: np.ndarray
    E: np.ndarray
    a_inf: np.ndarray
    n_inf: np.ndarray
    E_inf: np.ndarray

    # Reaction-type masks, shape [nr+1]
    is_forward: np.ndarray  # bool: True at odd indices
    is_three_body: np.ndarray  # bool: rate gets multiplied by M
    has_kinf: np.ndarray  # bool: uses Lindemann falloff
    is_special: np.ndarray  # bool: hardcoded rate (OH+CH3+M)
    is_conden: np.ndarray  # bool
    is_photo: np.ndarray  # bool: photodissociation
    is_ion: np.ndarray  # bool: photoionization

    # Section delimiters (parser-i values; 1-based)
    stop_rev_indx: int  # reverse slots filled for even i in 2..stop_rev_indx-1
    conden_indx: int  # parser-i of first condensation reaction

    # Photo metadata
    photo_sp: tuple[str, ...]  # species that photodissociate
    pho_rate_index: dict  # (species, branch) -> reaction parser-i
    n_branch: dict  # species -> num branches
    ion_sp: tuple[str, ...]  # species that photoionize
    ion_rate_index: dict  # (species, branch) -> reaction parser-i
    ion_branch: dict  # species -> num branches

    # Reaction text, indexed by parser-i (1-based)
    Rf: dict  # parser-i -> "A + B -> C + D"

    # Original file path for debugging
    network_path: str

    # Documented temperature validity, parser-i -> ((lo, hi), ...) in K, for
    # thermal (Arrhenius) forward rows only. An empty tuple means the row's
    # trailing annotation carries no parseable range. Advisory: nothing in
    # either VULCAN implementation enforces these at runtime. Consumed by
    # jwst-transit-authority (runtime_validation.rate_temp_range_exposure).
    temp_ranges: dict | None = None

    # (position, written id, reaction text) for every photo/ion row whose id
    # column disagrees with its parser position. Rates are indexed by
    # position, so the parse is correct either way, but a `cfg.remove_list`
    # written from a stale id column selects the wrong reaction; the setup
    # path announces it via `legacy_io._warn_stale_reaction_ids`.
    stale_ids: tuple[tuple[int, int, str], ...] = ()

    def __post_init__(self) -> None:
        """Make the tables read-only: one parse serves the whole process (`chem_funs.NETWORK`)."""
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)


_RE_LINE = re.compile(r"^\s*(\d*)\s*\[\s*([^\]]+)\s*\]\s*(.*)$")

# Temperature annotations in the trailing `Temp` column: '250-2580',
# '300-2.50E4', a bare measurement temperature '298', or comma-separated
# lists of these. No sign and no negative exponent: temperatures are
# positive Kelvin, and allowing 'E-' would make the range split ambiguous.
_NUM = r"\d+(?:\.\d+)?(?:[Ee]\+?\d+)?"
_RE_TEMP_RANGE = re.compile(rf"^({_NUM})-({_NUM})$")
_RE_TEMP_SINGLE = re.compile(rf"^{_NUM}$")


def _parse_temp_ranges(tokens: list[str]) -> tuple[tuple[float, float], ...]:
    """Parse the documented temperature range(s) off a row's annotation tokens.

    Scans from the end of the annotation (the Temp column is last) and stops
    at the first token that is not a range or a bare temperature. A reference
    fused to a range (e.g. '1986TSA/HAM1087300-2500') or an inverted 'lo-hi'
    yields no ranges rather than a garbage window.
    A bare temperature T becomes the degenerate range (T, T).
    """
    out: list[tuple[float, float]] = []
    for tok in reversed(tokens):
        t = tok.strip(",")
        if t == "/":  # k0 / k_inf window separator on three-body rows
            continue
        m = _RE_TEMP_RANGE.match(t)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                return ()
            out.append((lo, hi))
            continue
        if _RE_TEMP_SINGLE.match(t):
            v = float(t)
            out.append((v, v))
            continue
        break
    return tuple(reversed(out))


# Duplicate reactions listed by name in the parse error.
_DUPS_SHOWN = 4


def _announce_duplicate_reactions(network_path: str, dups: list[tuple[str, int, int]], duplicates_ok: bool) -> None:
    """Refuse (or, with `duplicates_ok`, warn about) repeated reactions.

    The parser is positional and appends every row, so both copies of a
    duplicated reaction become independent slots and their contributions are
    double-counted (forward AND reverse), silent scientific corruption, so
    the default is to raise. Upstream's `make_chem_funs.py` only prints
    (`check_duplicate()`). The vendored TiSNCHO network ships three such
    duplicates (an upstream data bug); no shipped config selects it.
    `duplicates_ok=True` downgrades to a RuntimeWarning for inspection and
    file-sweep uses that must parse the file anyway.
    """
    if not dups:
        return
    shown = "; ".join(f"{eq!r} at positions {first} and {second}" for eq, first, second in dups[:_DUPS_SHOWN])
    more = f" (+{len(dups) - _DUPS_SHOWN} more)" if len(dups) > _DUPS_SHOWN else ""
    msg = (
        f"Network {network_path} contains {len(dups)} duplicated reaction(s) "
        f"within the same section: {shown}{more}. Both copies are parsed as "
        "independent reactions, so their rate contributions are DOUBLE-COUNTED "
        "in both directions. Fix the network file before using it for science, "
        "or pass duplicates_ok=True to parse it anyway for inspection."
    )
    if not duplicates_ok:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _parse_term(term: str) -> tuple[int, str]:
    """Parse a reactant/product term like 'H', '2*H', '3*OH'.

    Returns (stoichiometric coefficient, species name).
    """
    parts = term.split("*")
    if len(parts) == 1:
        return 1, parts[0]
    return int(parts[0]), parts[1]


def _parse_eq(eq: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Parse 'A + B + M -> C + D + M' into ([(stoich, species), ...], [(stoich, species), ...]).

    M is preserved in the parsed lists -- the caller decides whether to
    include or exclude it.
    """
    if "->" not in eq:
        raise ValueError(f"Reaction equation missing '->': {eq!r}")
    lhs, rhs = eq.split("->", 1)
    reactants = [_parse_term(t.strip()) for t in lhs.split("+") if t.strip()]
    products = [_parse_term(t.strip()) for t in rhs.split("+") if t.strip()]
    return reactants, products


def _detect_section(line: str) -> str | None:
    """Return a new section name if `line` is a section marker, else None.

    Order matters: '# 3-body reactions without high-pressure rates' must
    match before '# 3-body'.
    """
    s = line.lstrip()
    if s.startswith("# 3-body reactions without high-pressure rates"):
        return _SECTION_THREE_BODY_NO_KINF
    if s.startswith("# 3-body"):
        return _SECTION_THREE_BODY_KINF
    if s.startswith("# special"):
        return _SECTION_SPECIAL
    if s.startswith("# condensation"):
        return _SECTION_CONDEN
    if s.startswith("# radiative"):
        return _SECTION_RADIATIVE
    if s.startswith("# photo"):
        return _SECTION_PHOTO
    if s.startswith("# ionisation") or s.startswith("# ionization"):
        return _SECTION_ION
    return None


@dataclass
class _ParseState:
    """What the reader accumulates while it walks a network file."""

    section: str = _SECTION_TWO_BODY
    parser_i: int = 1  # forwards: 1, 3, 5, ...
    stop_rev_indx: int | None = None
    conden_indx: int | None = None
    photo_indx: int | None = None
    species_order: list[str] = field(default_factory=list)
    species_idx: dict[str, int] = field(default_factory=dict)
    photo_sp: list[str] = field(default_factory=list)
    pho_rate_index: dict[tuple[str, int], int] = field(default_factory=dict)
    n_branch: dict[str, int] = field(default_factory=dict)
    ion_sp: list[str] = field(default_factory=list)
    ion_rate_index: dict[tuple[str, int], int] = field(default_factory=dict)
    ion_branch: dict[str, int] = field(default_factory=dict)
    forward_records: list[dict] = field(default_factory=list)
    Rf_text: dict[int, str] = field(default_factory=dict)
    temp_ranges: dict[int, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    stale_ids: list[tuple[int, int, str]] = field(default_factory=list)


def _enter_section(st: _ParseState, section: str, network_path: str, lineno: int) -> None:
    """Switch to `section`, recording where the condensation and photo sections start."""
    st.section = section
    if section == _SECTION_CONDEN and st.conden_indx is None:
        st.conden_indx = st.parser_i
    if section == _SECTION_RADIATIVE:
        raise ValueError(
            f"{network_path}:{lineno}: radiative-recombination networks are "
            "not supported -- rates_jax.py excludes those slots, "
            "so their reactions would be silently dropped."
        )
    if section == _SECTION_PHOTO and st.photo_indx is None:
        st.photo_indx = st.parser_i


def _collapse_side(side: Iterable[tuple[float, str]], species_idx: dict[str, int]) -> dict[int, float]:
    """Collapse one side of a reaction to species index -> stoichiometry, without M."""
    collapsed: dict[int, float] = {}
    for stoich, sp in side:
        if sp == "M":
            continue
        collapsed[species_idx[sp]] = collapsed.get(species_idx[sp], 0.0) + float(stoich)
    return collapsed


def _numeric_prefix(cols: list[str]) -> list[float]:
    """Return the leading numeric columns (Arrhenius and aux); the rest is commentary."""
    num_cols: list[float] = []
    for c in cols:
        try:
            num_cols.append(float(c))
        except ValueError:
            break
    return num_cols


def _read_reaction(st: _ParseState, file_id_str: str, eq: str, tail: str) -> None:
    """Record one reaction row at the current parser index."""
    section, parser_i = st.section, st.parser_i
    # A blank id column is legitimate: upstream never reads it and
    # renumbers from its own counter, so such rows are real reactions
    # (NCHO_full_photo_network.txt, SNCHO_photo_network_C3.txt).
    file_id = int(file_id_str) if file_id_str else 0
    eq = eq.strip()
    reactants, products = _parse_eq(eq)

    for _stoich, sp in reactants + products:
        if sp != "M" and sp not in st.species_idx:
            st.species_idx[sp] = len(st.species_order)
            st.species_order.append(sp)

    cols = tail.split()
    num_cols = _numeric_prefix(cols)
    rec: dict[str, Any] = {
        "parser_i": parser_i,
        "section": section,
        "Rf": eq,
        "reactants_collapsed": _collapse_side(reactants, st.species_idx),
        "products_collapsed": _collapse_side(products, st.species_idx),
        # M is tracked per side: a row with M on one side only gets different M-factors forward and reverse.
        "has_M_reac": any(sp == "M" for _, sp in reactants),
        "has_M_prod": any(sp == "M" for _, sp in products),
        "num_cols": num_cols,
        # Photo/ion-specific metadata: columns[0]=species, columns[1]=branch
        "photo_meta": (cols[0], int(cols[1]))
        if (section in (_SECTION_PHOTO, _SECTION_ION) and len(cols) >= 2 and cols[1].isdigit())
        else None,
    }
    st.forward_records.append(rec)
    st.Rf_text[parser_i] = eq
    if section in _THERMAL_SECTIONS:
        st.temp_ranges[parser_i] = _parse_temp_ranges(cols[len(num_cols) :])

    if section in (_SECTION_PHOTO, _SECTION_ION):
        if file_id != parser_i:
            st.stale_ids.append((parser_i, file_id, eq))
        sp_list, rate_index, n_br = (
            (st.photo_sp, st.pho_rate_index, st.n_branch)
            if section == _SECTION_PHOTO
            else (st.ion_sp, st.ion_rate_index, st.ion_branch)
        )
        target_sp = cols[0] if cols else eq.split()[0]
        if target_sp not in sp_list:
            sp_list.append(target_sp)
        if rec["photo_meta"] is not None:
            rate_index[rec["photo_meta"]] = parser_i
            n_br[target_sp] = max(n_br.get(target_sp, 0), rec["photo_meta"][1])

    st.parser_i += 2


def _read_file(network_path: str) -> _ParseState:
    """Walk the file once, in order."""
    st = _ParseState()
    with Path(network_path).open() as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # Section markers must be checked before the # reverse-stops marker.
            new_sec = _detect_section(line)
            if new_sec is not None:
                _enter_section(st, new_sec, network_path, lineno)
                continue

            stripped = line.lstrip()
            if stripped.startswith("# reverse stops"):
                st.stop_rev_indx = st.parser_i
                continue
            if stripped.startswith("#"):
                continue

            m = _RE_LINE.match(line)
            if m is None:
                raise ValueError(f"{network_path}:{lineno}: not a reaction row and not a comment: {line!r}")
            _read_reaction(st, m.group(1), m.group(2), m.group(3))
    return st


def _find_duplicates(forward_records: list[dict]) -> list[tuple[str, int, int]]:
    """Find reactions repeated within one section, which this positional parser would double-count."""
    seen_eq: dict[tuple, int] = {}
    dups: list[tuple[str, int, int]] = []
    for rec in forward_records:
        key = (
            rec["section"],
            tuple(sorted(rec["reactants_collapsed"].items())),
            tuple(sorted(rec["products_collapsed"].items())),
            rec["has_M_reac"],
            rec["has_M_prod"],
            rec["photo_meta"],
        )
        first_i = seen_eq.setdefault(key, rec["parser_i"])
        if first_i != rec["parser_i"]:
            dups.append((rec["Rf"], first_i, rec["parser_i"]))
    return dups


def _fill_stoichiometry(forward_records: list[dict], nr: int, ni: int) -> dict[str, np.ndarray]:
    """Build the reactant and product tables of both directions."""
    PAD = ni  # pad -> y[ni]=1, a no-op multiplier; RHS codegen and Jacobian tables skip it.

    # Reverse reactions store the forward's products in their reactant slot,
    # so both directions need to fit in the same `max_terms` width.
    max_terms = max([1] + [max(len(r["reactants_collapsed"]), len(r["products_collapsed"])) for r in forward_records])
    reactant_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    product_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    reactant_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    product_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    for rec in forward_records:
        i = rec["parser_i"]
        ir = i + 1
        for k_slot, (sp_idx, stoich) in enumerate(rec["reactants_collapsed"].items()):
            reactant_idx[i, k_slot] = sp_idx
            reactant_stoich[i, k_slot] = stoich
            product_idx[ir, k_slot] = sp_idx
            product_stoich[ir, k_slot] = stoich
        for k_slot, (sp_idx, stoich) in enumerate(rec["products_collapsed"].items()):
            product_idx[i, k_slot] = sp_idx
            product_stoich[i, k_slot] = stoich
            reactant_idx[ir, k_slot] = sp_idx
            reactant_stoich[ir, k_slot] = stoich
    return {
        "reactant_idx": reactant_idx,
        "product_idx": product_idx,
        "reactant_stoich": reactant_stoich,
        "product_stoich": product_stoich,
    }


def _fill_rows(forward_records: list[dict], nr: int) -> dict[str, np.ndarray]:
    """Build the Arrhenius coefficients and the per-row type masks."""
    arrays = {key: np.zeros(nr + 1, dtype=np.float64) for key in ("a", "n", "E", "a_inf", "n_inf", "E_inf")}
    masks = {
        key: np.zeros(nr + 1, dtype=bool)
        for key in ("is_forward", "is_three_body", "has_kinf", "is_special", "is_conden", "is_photo", "is_ion")
    }
    for rec in forward_records:
        i = rec["parser_i"]
        ir = i + 1
        masks["is_forward"][i] = True
        # M per side: a row with M on the LHS only gets a reverse without M.
        masks["is_three_body"][i] = rec["has_M_reac"]
        masks["is_three_body"][ir] = rec["has_M_prod"]

        sec = rec["section"]
        if sec == _SECTION_THREE_BODY_KINF:
            masks["has_kinf"][[i, ir]] = True
        if sec == _SECTION_SPECIAL:
            masks["is_special"][[i, ir]] = True
        if sec == _SECTION_CONDEN:
            masks["is_conden"][[i, ir]] = True
        if sec == _SECTION_PHOTO:
            masks["is_photo"][i] = True  # photo has no reverse; guarded via stop_rev_indx
        if sec == _SECTION_ION:
            masks["is_ion"][i] = True

        cols = rec["num_cols"]
        if sec in _THERMAL_SECTIONS:
            if len(cols) >= 3:
                arrays["a"][i], arrays["n"][i], arrays["E"][i] = cols[0], cols[1], cols[2]
            if sec == _SECTION_THREE_BODY_KINF and len(cols) >= 6:
                arrays["a_inf"][i], arrays["n_inf"][i], arrays["E_inf"][i] = cols[3], cols[4], cols[5]
    return {**arrays, **masks}


def parse_network(network_path: str | Path, *, duplicates_ok: bool = False) -> Network:
    """Parse a VULCAN network file; the format is in docs/method.md.

    Raises ValueError if a reaction is duplicated within one section (its
    contribution would be silently double-counted); `duplicates_ok=True`
    downgrades that to a RuntimeWarning for inspection/file-sweep uses.
    """
    from ._paths import resolve_data_path

    network_path = str(resolve_data_path(str(network_path)))
    st = _read_file(network_path)

    # parser_i has been bumped past the last reverse -> nr = parser_i - 1.
    nr = st.parser_i - 1
    ni = len(st.species_order)

    # upstream's check_duplicate() only prints; this parser refuses.
    _announce_duplicate_reactions(network_path, _find_duplicates(st.forward_records), duplicates_ok)

    stop_rev_indx = st.stop_rev_indx
    if stop_rev_indx is None:
        # Older networks may omit `# reverse stops`; default: reverses are
        # computed up to the photo section.
        stop_rev_indx = st.photo_indx if st.photo_indx is not None else nr + 1
    conden_indx = st.conden_indx if st.conden_indx is not None else nr + 1

    return Network(
        species=tuple(st.species_order),
        species_idx=dict(st.species_idx),
        ni=ni,
        nr=nr,
        **_fill_stoichiometry(st.forward_records, nr, ni),
        **_fill_rows(st.forward_records, nr),
        stop_rev_indx=stop_rev_indx,
        conden_indx=conden_indx,
        photo_sp=tuple(st.photo_sp),
        pho_rate_index=dict(st.pho_rate_index),
        n_branch=dict(st.n_branch),
        ion_sp=tuple(st.ion_sp),
        ion_rate_index=dict(st.ion_rate_index),
        ion_branch=dict(st.ion_branch),
        Rf=dict(st.Rf_text),
        network_path=network_path,
        temp_ranges=dict(st.temp_ranges),
        stale_ids=tuple(st.stale_ids),
    )
