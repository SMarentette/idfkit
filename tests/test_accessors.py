"""Tests for document accessor name resolution (:mod:`idfkit._accessors`).

The sweep tests are the important ones. A resolver verified on a few hand-picked
examples with special uppercase in name
"""

from __future__ import annotations

import collections
import gc
import pickle
import weakref

import pytest

from idfkit import IDFDocument, new_document
from idfkit._accessors import AccessorAttributeError, AccessorResolver, pluralize, snake_case
from idfkit.objects import IDFCollection
from idfkit.schema import get_schema, get_schema_manager
from idfkit.versions import ENERGYPLUS_VERSIONS, LATEST_VERSION

# --------------------------------------------------------------------------
# snake_case: acronyms, colons, hyphens, digits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obj_type", "expected"),
    [
        ("Zone", "zone"),
        ("AirLoopHVAC", "air_loop_hvac"),
        ("AirLoopHVAC:UnitarySystem", "air_loop_hvac_unitary_system"),
        ("Coil:Cooling:DX:SingleSpeed", "coil_cooling_dx_single_speed"),
        ("AirTerminal:SingleDuct:VAV:Reheat", "air_terminal_single_duct_vav_reheat"),
        # HVACTemplate must not split as hvact_emplate
        ("HVACTemplate:Zone:VAV", "hvac_template_zone_vav"),
        ("ZoneHVAC:EquipmentList", "zone_hvac_equipment_list"),
        ("Chiller:Electric:ReformulatedEIR", "chiller_electric_reformulated_eir"),
        (
            "ElectricLoadCenter:Storage:LiIonNMCBattery",
            "electric_load_center_storage_li_ion_nmc_battery",
        ),
        ("ElectricLoadCenter:Inverter:PVWatts", "electric_load_center_inverter_pv_watts"),
        ("Output:SQLite", "output_sq_lite"),
        # The one hyphenated type in the schema -- must be a valid identifier.
        (
            "PhotovoltaicPerformance:EquivalentOne-Diode",
            "photovoltaic_performance_equivalent_one_diode",
        ),
    ],
)
def test_snake_case(obj_type: str, expected: str) -> None:
    assert snake_case(obj_type) == expected
    assert expected.isidentifier()


# --------------------------------------------------------------------------
# pluralize: the real cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("singular", "expected"),
    [
        ("zone", "zones"),
        ("branch", "branches"),  # sibilant
        ("refrigeration_case", "refrigeration_cases"),
        ("curve_fan_pressure_rise", "curve_fan_pressure_rises"),
        ("internal_mass", "internal_masses"),  # singular ending in ss
        ("material_no_mass", "material_no_masses"),
        ("lights", "lights"),  # already plural, not 'lightses'
        ("output_schedules", "output_schedules"),
        ("convergence_limits", "convergence_limits"),
        ("people", "people"),  # irregular, not 'peoples'
    ],
)
def test_pluralize(singular: str, expected: str) -> None:
    assert pluralize(singular) == expected


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


@pytest.fixture
def resolver() -> AccessorResolver:
    return AccessorResolver([
        "Zone",
        "ZoneList",
        "AirLoopHVAC",
        "AirLoopHVAC:UnitarySystem",
        "Branch",
        "BranchList",
        "Lights",
        "People",
        "InternalMass",
        "Coil:Cooling:DX:SingleSpeed",
        "Schedule:Compact",
    ])


def test_canonical_attribute_names(resolver: AccessorResolver) -> None:
    assert resolver.attr_for["Zone"] == "zones"
    assert resolver.attr_for["AirLoopHVAC"] == "air_loop_hvacs"
    assert resolver.attr_for["Coil:Cooling:DX:SingleSpeed"] == "coil_cooling_dx_single_speeds"
    assert resolver.attr_for["Lights"] == "lights"
    assert resolver.attr_for["People"] == "people"


def test_plural_singular_and_raw_all_resolve(resolver: AccessorResolver) -> None:
    for form in ("air_loop_hvacs", "air_loop_hvac", "AirLoopHVAC", "airloophvac", "AIRLOOPHVAC"):
        assert resolver.resolve(form) == "AirLoopHVAC"


def test_raw_name_with_colons_resolves(resolver: AccessorResolver) -> None:
    assert resolver.resolve("Coil:Cooling:DX:SingleSpeed") == "Coil:Cooling:DX:SingleSpeed"
    assert resolver.resolve("CoilCoolingDXSingleSpeed") == "Coil:Cooling:DX:SingleSpeed"


@pytest.mark.parametrize(
    "bad", ["z_o_n_e", "zone_", "zone_s", "_zone", "airloophvacs", "air_loophvacs", "AIR_LOOP_HVACS"]
)
def test_separator_noise_does_not_resolve(resolver: AccessorResolver, bad: str) -> None:
    assert resolver.resolve(bad) is None


def test_case_never_changes_which_type(resolver: AccessorResolver) -> None:
    """No string resolves to one type while its re-cased twin resolves to another."""
    clashes = [k for k in resolver._raw if k in resolver._exact and resolver._exact[k] != resolver._raw[k]]
    assert not clashes


def test_unknown_resolves_to_none(resolver: AccessorResolver) -> None:
    assert resolver.resolve("totally_not_a_type") is None


def test_suggestions_on_typo(resolver: AccessorResolver) -> None:
    assert "zones" in resolver.suggest("zonez")
    assert "branches" in resolver.suggest("brnaches")
    assert "lights" in resolver.suggest("lightss")


def test_attribute_error_names_the_intent(resolver: AccessorResolver) -> None:
    msg = str(AccessorAttributeError("Document", "zonez", resolver))
    assert "zonez" in msg
    assert "Did you mean" in msg
    assert "zones" in msg
    assert "(Zone)" in msg  # the object type is shown alongside


def test_attribute_error_stays_plain_when_nothing_is_close(resolver: AccessorResolver) -> None:
    msg = str(AccessorAttributeError("Document", "qqqqqqqqqq", resolver))
    assert "Did you mean" not in msg


def test_attribute_error_without_resolver_is_plain() -> None:
    assert str(AccessorAttributeError("Document", "zonez")) == "'Document' object has no attribute 'zonez'"


def test_attribute_error_is_an_attribute_error(resolver: AccessorResolver) -> None:
    err = AccessorAttributeError("Document", "zonez", resolver)
    assert isinstance(err, AttributeError)
    assert err.name == "zonez"


def test_attribute_error_pickles_as_plain_attribute_error(resolver: AccessorResolver) -> None:
    """Crossing a process boundary must not fail on the three-argument __init__."""
    restored = pickle.loads(pickle.dumps(AccessorAttributeError("Document", "zonez", resolver)))  # noqa: S301
    assert type(restored) is AttributeError
    assert "zones" in str(restored)


def test_hasattr_never_computes_suggestions(empty_doc: IDFDocument, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic rather than timed: a silent probe must never reach suggest()."""

    def boom(*_args: object, **_kwargs: object) -> list[str]:
        raise AssertionError("suggest() ran on a silent probe")  # noqa: TRY003

    monkeypatch.setattr(AccessorResolver, "suggest", boom)
    assert hasattr(empty_doc, "definitely_not_a_type") is False
    assert getattr(empty_doc, "definitely_not_a_type", None) is None


# --------------------------------------------------------------------------
# Integration with IDFDocument.__getattr__
# --------------------------------------------------------------------------


def test_document_resolves_arbitrary_type_by_attribute() -> None:
    doc = new_document(version=LATEST_VERSION)
    for form in ("air_loop_hvacs", "air_loop_hvac", "AirLoopHVAC"):
        coll = getattr(doc, form)
        assert isinstance(coll, IDFCollection)
    doc.add("AirLoopHVAC", "Main Loop", validate=False)
    assert len(doc.air_loop_hvacs) == 1
    assert len(doc.coil_cooling_dx_single_speeds) == 0


@pytest.mark.parametrize("good", ["zones", "zone", "Zone", "ZONE"])
def test_document_documented_forms_resolve(empty_doc: IDFDocument, good: str) -> None:
    # An empty type hands back a fresh collection each time, so compare the type, not identity.
    assert getattr(empty_doc, good).obj_type == "Zone"


def test_document_z_o_n_e_does_not_resolve(empty_doc: IDFDocument) -> None:
    """From the #202 review: "tighten _key a bit (doc.z_o_n_e shouldn't resolve)".

    Underscores are never stripped, so interleaving them through a type name is a
    typo that fails loudly instead of silently returning ``Zone``.
    """
    assert empty_doc.zones.obj_type == "Zone"  # the real name still works
    with pytest.raises(AttributeError, match="z_o_n_e"):
        _ = empty_doc.z_o_n_e


@pytest.mark.parametrize("bad", ["z_o_n_e", "zone_", "zone_s", "airloophvacs"])
def test_document_separator_noise_does_not_resolve(empty_doc: IDFDocument, bad: str) -> None:
    assert not hasattr(empty_doc, bad)


def test_document_attribute_error_suggests(empty_doc: IDFDocument) -> None:
    with pytest.raises(AttributeError, match="Did you mean"):
        _ = empty_doc.zonez


def test_document_without_schema_falls_back() -> None:
    doc = IDFDocument(version=LATEST_VERSION)  # no schema
    with pytest.raises(AttributeError):
        _ = doc.air_loop_hvacs


def test_resolver_is_built_once_per_schema() -> None:
    doc = new_document(version=LATEST_VERSION)
    first = doc._accessor_resolver_or_none()
    assert first is not None
    assert doc._accessor_resolver_or_none() is first
    assert new_document(version=LATEST_VERSION)._accessor_resolver_or_none() is first  # same cached schema


def test_schema_is_collectable_after_accessor_use() -> None:
    """The resolver must not pin its schema once the schema manager lets go of it.

    Uses the oldest bundled version so no shared fixture holds the same schema.
    """
    doc = new_document(version=ENERGYPLUS_VERSIONS[0])
    _ = doc.air_loop_hvacs  # forces the resolver to build
    ref = weakref.ref(doc.schema)

    del doc, _
    get_schema_manager().clear_cache()
    gc.collect()

    assert ref() is None, "schema still pinned after clear_cache()"


# --------------------------------------------------------------------------
# Sweeps across every bundled schema
# --------------------------------------------------------------------------


@pytest.fixture(params=ENERGYPLUS_VERSIONS, ids=lambda v: f"{v[0]}.{v[1]}.{v[2]}")
def all_obj_types(request: pytest.FixtureRequest) -> list[str]:
    version: tuple[int, int, int] = request.param
    return sorted(get_schema(version).object_types)


def test_no_attribute_name_collisions(all_obj_types: list[str]) -> None:
    r = AccessorResolver(all_obj_types)
    counts = collections.Counter(r.attr_for.values())
    dupes = {a: n for a, n in counts.items() if n > 1}
    assert not dupes, f"attribute name collisions: {dupes}"


def test_every_type_resolves_from_every_alias_form(all_obj_types: list[str]) -> None:
    r = AccessorResolver(all_obj_types)
    failures: list[tuple[str, str, str | None]] = []
    for obj_type in all_obj_types:
        for form in (r.attr_for[obj_type], snake_case(obj_type), obj_type):
            if r.resolve(form) != obj_type:
                failures.append((obj_type, form, r.resolve(form)))
    assert not failures, f"{len(failures)} alias failures, first 10: {failures[:10]}"


def test_case_never_changes_which_type_in_any_schema(all_obj_types: list[str]) -> None:
    r = AccessorResolver(all_obj_types)
    clashes = {k: (r._exact[k], r._raw[k]) for k in r._raw if k in r._exact and r._exact[k] != r._raw[k]}
    assert not clashes, f"re-casing changes the type: {clashes}"


def test_attribute_names_are_valid_identifiers(all_obj_types: list[str]) -> None:
    r = AccessorResolver(all_obj_types)
    bad = [a for a in r.attr_for.values() if not a.isidentifier()]
    assert not bad, f"not valid Python identifiers: {bad[:10]}"


def test_no_attribute_name_shadows_a_document_member(all_obj_types: list[str]) -> None:
    """An accessor must never collide with a real attribute or method.

    ``__getattr__`` only fires when normal lookup fails, so a collision would
    silently make that object type unreachable by attribute rather than raise.
    """
    reserved = set(dir(IDFDocument))
    r = AccessorResolver(all_obj_types)
    clashes = {a: t for t, a in r.attr_for.items() if a in reserved}
    assert not clashes, f"accessor names shadow IDFDocument members: {clashes}"
