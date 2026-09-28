"""Attribute-name resolution for :class:`~idfkit.document.IDFDocument` collection accessors.

Lets every object type in a document's schema be reached as an attribute,

for example:

    doc.zones                            # Zone
    doc.air_loop_hvacs                   # AirLoopHVAC
    doc.coil_cooling_dx_single_speeds    # Coil:Cooling:DX:SingleSpeed
    doc.air_terminal_single_duct_vav_reheats
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

__all__ = [
    "AccessorAttributeError",
    "AccessorResolver",
    "pluralize",
    "snake_case",
]

# Names whose English plural is not formed by rule
_IRREGULAR_PLURALS = {"people": "people"}

_SPLIT_BEFORE_WORD = re.compile(r"(.)([A-Z][a-z]+)")
_SPLIT_AFTER_LOWER = re.compile(r"([a-z0-9])([A-Z])")
_COLLAPSE = re.compile(r"_+")


def snake_case(obj_type: str) -> str:
    """Convert an EnergyPlus object type to its ``snake_case`` singular form.

    Both ``:`` and ``-`` are treated as separators. The hyphen matters for exactly
    one type in the schema, ``PhotovoltaicPerformance:EquivalentOne-Diode``; without
    splitting on it the result is not a valid Python identifier.

    >>> snake_case("AirLoopHVAC")
    'air_loop_hvac'
    >>> snake_case("Coil:Cooling:DX:SingleSpeed")
    'coil_cooling_dx_single_speed'
    >>> snake_case("HVACTemplate:Zone:VAV")
    'hvac_template_zone_vav'
    >>> snake_case("PhotovoltaicPerformance:EquivalentOne-Diode")
    'photovoltaic_performance_equivalent_one_diode'

    The two regexes handle acronyms between them. ``_SPLIT_BEFORE_WORD`` requires a
    lowercase run after the capital, which is what makes ``HVACTemplate`` split as
    ``HVAC_Template`` rather than ``HVACT_emplate``.
    """
    s = obj_type.replace(":", "_").replace("-", "_")
    s = _SPLIT_BEFORE_WORD.sub(r"\1_\2", s)
    s = _SPLIT_AFTER_LOWER.sub(r"\1_\2", s)
    return _COLLAPSE.sub("_", s).lower().strip("_")


def pluralize(singular: str) -> str:
    """Pluralize a ``snake_case`` name.

    Handles the cases EnergyPlus type names actually present:

    >>> pluralize("zone")            # ordinary
    'zones'
    >>> pluralize("branch")          # sibilant
    'branches'
    >>> pluralize("lights")          # already plural, must not become 'lightses'
    'lights'
    >>> pluralize("internal_mass")   # singular ending in ss
    'internal_masses'
    >>> pluralize("people")          # irregular, not 'peoples'
    'people'
    """
    tail = singular.rsplit("_", 1)[-1]
    if tail in _IRREGULAR_PLURALS:
        return singular[: len(singular) - len(tail)] + _IRREGULAR_PLURALS[tail]
    if singular.endswith(("ss", "x", "z", "ch", "sh")):
        return singular + "es"
    if singular.endswith("s"):
        # Lights, Output:Schedules, ConvergenceLimits: already plural.
        return singular
    if singular.endswith("y") and len(singular) > 1 and singular[-2] not in "aeiou":
        return singular[:-1] + "ies"
    return singular + "s"


def _raw_key(obj_type: str) -> str:
    """Key for a raw type name used as an attribute.

    Case-insensitive, matching idfkit's own type lookup. Only ``:`` is stripped,
    never ``_``, so ``z_o_n_e`` cannot reach ``Zone``.

    >>> _raw_key("AirLoopHVAC")
    'airloophvac'
    >>> _raw_key("Coil:Cooling:DX:SingleSpeed")
    'coilcoolingdxsinglespeed'
    >>> _raw_key("z_o_n_e")
    'z_o_n_e'
    """
    return obj_type.replace(":", "").lower()


class AccessorResolver:
    """Maps attribute names to object types for one schema.

    Build once per schema and cache it on the schema (see
    :meth:`~idfkit.schema.EpJSONSchema.accessor_resolver`); construction is
    O(number of object types) and lookup is at most two dict hits.
    """

    def __init__(self, obj_types: Iterable[str]) -> None:
        self.attr_for: dict[str, str] = {}
        # snake_case plural and singular, matched exactly
        self._exact: dict[str, str] = {}
        # Raw type names, case-insensitive, with only ':' stripped
        self._raw: dict[str, str] = {}

        for obj_type in obj_types:
            singular = snake_case(obj_type)
            plural = pluralize(singular)
            self.attr_for[obj_type] = plural
            for alias in (plural, singular):
                self._exact.setdefault(alias, obj_type)
            self._raw.setdefault(_raw_key(obj_type), obj_type)

    def resolve(self, attr: str) -> str | None:
        """Return the object type an attribute name refers to, or ``None``.

        Snake_case forms match exactly. Raw type names match case-insensitively,
        so ``doc.ZONE`` works like ``doc["ZONE"]``, but separator noise such as
        ``doc.z_o_n_e`` or ``doc.zone_`` does not resolve.
        """
        hit = self._exact.get(attr)
        if hit is not None:
            return hit
        return self._raw.get(_raw_key(attr))

    def suggest(self, attr: str, n: int = 3) -> list[str]:
        """Closest canonical attribute names, for an ``AttributeError`` message."""
        return difflib.get_close_matches(attr.lower(), list(self.attr_for.values()), n=n, cutoff=0.6)


class AccessorAttributeError(AttributeError):
    """``AttributeError`` whose "Did you mean" suggestions are computed only when read.

    ``hasattr()`` and ``getattr(obj, name, default)`` catch this without ever
    calling ``__str__``, so the difflib cost is never paid on a silent probe.

    Pickles as a plain ``AttributeError`` carrying the final message, so it
    survives a trip from a worker process without shipping the resolver.
    """

    def __init__(self, owner: str, attr: str, resolver: AccessorResolver | None = None) -> None:
        super().__init__(attr)
        self.name = attr  # the standard AttributeError.name, typed str | None by the base class
        self._attr = attr  # a plain str copy for our own use, so pyright strict is satisfied
        self._owner = owner
        self._resolver = resolver
        self._message: str | None = None

    def __str__(self) -> str:
        if self._message is None:
            self._message = self._build()
        return self._message

    def __reduce__(self) -> tuple[type[AttributeError], tuple[str]]:
        # Rebuilding from self.args would call __init__ with one argument and fail.
        return (AttributeError, (str(self),))

    def _build(self) -> str:
        base = f"{self._owner!r} object has no attribute {self._attr!r}"
        if self._resolver is None:
            return base
        hits = self._resolver.suggest(self._attr)
        if not hits:
            return base
        width = max(len(h) for h in hits)
        lines = "\n".join(f"  {h:<{width}}  ({self._resolver.resolve(h)})" for h in hits)
        return f"{base}.\nDid you mean:\n{lines}"
