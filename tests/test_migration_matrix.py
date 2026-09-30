"""The record migration matrix (pre-Phase-8 hardening; docs/PRE_PHASE8_INVARIANTS.md).

For every registered migration chain, a real current record (from the golden workflow) is
written as each older version -- the fields that version did not have are removed -- and
decoded. The migrated record must carry the documented *old meaning* for every added field
(``None`` = unknown / every unit / undeclared; never an invented value) and must survive
current -> JSON -> current unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import beyondnn as bnn
from beyondnn.schema import from_dict, from_json, to_dict, to_json
from beyondnn.schema.codec import _expected_id

AU = bnn.audits
Data = dict[str, Any]


def _drop(*fields: str) -> Callable[[Data], Data]:
    def f(d: Data) -> Data:
        return {k: v for k, v in d.items() if k not in fields}

    return f


def _drop_nested(key: str, *fields: str) -> Callable[[Data], Data]:
    def f(d: Data) -> Data:
        inner = {k: v for k, v in d[key].items() if k not in fields}
        return d | {key: inner}

    return f


def _plan_v2(d: Data) -> Data:  # v3 added selection.eligibility
    claims = [
        c if c.get("selection") is None else c | {"selection": _drop("eligibility")(c["selection"])}
        for c in d["claims"]
    ]
    return d | {"claims": claims}


def _plan_v1(d: Data) -> Data:  # v2 added claims[].roles and concepts[].roles
    return d | {
        "claims": [_drop("roles")(c) for c in d["claims"]],
        "concepts": [_drop("roles")(c) for c in d["concepts"]],
    }


def _plan_defaults_v2(d: Data) -> None:
    for c in d["claims"]:
        if c.get("selection") is not None:
            assert c["selection"]["eligibility"] is None  # old selection claims: every unit


def _plan_defaults_v1(d: Data) -> None:
    _plan_defaults_v2(d)
    assert all(c["roles"] == [] for c in d["claims"])  # no roles: Phase-7 standings
    assert all(c["roles"] == [] for c in d["concepts"])


def _flat_defaults(**defaults: Any) -> Callable[[Data], None]:
    def f(d: Data) -> None:
        for k, v in defaults.items():
            assert d[k] == v, (k, d[k])

    return f


def _nested_defaults(key: str, **defaults: Any) -> Callable[[Data], None]:
    def f(d: Data) -> None:
        for k, v in defaults.items():
            assert d[key][k] == v, (key, k, d[key][k])

    return f


# (kind, current version) -> [(older version, down-step from the next version, expected)]
MATRIX: dict[str, list[tuple[int, Callable[[Data], Data], Callable[[Data], None]]]] = {
    "evidence_selection": [
        (2, _drop("eligible", "eligibility"), _flat_defaults(eligible=None, eligibility=None)),
        (
            1,
            _drop("unit_axes", "unit_reduction"),
            _flat_defaults(unit_axes=None, unit_reduction=None),
        ),
    ],
    "audit_plan": [(2, _plan_v2, _plan_defaults_v2), (1, _plan_v1, _plan_defaults_v1)],
    "claim": [
        (2, _drop_nested("subject", "feature"), _nested_defaults("subject", feature=None)),
        (1, _drop_nested("subject", "unit_axes"), _nested_defaults("subject", unit_axes=None)),
    ],
    "intervention": [
        (
            3,
            _drop("direction", "direction_axis"),
            _flat_defaults(direction=None, direction_axis=None),
        ),
        (2, _drop("unit_axes"), _flat_defaults(unit_axes=None)),
        (1, _drop("units", "retain"), _flat_defaults(units=None, retain=False)),
    ],
    "input": [
        (2, _drop("sample_id"), _flat_defaults(sample_id=None)),  # never invented
        (1, _drop("pass_index"), _flat_defaults(pass_index=None)),
    ],
    "output": [(1, _drop("pass_index"), _flat_defaults(pass_index=None))],
    "execution_occurrence": [(1, _drop("pass_index"), _flat_defaults(pass_index=None))],
    "provenance": [(1, _drop("declared_model"), _flat_defaults(declared_model=None))],
    "causal_effect": [
        (1, _drop_nested("metric", "declaration"), _nested_defaults("metric", declaration=None))
    ],
}


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real current record of every migrated kind, from the golden workflow."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "golden_workflow", Path(__file__).parent / "test_golden_workflow.py"
    )
    assert spec is not None
    assert spec.loader is not None
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    out_dir: Path = tmp_path_factory.mktemp("golden")
    G.build(out_dir)
    found: dict[str, Any] = {}
    plan = from_json((out_dir / "plan.json").read_text())
    found["audit_plan"] = plan
    for path in AU.load_evidence(out_dir / "evidence"):
        for rec in bnn.load_trace(path).records:
            if rec.KIND in MATRIX and rec.KIND not in found:
                found[rec.KIND] = rec
    return found


def test_every_registered_migration_is_in_the_matrix() -> None:
    from beyondnn.schema.base import _MIGRATIONS

    registered = {(kind, v) for kind, v in _MIGRATIONS}
    covered = {(kind, v) for kind, rows in MATRIX.items() for v, _, _ in rows}
    assert registered == covered


@pytest.mark.parametrize(
    ("kind", "old"), [(k, v) for k, rows in MATRIX.items() for v, _, _ in rows]
)
def test_old_version_migrates_to_current_with_its_old_meaning(
    records: dict[str, Any], kind: str, old: int
) -> None:
    current = records[kind]
    env = to_dict(current)
    data = copy.deepcopy(env["data"])
    expect: list[Callable[[Data], None]] = []
    for v, down, expected in MATRIX[kind]:  # step down: current -> ... -> old
        data = down(data)
        expect.append(expected)
        if v == old:
            break
    old_env = env | {
        "record_version": old,
        "id": _expected_id(kind, old, data),
        "data": data,
    }
    migrated = from_dict(copy.deepcopy(old_env))
    assert migrated.RECORD_VERSION == type(current).RECORD_VERSION
    migrated_data = to_dict(migrated)["data"]
    for check in expect:
        check(migrated_data)
    # every field the old version had is unchanged
    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                assert migrated_data[key][k] == v, (key, k)
        elif key not in ("claims", "concepts"):
            assert migrated_data[key] == value, key
    assert from_json(to_json(migrated)) == migrated  # current -> JSON -> current


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_current_records_round_trip(records: dict[str, Any], kind: str) -> None:
    rec = records[kind]
    assert from_json(to_json(rec)) == rec


def test_declared_eligibility_roles_targets_and_replacements_round_trip(
    records: dict[str, Any],
) -> None:
    plan = records["audit_plan"]
    again = from_json(to_json(plan))
    assert again == plan
    eligibilities = {c.selection.eligibility for c in plan.claims if c.selection is not None}
    assert eligibilities == {None, "units_1_to_23"}
    assert all(c.roles for c in plan.claims)  # roles survive
    assert all(len(c.sample_targets) == len(plan.samples) for c in plan.claims)  # per-sample
