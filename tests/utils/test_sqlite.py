import datetime
import threading

import pyarrow as pa
import pytest

from triad.exceptions import InvalidOperationError
from triad.utils.sqlite import (
    RepairStatus,
    SQLiteRepairLoader,
)

INPUT = "name:str,amount:str,unit_price:str"
OUTPUT = "name:str,amount:double,qty:long"


def _rows(amount, qty=1):
    return [{"name": "item", "amount": float(amount), "qty": qty}]


def _cleaner(row):
    price = float(row["amount"])
    unit = 1.0 if row["unit_price"] is None else float(row["unit_price"])
    qty = int(round(price / unit)) if unit else 1
    yield {"name": (row["name"] or "").upper(), "amount": price, "qty": qty}


def _loader(**kwargs):
    params = dict(input_schema=INPUT, output_schema=OUTPUT)
    params.update(kwargs)
    return SQLiteRepairLoader(":memory:", **params)


def test_add_and_correct_retracts_rows_and_bumps_revision():
    loader = _loader()
    assert loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"}) == 1
    record = loader.get_record("r1")
    assert record.revision == 1
    assert record.status == RepairStatus.PENDING
    assert record.rows == 0
    assert record.raw == {"name": "a", "amount": "10", "unit_price": "2"}

    assert loader.tasks("r1") == []

    result = loader.run("r1", _cleaner)
    assert result.status == RepairStatus.SUCCESS
    assert result.applied
    assert result.rows == 1
    assert result.reason is None
    assert not result.stale and not result.duplicate
    assert loader.get_record("r1").status == RepairStatus.SUCCESS
    assert loader.read("r1").to_pylist() == [{"name": "A", "amount": 10.0, "qty": 5}]

    assert loader.correct("r1", {"name": "b", "amount": "20", "unit_price": "4"}) == 2
    record = loader.get_record("r1")
    assert record.revision == 2
    assert record.status == RepairStatus.PENDING
    assert record.rows == 0
    assert record.reason is None
    assert record.raw == {"name": "b", "amount": "20", "unit_price": "4"}
    # the old cleaned row is retracted immediately
    assert loader.read("r1").to_pylist() == []

    # every revision keeps its own raw snapshot
    assert [x.revision for x in loader.revisions("r1")] == [1, 2]
    assert loader.revisions("r1")[0].raw["name"] == "a"
    assert loader.revisions("r1")[1].raw["name"] == "b"

    result = loader.run("r1", _cleaner)
    assert result.rows == 1
    assert loader.read("r1").to_pylist() == [{"name": "B", "amount": 20.0, "qty": 5}]


def test_correct_without_new_values_keeps_snapshot_for_retry():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    assert loader.run("r1", lambda r: 1 / 0).status == RepairStatus.FAILED
    assert loader.correct("r1") == 2
    assert loader.get_record("r1").raw == {
        "name": "a",
        "amount": "10",
        "unit_price": "2",
    }
    result = loader.run("r1", _cleaner)
    assert result.applied
    assert loader.read("r1").to_pylist() == [{"name": "A", "amount": 10.0, "qty": 5}]


def test_task_uses_submission_snapshot_and_stale_completion_changes_nothing():
    loader = _loader()
    loader.add("r1", {"name": "old", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    assert task.raw == {"name": "old", "amount": "10", "unit_price": "2"}
    assert task.status == RepairStatus.PENDING
    loader.correct("r1", {"name": "new", "amount": "30", "unit_price": "5"})
    assert loader.get_record("r1").status == RepairStatus.PENDING

    result = loader.run_task(task, _cleaner)
    assert result.stale
    assert not result.applied
    assert result.status == RepairStatus.SUCCESS
    assert result.rows == 1  # computed from the snapshot of revision 1
    assert loader.get_record("r1").revision == 2
    assert loader.get_record("r1").status == RepairStatus.PENDING
    assert loader.read("r1").to_pylist() == []
    stored = loader.get_task(task.task_id)
    assert stored.status == RepairStatus.SUCCESS
    assert not stored.applied

    # the task row keeps the trace, the current record is untouched
    assert loader.tasks("r1", RepairStatus.PENDING) == []


def test_stale_failure_changes_nothing():
    loader = _loader()
    loader.add("r1", {"name": "old", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    loader.correct("r1", {"name": "new", "amount": "30", "unit_price": "5"})
    result = loader.run_task(task, lambda r: 1 / 0)
    assert result.status == RepairStatus.FAILED
    assert result.stale
    assert not result.applied
    assert loader.get_record("r1").status == RepairStatus.PENDING
    assert loader.get_record("r1").reason is None


def test_zero_to_many_rows_and_success_with_zero_rows():
    loader = _loader()
    loader.add("zero", {"name": "a", "amount": "0", "unit_price": "1"})
    result = loader.run("zero", lambda r: None)
    assert result.status == RepairStatus.SUCCESS
    assert result.applied
    assert result.rows == 0
    assert loader.get_record("zero").status == RepairStatus.SUCCESS
    assert loader.get_record("zero").rows == 0
    assert loader.read("zero").to_pylist() == []
    assert loader.read("zero").schema == loader.output_schema

    loader.add("many", {"name": "a", "amount": "3", "unit_price": "1"})
    result = loader.run(
        "many", lambda r: ({"name": "x", "amount": 1.0, "qty": i} for i in range(3))
    )
    assert result.rows == 3
    assert loader.get_record("many").rows == 3
    assert [x["qty"] for x in loader.read("many").to_pylist()] == [0, 1, 2]


def test_cleaner_output_forms():
    loader = _loader()

    def single(row):
        return {"name": "single", "amount": 1.0, "qty": 1}

    def listing(row):
        return [{"name": "list", "amount": 2.0, "qty": 2}] * 2

    loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1"})
    assert loader.run("r1", single).rows == 1
    loader.add("r2", {"name": "a", "amount": "1", "unit_price": "1"})
    assert loader.run("r2", listing).rows == 2
    assert loader.read_all().to_pylist() == [
        {"record_id": "r1", "name": "single", "amount": 1.0, "qty": 1},
        {"record_id": "r2", "name": "list", "amount": 2.0, "qty": 2},
        {"record_id": "r2", "name": "list", "amount": 2.0, "qty": 2},
    ]


def test_mid_generator_failure_leaves_no_partial_rows():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "20", "unit_price": "1"})
    loader.run("r1", _cleaner)
    good = loader.read("r1").to_pylist()

    loader.add("r2", {"name": "a", "amount": "20", "unit_price": "1"})

    def boom(row):
        for i in range(3):
            if i == 1:
                raise RuntimeError("cleaner exploded")
            yield {"name": "x", "amount": float(i), "qty": i}

    result = loader.run("r2", boom)
    assert result.status == RepairStatus.FAILED
    assert result.applied
    assert result.rows == 0
    assert "cleaner exploded" in result.reason
    record = loader.get_record("r2")
    assert record.status == RepairStatus.FAILED
    assert record.rows == 0
    assert "cleaner exploded" in record.reason
    assert loader.read("r2").to_pylist() == []
    # other records are untouched
    assert loader.read("r1").to_pylist() == good


def test_output_conversion_failure_leaves_no_partial_rows():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "20", "unit_price": "1"})

    def bad_second_row(row):
        yield {"name": "ok", "amount": 1.0, "qty": 1}
        yield {"name": "no", "amount": "not-a-number", "qty": 2}

    result = loader.run("r1", bad_second_row)
    assert result.status == RepairStatus.FAILED
    assert "amount" in result.reason
    assert loader.get_record("r1").rows == 0
    assert loader.read("r1").to_pylist() == []


def test_output_integer_overflow_fails_instead_of_breaking_read():
    loader = SQLiteRepairLoader(
        ":memory:", input_schema="x:str", output_schema="a:int8,b:uint8"
    )
    loader.add("bad", {"x": "1"})
    result = loader.run("bad", lambda row: {"a": 200, "b": 1})
    assert result.status == RepairStatus.FAILED
    assert "int8" in result.reason
    assert loader.get_record("bad").rows == 0
    # the stored state is still readable in the declared types
    assert loader.read("bad").schema == loader.output_schema
    assert loader.read("bad").num_rows == 0

    loader.add("ok", {"x": "1"})
    assert loader.run("ok", lambda row: {"a": 127, "b": 255}).applied
    assert loader.read("ok").to_pylist() == [{"a": 127, "b": 255}]


def test_input_schema_conversion_runs_before_cleaner():
    loader = SQLiteRepairLoader(
        ":memory:", input_schema="name:str,amount:double", output_schema=OUTPUT
    )
    loader.add("r1", {"name": "a", "amount": "10.5"})
    seen = {}

    def capture(row):
        seen.update(row)
        return {"name": row["name"], "amount": row["amount"], "qty": 1}

    loader.run("r1", capture)
    assert seen["amount"] == 10.5


def test_input_conversion_failure_is_recorded():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "abc", "unit_price": "1"})
    result = loader.run("r1", _cleaner)
    assert result.status == RepairStatus.FAILED
    assert result.reason
    assert loader.read("r1").to_pylist() == []


def test_only_first_completion_of_current_pending_revision_applies():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    first = loader.submit("r1")
    second = loader.submit("r1")
    assert first.task_id != second.task_id

    r1 = loader.run_task(first, _cleaner)
    assert r1.applied
    assert loader.read("r1").to_pylist() == [{"name": "A", "amount": 10.0, "qty": 5}]

    r2 = loader.run_task(second, lambda r: _rows(999.0, 7))
    assert not r2.applied
    assert not r2.duplicate
    assert loader.get_record("r1").status == RepairStatus.SUCCESS
    assert loader.read("r1").to_pylist() == [{"name": "A", "amount": 10.0, "qty": 5}]
    assert loader.get_task(second.task_id).status == RepairStatus.SUCCESS


def test_duplicate_completion_of_same_task_changes_nothing():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    loader.run_task(task, _cleaner)
    before = loader.read("r1").to_pylist()
    assert before == [{"name": "A", "amount": 10.0, "qty": 5}]

    again = loader.run_task(task, lambda r: _rows(1.0, 1))
    assert again.duplicate
    assert not again.applied
    assert loader.read("r1").to_pylist() == before
    assert loader.get_record("r1").status == RepairStatus.SUCCESS
    assert loader.get_task(task.task_id).status == RepairStatus.SUCCESS
    assert loader.get_task(task.task_id).rows == 1


def test_first_failure_blocks_later_success_until_correction():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    failed = loader.run_task(task, lambda r: 1 / 0)
    assert failed.applied
    assert loader.get_record("r1").status == RepairStatus.FAILED

    other = loader.submit("r1")
    ok = loader.run_task(other, _cleaner)
    assert not ok.applied
    assert loader.get_record("r1").status == RepairStatus.FAILED
    assert loader.read("r1").to_pylist() == []
    assert loader.get_task(other.task_id).status == RepairStatus.SUCCESS

    assert loader.correct("r1", {"name": "b", "amount": "20", "unit_price": "4"}) == 2
    applied = loader.run("r1", _cleaner)
    assert applied.applied
    assert loader.get_record("r1").status == RepairStatus.SUCCESS
    assert loader.read("r1").to_pylist() == [{"name": "B", "amount": 20.0, "qty": 5}]


def test_pending_revision_only_gets_first_completion():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    loader.run_task(task, _cleaner)
    assert loader.get_record("r1").status == RepairStatus.SUCCESS

    # no correction: another task for the same revision can not overwrite
    result = loader.run("r1", _cleaner)
    assert not result.applied
    assert loader.get_record("r1").status == RepairStatus.SUCCESS


def test_concurrent_duplicate_runs_apply_once():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    results = []
    barrier = threading.Barrier(4)

    def work(n):
        barrier.wait()
        results.append(loader.run_task(task, lambda r: _rows(1.0, n)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for x in results if x.applied) == 1
    assert len(loader.read("r1").to_pylist()) == 1


def test_read_keeps_declared_types_duplicates_and_nulls():
    loader = SQLiteRepairLoader(
        ":memory:",
        input_schema="i:long,j:long,s:str",
        output_schema="a:int,b:uint64,c:float,d:double,e:bool,f:str,g:binary,"
        "h:date,t:timestamp(us,UTC),n:null",
    )
    loader.add(1, {"i": 1, "j": 2, "s": "x"})
    loader.add("1", {"i": 3, "j": 4, "s": "y"})
    row = {
        "a": -3,
        "b": 2**64 - 1,
        "c": 1.5,
        "d": 2.25,
        "e": True,
        "f": "x\u0000y",
        "g": b"\x00\xff",
        "h": datetime.date(2026, 1, 2),
        "t": datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        "n": None,
    }
    nulls = dict(row)
    nulls.update(a=None, b=None, c=None, d=None, e=None, f=None, g=None, n=None)
    nulls.pop("h")
    nulls.pop("t")
    nulls.update(h=None, t=None)

    def clean(r):
        return [row, row, nulls]

    for rid in [1, "1"]:
        loader.run(rid, clean)
    table = loader.read(1)
    assert table.schema == loader.output_schema
    assert table.num_rows == 3
    assert table.to_pylist() == [row, row, nulls]
    assert loader.read("1").num_rows == 3
    assert len(loader.read_all().to_pylist()) == 6

    b = table.column("b")
    assert b.type == pa.uint64()
    assert b[0].as_py() == 2**64 - 1


def test_reopen_database_keeps_records_tasks_and_rows():
    tmp = pytest.importorskip("tempfile").mkdtemp()
    path = tmp + "/repair.db"
    loader = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    loader.add(7, {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit(7)
    loader.run_task(task, _cleaner)
    loader.correct(7, {"name": "z", "amount": "5", "unit_price": "1"})
    loader.close()

    loader = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    assert loader.get_record(7).revision == 2
    assert loader.get_record(7).status == RepairStatus.PENDING
    assert loader.read(7).to_pylist() == []
    assert loader.get_task(task.task_id).status == RepairStatus.SUCCESS
    assert [r.revision for r in loader.revisions(7)] == [1, 2]
    result = loader.run("7" if False else 7, _cleaner)
    assert result.applied
    assert loader.read(7).to_pylist() == [{"name": "Z", "amount": 5.0, "qty": 5}]
    loader.close()


def test_reopen_with_different_schema_is_rejected():
    tmp = pytest.importorskip("tempfile").mkdtemp()
    path = tmp + "/repair.db"
    loader = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    loader.close()
    with pytest.raises(InvalidOperationError):
        SQLiteRepairLoader(path, input_schema=INPUT, output_schema="name:str")


def test_scalar_types_only():
    loader = _loader()
    for schema in [
        pa.schema([("a", pa.list_(pa.int64()))]),
        pa.schema([("a", pa.struct([("x", pa.int64())]))]),
        pa.schema([("a", pa.decimal128(10, 2))]),
        pa.schema([("a", pa.date64())]),
        pa.schema([("a", pa.large_string())]),
        pa.schema([("a", pa.binary(3))]),
    ]:
        with pytest.raises(NotImplementedError):
            SQLiteRepairLoader(":memory:", input_schema="a:str", output_schema=schema)
    assert loader.input_schema.names == ["name", "amount", "unit_price"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SQLiteRepairLoader(":memory:", input_schema="", output_schema="a:str")
    with pytest.raises(ValueError):
        SQLiteRepairLoader(":memory:", input_schema="a:str", output_schema="")
    with pytest.raises(ValueError):
        SQLiteRepairLoader(":memory:", input_schema="a:str,a:str", output_schema="a")
    loader = _loader()
    with pytest.raises(ValueError):
        loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1", "x": 1})
    with pytest.raises(TypeError):
        loader.add(True, {"name": "a"})
    with pytest.raises(TypeError):
        loader.add(1.5, {"name": "a"})
    with pytest.raises(InvalidOperationError):
        loader.add("r1", {"name": "a"})
        loader.add("r1", {"name": "a"})
    with pytest.raises(KeyError):
        loader.get_record("missing")
    with pytest.raises(KeyError):
        loader.correct("missing", {"name": "a"})
    with pytest.raises(KeyError):
        loader.submit("missing")
    with pytest.raises(KeyError):
        loader.read("missing")
    with pytest.raises(KeyError):
        loader.revisions("missing")
    with pytest.raises(KeyError):
        loader.get_task(123)
    with pytest.raises(KeyError):
        loader.run_task(123, _cleaner)
    with pytest.raises(ValueError):
        loader.records("unknown")
    with pytest.raises(TypeError):
        loader.run("r1", "not callable")


def test_extra_output_field_fails_without_rows():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1"})
    result = loader.run(
        "r1", lambda r: {"name": "x", "amount": 1.0, "qty": 1, "oops": 2}
    )
    assert result.status == RepairStatus.FAILED
    assert "oops" in result.reason
    assert loader.read("r1").to_pylist() == []


def test_closed_loader_rejects_all_operations():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1"})
    with loader:
        loader.run("r1", _cleaner)
        assert loader.read("r1").num_rows == 1
    for op in [
        lambda: loader.add("r2", {"name": "b"}),
        lambda: loader.read("r1"),
        lambda: loader.get_record("r1"),
        lambda: loader.run("r1", _cleaner),
    ]:
        with pytest.raises(InvalidOperationError):
            op()


def test_float_specials_round_trip():
    loader = SQLiteRepairLoader(
        ":memory:", input_schema="a:str", output_schema="x:double"
    )
    loader.add("r", {"a": "x"})
    values = [float("nan"), float("inf"), float("-inf"), 1.5, None]
    loader.run("r", lambda row: [{"x": v} for v in values])
    result = loader.read("r").to_pylist()
    assert len(result) == len(values)
    # apply_schema normalizes NaN to NULL, the other special values survive
    assert result[0]["x"] is None
    assert result[1]["x"] == float("inf")
    assert result[2]["x"] == float("-inf")
    assert result[3]["x"] == 1.5
    assert result[4]["x"] is None


def test_record_id_int_and_str_are_distinct():
    loader = _loader()
    raw = {"name": "a", "amount": "1", "unit_price": "1"}
    loader.add(1, raw)
    loader.add("1", raw)
    assert loader.get_record(1).record_id == 1
    assert loader.get_record("1").record_id == "1"
    loader.run(1, _cleaner)
    loader.run("1", _cleaner)
    assert loader.read(1).num_rows == 1
    assert loader.read("1").num_rows == 1
    loader.correct(1)
    assert loader.get_record(1).revision == 2
    assert loader.get_record("1").revision == 1


def test_retry_after_correction_uses_new_snapshot():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    loader.run("r1", lambda r: 1 / 0)
    record = loader.get_record("r1")
    assert record.status == RepairStatus.FAILED
    assert record.reason

    loader.correct("r1", {"name": "b", "amount": "20", "unit_price": "4"})
    record = loader.get_record("r1")
    assert record.status == RepairStatus.PENDING
    assert record.reason is None

    result = loader.run("r1", _cleaner)
    assert result.applied
    assert loader.read("r1").to_pylist() == [{"name": "B", "amount": 20.0, "qty": 5}]


def test_reopen_then_first_completion_still_applies_once():
    tmp = pytest.importorskip("tempfile").mkdtemp()
    path = tmp + "/repair.db"
    loader = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    task = loader.submit("r1")
    loader.close()

    # a second process finished the task while we were away
    other = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    assert other.run_task(task, _cleaner).applied
    other.close()

    loader = SQLiteRepairLoader(path, input_schema=INPUT, output_schema=OUTPUT)
    late = loader.run_task(task, lambda r: _rows(9.0, 9))
    assert late.duplicate
    assert not late.applied
    assert loader.read("r1").to_pylist() == [{"name": "A", "amount": 10.0, "qty": 5}]
    loader.close()


def test_read_all_orders_by_record_and_keeps_duplicates():
    loader = _loader()
    for rid in ["b", "a"]:
        loader.add(rid, {"name": rid, "amount": "1", "unit_price": "1"})
        loader.run(rid, lambda r: [{"name": r["name"], "amount": 1.0, "qty": 1}] * 2)
    table = loader.read_all()
    assert table.column_names == ["record_id"] + loader.output_schema.names
    assert [x["record_id"] for x in table.to_pylist()] == ["a", "a", "b", "b"]


def test_record_schema_types_are_declared_after_reopen():
    tmp = pytest.importorskip("tempfile").mkdtemp()
    path = tmp + "/repair.db"
    schema = "a:int,b:uint64,c:float,d:bool,e:str,f:binary,g:date,h:timestamp(us,UTC)"
    loader = SQLiteRepairLoader(path, input_schema="i:str", output_schema=schema)
    loader.add("r1", {"i": "x"})
    row = {
        "a": -1,
        "b": 2**64 - 1,
        "c": 0.5,
        "d": False,
        "e": "text",
        "f": b"\x01",
        "g": datetime.date(2026, 5, 6),
        "h": datetime.datetime(2026, 5, 6, 7, 8, 9, tzinfo=datetime.timezone.utc),
    }
    loader.run("r1", lambda r: row)
    expected = loader.read("r1")
    loader.close()

    loader = SQLiteRepairLoader(path, input_schema="i:str", output_schema=schema)
    actual = loader.read("r1")
    assert actual.schema == expected.schema
    assert actual.equals(expected)
    loader.close()


def test_records_filter_and_lookup():
    loader = _loader()
    loader.add("a", {"name": "a", "amount": "1", "unit_price": "1"})
    loader.add("b", {"name": "b", "amount": "2", "unit_price": "1"})
    loader.run("a", _cleaner)
    assert [r.record_id for r in loader.records()] == ["a", "b"]
    assert [r.record_id for r in loader.records(RepairStatus.SUCCESS)] == ["a"]
    assert [r.record_id for r in loader.records(RepairStatus.PENDING)] == ["b"]
    assert loader.records(RepairStatus.FAILED) == []


def test_schema_like_inputs_and_properties():
    from triad.collections.schema import Schema

    schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
    loader = SQLiteRepairLoader(
        ":memory:", input_schema=schema, output_schema=Schema("c:double")
    )
    assert loader.input_schema.equals(schema, check_metadata=True)
    assert loader.output_schema == Schema("c:double").pa_schema
    assert repr(loader) == "SQLiteRepairLoader(':memory:')"


def test_reject_invalid_schema_objects():
    for bad in [123, object(), "a:not_a_type"]:
        with pytest.raises(ValueError):
            SQLiteRepairLoader(":memory:", input_schema=bad, output_schema="a:str")


def test_cleaner_returns_invalid_types():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1"})
    result = loader.run("r1", lambda r: 42)
    assert result.status == RepairStatus.FAILED
    assert "output rows" in result.reason

    loader.correct("r1")
    result = loader.run("r1", lambda r: [42])
    assert result.status == RepairStatus.FAILED
    assert "mapping" in result.reason


def test_output_not_a_mapping_and_storage_error():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "1", "unit_price": "1"})

    class Weird:
        def keys(self):
            return ["name"]

        def __getitem__(self, k):
            return "x"

        def get(self, k, default=None):
            return "x"

    result = loader.run("r1", lambda r: Weird())
    assert result.status == RepairStatus.FAILED


def test_retry_reads_same_rows_when_corrected_without_changes():
    loader = _loader()
    loader.add("r1", {"name": "a", "amount": "10", "unit_price": "2"})
    loader.run("r1", lambda r: 1 / 0)
    assert loader.get_record("r1").status == RepairStatus.FAILED
    # correcting without new values keeps the snapshot and allows a retry
    loader.correct("r1")
    result = loader.run("r1", _cleaner)
    assert result.applied
    assert loader.get_record("r1").status == RepairStatus.SUCCESS
