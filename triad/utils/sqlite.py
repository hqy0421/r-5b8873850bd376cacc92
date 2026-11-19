"""SQLite backed repair loading for correction driven cleaning queues.

A repair loader manages records that are pushed through a cleaning pipeline
while users are allowed to correct the raw record at any time. A correction
*retracts* the previously cleaned rows, bumps the record's revision, and
creates a new pending task. Every task runs against the raw snapshot captured
when it was submitted, so a slow task can never observe a newer correction.

The pipeline of a task is always the same:

1. :func:`~triad.utils.pyarrow.apply_schema` converts the raw snapshot into
   the declared input types
2. the user supplied cleaner turns each input row into ``0`` to ``N`` output
   rows
3. the output rows are converted into the declared output schema

Rows are loaded only if all three steps complete. A failure is recorded with
its reason and can never leave partial rows behind, while a success that
produces zero rows is still a success.

Only the first completion of the *current pending revision* can change the
record's status and rows. Completions of older revisions (stale) and repeated
completions (duplicate) stay on their own task row and never touch the current
state. All state lives in SQLite, so everything survives closing and reopening
the database.

.. admonition:: Examples

    .. code-block:: python

        from triad.utils.sqlite import SQLiteRepairLoader

        loader = SQLiteRepairLoader(
            "repair.db",
            input_schema="name:str,age:str",
            output_schema="name:str,age:long,score:double",
        )
        loader.add(1, {"name": "Alice", "age": "30"})
        loader.run(1, lambda row: [{"name": row["name"], "age": row["age"], "score": 1.0}])
        print(loader.read(1).to_pylist())

        loader.correct(1, {"name": "Alice", "age": "31"})  # revision 2, rows retracted
        loader.run(1, lambda row: [{"name": row["name"], "age": row["age"], "score": 2.0}])

        # a failed record can be retried on its next revision
        loader.correct(1)
        loader.run(1, lambda row: [{"name": row["name"], "age": row["age"], "score": 0.0}])

.. note::

    Only SQLite scalar field types are supported for now: ``null``, ``bool``,
    every integer and float width, ``string``, ``binary``, ``date32`` and
    ``timestamp``. Nested, decimal, ``date64`` and large types are rejected
    with ``NotImplementedError``.

    The row tables are for durability, not for querying: use
    :meth:`SQLiteRepairLoader.read` or :meth:`SQLiteRepairLoader.read_all` to
    materialize arrow tables in the declared output types.
"""

import base64
import datetime
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pyarrow as pa

from triad.collections.schema import Schema, SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.pyarrow import apply_schema

RecordId = Union[int, str]

_BYTES_KEY = "$bytes"
_DATE_KEY = "$date"
_DATETIME_KEY = "$datetime"
_FLOAT_KEY = "$float"


class RepairStatus:
    """Status values of records and tasks"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ALL = (PENDING, SUCCESS, FAILED)


class _RepairError(Exception):
    """Internal exception to abort a transaction"""


@dataclass(frozen=True)
class RepairRecord:
    """The current state of a record

    :param record_id: the record id
    :param revision: the current revision, starting from ``1``
    :param status: one of :class:`RepairStatus`
    :param rows: number of currently loaded rows
    :param reason: failure reason of the current revision, ``None`` otherwise
    :param raw: the raw snapshot of the current revision
    """

    record_id: RecordId
    revision: int
    status: str
    rows: int
    reason: Optional[str]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class RepairRevision:
    """A raw snapshot kept for a record

    :param revision: the revision number, starting from ``1``
    :param raw: the raw snapshot of that revision
    :param created_at: unix timestamp when the revision was recorded
    """

    revision: int
    raw: Dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class RepairTask:
    """A cleaning task bound to a revision of a record

    A task always carries the raw snapshot captured when it was submitted, so
    it is never affected by corrections made after that point.

    :param task_id: the task id
    :param record_id: the record id
    :param revision: the revision of the record this task was submitted for
    :param raw: the raw snapshot submitted with this task
    :param status: one of :class:`RepairStatus`, ``pending`` means it has not
      completed yet
    :param rows: number of rows loaded by the completion, ``0`` until success
    :param reason: failure reason, ``None`` when it succeeded or is pending
    :param applied: whether the completion changed the current record state and
      rows, always ``False`` for a pending task
    """

    task_id: int
    record_id: RecordId
    revision: int
    raw: Dict[str, Any]
    status: str
    rows: int = 0
    reason: Optional[str] = None
    applied: bool = False


@dataclass(frozen=True)
class RepairResult:
    """The outcome of :meth:`SQLiteRepairLoader.run`

    :param record_id: the record id
    :param revision: the revision this run belongs to
    :param status: ``success`` or ``failed``
    :param rows: number of rows loaded, always ``0`` for a failure
    :param reason: failure reason, ``None`` for a success
    :param applied: whether this run changed the current record state and rows
    :param stale: whether the task is for a revision older than the current one
    :param duplicate: whether this task had already completed before this run
    """

    record_id: RecordId
    revision: int
    status: str
    rows: int
    reason: Optional[str]
    applied: bool
    stale: bool
    duplicate: bool


def _encode_scalar(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return _encode_scalar(obj.item())
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            # NaN and infinities are not valid json, keep them exact
            return {_FLOAT_KEY: float.hex(obj)}
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return {_BYTES_KEY: base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, datetime.datetime):
        if obj != obj:  # NaT
            return None
        return {_DATETIME_KEY: obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {_DATE_KEY: obj.isoformat()}
    raise TypeError(f"{type(obj)} {obj!r} is not a SQLite scalar value")


def _decode_scalar(obj: Any) -> Any:  # noqa: C901
    if isinstance(obj, dict):
        if _BYTES_KEY in obj:
            return base64.b64decode(obj[_BYTES_KEY])
        if _DATETIME_KEY in obj:
            return datetime.datetime.fromisoformat(obj[_DATETIME_KEY])
        if _DATE_KEY in obj:
            return datetime.date.fromisoformat(obj[_DATE_KEY])
        if _FLOAT_KEY in obj:
            return float.fromhex(obj[_FLOAT_KEY])
        raise ValueError(f"invalid encoded value {obj}")
    if isinstance(obj, list):
        raise ValueError(f"{obj!r} is not a scalar value")
    return obj


def _encode_row(values: Iterable[Any]) -> str:
    return json.dumps([_encode_scalar(x) for x in values], ensure_ascii=False)


def _decode_row(payload: str) -> List[Any]:
    obj = json.loads(payload)
    if not isinstance(obj, list):
        raise ValueError(f"invalid row payload {payload}")
    return [_decode_scalar(x) for x in obj]


def _encode_snapshot(raw: Mapping[str, Any], names: Sequence[str]) -> str:
    return json.dumps(
        {name: _encode_scalar(raw[name]) for name in names}, ensure_ascii=False
    )


def _decode_snapshot(payload: str) -> Dict[str, Any]:
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid snapshot payload {payload}")
    return {str(k): _decode_scalar(v) for k, v in obj.items()}


def _record_key(record_id: Any) -> str:
    if isinstance(record_id, bool) or not isinstance(record_id, (int, str)):
        raise TypeError(f"record id must be int or str, got {type(record_id)}")
    kind = "i" if isinstance(record_id, int) else "s"
    return json.dumps([kind, record_id], ensure_ascii=False)


def _decode_record_key(key: str) -> RecordId:
    kind, value = json.loads(key)
    return int(value) if kind == "i" else value


def _assert_sqlite_scalar(field: pa.Field, schema_name: str) -> None:
    tp = field.type
    if tp == pa.null() or tp == pa.bool_() or tp == pa.string():
        return
    if tp == pa.binary() or tp == pa.date32():
        return
    if pa.types.is_integer(tp) or pa.types.is_floating(tp):
        return
    if pa.types.is_timestamp(tp):
        return
    raise NotImplementedError(
        f"{schema_name} field {field.name!r} has type {tp}, "
        "only SQLite scalar types are supported"
    )


def _normalize_schema(obj: Any, schema_name: str) -> pa.Schema:
    try:
        if isinstance(obj, pa.Schema):
            schema = obj
        elif isinstance(obj, Schema):
            schema = obj.pyarrow_schema
        else:
            schema = Schema(obj).pyarrow_schema
    except (SyntaxError, TypeError, SchemaError) as e:
        raise ValueError(f"{schema_name} is invalid: {e}") from e
    if len(schema) == 0:
        raise ValueError(f"{schema_name} can't be empty")
    names = [f.name for f in schema]
    if any(n == "" for n in names):
        raise ValueError(f"{schema_name} has empty field names: {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"{schema_name} has duplicated field names: {names}")
    for f in schema:
        _assert_sqlite_scalar(f, schema_name)
    return schema


def _failed_fields(schema: pa.Schema, values: List[Any]) -> List[str]:
    """Find which fields of a row failed the schema conversion"""
    failed: List[str] = []
    for i, field in enumerate(schema):
        try:
            list(apply_schema(pa.schema([field]), [[values[i]]]))
        except Exception:
            failed.append(field.name)
    return failed


def _iter_outputs(obj: Any) -> Iterator[Mapping[str, Any]]:
    if obj is None:
        return
    if isinstance(obj, Mapping):
        yield obj
        return
    try:
        it = iter(obj)
    except TypeError:
        raise TypeError(f"expect None or output rows, got {type(obj)}")
    for item in it:
        if not isinstance(item, Mapping):
            raise TypeError(f"expect output rows as mappings, got {type(item)}")
        yield item


_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS triad_repair_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS triad_repair_records (
        record_key TEXT PRIMARY KEY,
        revision INTEGER NOT NULL,
        status TEXT NOT NULL,
        reason TEXT,
        row_count INTEGER NOT NULL,
        raw TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS triad_repair_tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        raw TEXT NOT NULL,
        status TEXT NOT NULL,
        row_count INTEGER NOT NULL DEFAULT 0,
        reason TEXT,
        applied INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        completed_at REAL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS triad_repair_tasks_by_record
    ON triad_repair_tasks (record_key, revision)
    """,
    """
    CREATE TABLE IF NOT EXISTS triad_repair_rows (
        record_key TEXT NOT NULL,
        seq INTEGER NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (record_key, seq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS triad_repair_revisions (
        record_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        raw TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (record_key, revision)
    )
    """,
]


class SQLiteRepairLoader:
    """Repair loading of records that are corrected while being cleaned

    :param path: sqlite database path, ``":memory:"`` or ``None`` for an
      in-memory database
    :param input_schema: schema of the raw record snapshot, any schema like
      object accepted by :class:`~triad.collections.schema.Schema`
    :param output_schema: schema of the cleaned rows
    :param timeout: sqlite busy timeout in seconds, defaults to 30

    .. note::

        The current raw snapshot of every record is kept in the database, and
        every task keeps the snapshot it was submitted with, so corrections
        never erase the value a task has to work on.

    .. admonition:: Examples

        .. code-block:: python

            loader = SQLiteRepairLoader(
                ":memory:",
                input_schema="name:str,age:str",
                output_schema="name:str,age:long",
            )
            loader.add("c1", {"name": "a", "age": "1"})
            loader.run("c1", lambda row: [{"name": row["name"], "age": row["age"]}])
            assert loader.get_record("c1").status == "success"
            assert loader.read("c1").to_pylist() == [{"name": "a", "age": 1}]
    """

    _FORMAT_VERSION = "1"

    def __init__(
        self,
        path: Any,
        input_schema: Any,
        output_schema: Any,
        timeout: float = 30.0,
    ):
        self._input_schema = _normalize_schema(input_schema, "input_schema")
        self._output_schema = _normalize_schema(output_schema, "output_schema")
        self._input_names = [f.name for f in self._input_schema]
        self._output_names = [f.name for f in self._output_schema]
        self._input_set = set(self._input_names)
        self._output_set = set(self._output_names)
        self._lock = threading.RLock()
        self._closed = False
        self._path = ":memory:" if path is None else str(path)
        self._conn = sqlite3.connect(
            self._path,
            timeout=timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        self._init_db()

    @property
    def input_schema(self) -> pa.Schema:
        """Schema of the raw record snapshot"""
        return self._input_schema

    @property
    def output_schema(self) -> pa.Schema:
        """Schema of the cleaned rows"""
        return self._output_schema

    def add(self, record_id: RecordId, raw: Mapping[str, Any]) -> int:
        """Add a new record and mark it as pending

        :param record_id: integer or string record id
        :param raw: the raw record snapshot
        :raises InvalidOperationError: if the record already exists, use
          :meth:`correct` instead
        :return: the new revision, always ``1``
        """
        key = _record_key(record_id)
        payload = self._encode_snapshot(raw)
        now = time.time()
        with self._txn() as cur:
            if self._select_record(cur, key) is not None:
                raise InvalidOperationError(f"record {record_id!r} already exists")
            cur.execute(
                "INSERT INTO triad_repair_records "
                "(record_key, revision, status, reason, row_count, raw, updated_at) "
                "VALUES (?, ?, ?, NULL, 0, ?, ?)",
                (key, 1, RepairStatus.PENDING, payload, now),
            )
            cur.execute(
                "INSERT INTO triad_repair_revisions "
                "(record_key, revision, raw, created_at) VALUES (?, ?, ?, ?)",
                (key, 1, payload, now),
            )
        return 1

    def correct(
        self, record_id: RecordId, raw: Optional[Mapping[str, Any]] = None
    ) -> int:
        """Correct a record: keep its raw snapshot, bump the revision, mark it
        pending and retract all of its cleaned rows

        :param record_id: integer or string record id
        :param raw: the corrected raw snapshot, ``None`` keeps the current one,
          which is handy to retry a revision that failed
        :raises KeyError: if the record does not exist
        :return: the new revision
        """
        key = _record_key(record_id)
        now = time.time()
        with self._txn() as cur:
            record = self._select_record(cur, key)
            if record is None:
                raise KeyError(f"record {record_id!r} does not exist")
            payload = record["raw"] if raw is None else self._encode_snapshot(raw)
            revision = int(record["revision"]) + 1
            cur.execute("DELETE FROM triad_repair_rows WHERE record_key = ?", (key,))
            cur.execute(
                "UPDATE triad_repair_records SET revision = ?, status = ?, "
                "reason = NULL, row_count = 0, raw = ?, updated_at = ? "
                "WHERE record_key = ?",
                (revision, RepairStatus.PENDING, payload, now, key),
            )
            cur.execute(
                "INSERT OR REPLACE INTO triad_repair_revisions "
                "(record_key, revision, raw, created_at) VALUES (?, ?, ?, ?)",
                (key, revision, payload, now),
            )
        return revision

    def revisions(self, record_id: RecordId) -> List[RepairRevision]:
        """Get every raw snapshot kept for a record, oldest revision first

        :raises KeyError: if the record does not exist
        """
        key = _record_key(record_id)
        with self._lock:
            if self._select_record(self._cursor(), key) is None:
                raise KeyError(f"record {record_id!r} does not exist")
        rows = self._query(
            "SELECT revision, raw, created_at FROM triad_repair_revisions "
            "WHERE record_key = ? ORDER BY revision",
            (key,),
        )
        return [
            RepairRevision(
                revision=int(r["revision"]),
                raw=_decode_snapshot(r["raw"]),
                created_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def get_record(self, record_id: RecordId) -> RepairRecord:
        """Get the current state of a record

        :raises KeyError: if the record does not exist
        """
        key = _record_key(record_id)
        with self._lock:
            record = self._select_record(self._cursor(), key)
        if record is None:
            raise KeyError(f"record {record_id!r} does not exist")
        return self._to_record(record)

    def records(self, status: Optional[str] = None) -> List[RepairRecord]:
        """Get all current records, optionally filtered by status"""
        sql = "SELECT * FROM triad_repair_records"
        args: List[Any] = []
        if status is not None:
            self._assert_status(status)
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY record_key"
        rows = self._query(sql, args)
        return [self._to_record(r) for r in rows]

    def submit(self, record_id: RecordId) -> RepairTask:
        """Submit a cleaning task for the current revision of a record

        The raw snapshot is copied into the task, so later corrections can not
        change what this task will process.

        :raises KeyError: if the record does not exist
        """
        key = _record_key(record_id)
        with self._txn() as cur:
            record = self._select_record(cur, key)
            if record is None:
                raise KeyError(f"record {record_id!r} does not exist")
            created_at = time.time()
            cur.execute(
                "INSERT INTO triad_repair_tasks "
                "(record_key, revision, raw, status, row_count, reason, applied, "
                "created_at) VALUES (?, ?, ?, ?, 0, NULL, 0, ?)",
                (
                    key,
                    int(record["revision"]),
                    record["raw"],
                    RepairStatus.PENDING,
                    created_at,
                ),
            )
            task_id = int(cur.lastrowid)
        return RepairTask(
            task_id=task_id,
            record_id=record_id,
            revision=int(record["revision"]),
            raw=_decode_snapshot(record["raw"]),
            status=RepairStatus.PENDING,
        )

    def get_task(self, task_id: int) -> RepairTask:
        """Get a task by id

        :raises KeyError: if the task does not exist
        """
        with self._lock:
            row = self._select_task(self._cursor(), task_id)
        if row is None:
            raise KeyError(f"task {task_id} does not exist")
        return self._to_task(row)

    def tasks(
        self, record_id: Optional[RecordId] = None, status: Optional[str] = None
    ) -> List[RepairTask]:
        """Get tasks, optionally filtered by record and status"""
        sql = "SELECT * FROM triad_repair_tasks"
        cond: List[str] = []
        args: List[Any] = []
        if record_id is not None:
            cond.append("record_key = ?")
            args.append(_record_key(record_id))
        if status is not None:
            self._assert_status(status)
            cond.append("status = ?")
            args.append(status)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY task_id"
        rows = self._query(sql, args)
        return [self._to_task(r) for r in rows]

    def run(
        self,
        record_id: RecordId,
        cleaner: Callable[[Dict[str, Any]], Any],
    ) -> RepairResult:
        """Submit a task for the current revision and run it immediately

        :param record_id: integer or string record id
        :param cleaner: a function taking a dict of the input row and returning
          ``None``, a single output dict, or an iterable of output dicts
        """
        return self.run_task(self.submit(record_id), cleaner)

    def run_task(
        self,
        task: Union[RepairTask, int],
        cleaner: Callable[[Dict[str, Any]], Any],
    ) -> RepairResult:
        """Run a submitted task

        The task is processed against the snapshot it was submitted with:

        1. :func:`~triad.utils.pyarrow.apply_schema` normalizes the snapshot
        2. ``cleaner`` produces ``0`` to ``N`` output rows
        3. the outputs are converted into the output schema

        All rows are buffered and only loaded if every step succeeds. A
        failure records its reason and leaves no row behind, and a success
        with zero rows is a success.

        Only the first completion of the current pending revision changes the
        record status and rows. A stale task (older revision) and a repeated
        completion are recorded on the task itself and change nothing else.

        :param task: a task from :meth:`submit` or a task id
        :param cleaner: a function taking a dict of the input row and returning
          ``None``, a single output dict, or an iterable of output dicts
        :raises KeyError: if the task does not exist
        :return: the result of this run
        """
        if not callable(cleaner):
            raise TypeError(f"cleaner {cleaner!r} is not callable")
        task_id = task.task_id if isinstance(task, RepairTask) else int(task)
        with self._lock:
            row = self._select_task(self._cursor(), task_id)
        if row is None:
            raise KeyError(f"task {task_id} does not exist")
        stored = self._to_task(row)
        status, payloads, reason = self._pipeline(stored.raw, cleaner)
        return self._complete(stored, status, payloads, reason)

    def read(self, record_id: RecordId) -> pa.Table:
        """Read the current cleaned rows of a record as an arrow table

        The table has exactly the declared output schema and types; duplicate
        rows and nulls are preserved. A record with no current rows (pending,
        failed, or a successful cleanup that produced nothing) returns an empty
        table with that schema.

        :raises KeyError: if the record does not exist
        """
        key = _record_key(record_id)
        with self._lock:
            if self._select_record(self._cursor(), key) is None:
                raise KeyError(f"record {record_id!r} does not exist")
            payloads = self._select_payloads(self._cursor(), key)
        return self._to_table([_decode_row(p) for p in payloads])

    def read_all(self) -> pa.Table:
        """Read the current cleaned rows of all records as one arrow table

        An extra leading ``record_id`` string column identifies the record,
        followed by the output schema columns.
        """
        with self._lock:
            rows = self._query(
                "SELECT record_key, payload FROM triad_repair_rows "
                "ORDER BY record_key, seq"
            )
        ids: List[str] = []
        values: List[List[Any]] = []
        for row in rows:
            ids.append(str(_decode_record_key(row["record_key"])))
            values.append(_decode_row(row["payload"]))
        table = self._to_table(values)
        schema = pa.schema(
            [pa.field("record_id", pa.string())] + list(self._output_schema)
        )
        return pa.Table.from_arrays(
            [pa.array(ids, type=pa.string())]
            + [table.column(i) for i in range(table.num_columns)],
            schema=schema,
        )

    def close(self) -> None:
        """Close the underlying database connection"""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()

    def __enter__(self) -> "SQLiteRepairLoader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteRepairLoader({self._path!r})"

    def _init_db(self) -> None:
        with self._lock:
            for stmt in _SCHEMA_STATEMENTS:
                self._conn.execute(stmt)
            row = self._conn.execute(
                "SELECT value FROM triad_repair_meta WHERE key = ?",
                ("format_version",),
            ).fetchone()
            if row is None:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO triad_repair_meta (key, value) VALUES (?, ?)",
                        ("format_version", self._FORMAT_VERSION),
                    )
                    self._conn.execute(
                        "INSERT INTO triad_repair_meta (key, value) VALUES (?, ?)",
                        (
                            "input_schema",
                            self._input_schema.serialize().to_pybytes().hex(),
                        ),
                    )
                    self._conn.execute(
                        "INSERT INTO triad_repair_meta (key, value) VALUES (?, ?)",
                        (
                            "output_schema",
                            self._output_schema.serialize().to_pybytes().hex(),
                        ),
                    )
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
            else:
                if row["value"] != self._FORMAT_VERSION:
                    raise InvalidOperationError(
                        f"{self._path} uses format {row['value']}, "
                        f"expected {self._FORMAT_VERSION}"
                    )
                self._check_schema("input_schema", self._input_schema)
                self._check_schema("output_schema", self._output_schema)

    def _check_schema(self, name: str, schema: pa.Schema) -> None:
        row = self._conn.execute(
            "SELECT value FROM triad_repair_meta WHERE key = ?", (name,)
        ).fetchone()
        if row is None:  # pragma: no cover
            return
        stored = pa.ipc.read_schema(pa.py_buffer(bytes.fromhex(row["value"])))
        if not stored.equals(schema, check_metadata=True):
            raise InvalidOperationError(
                f"{name} of {self._path} is {stored}, got {schema}"
            )

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            self._assert_open()
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            else:
                cur.execute("COMMIT")
            finally:
                cur.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise InvalidOperationError(f"{self} is closed")

    def _cursor(self) -> sqlite3.Cursor:
        with self._lock:
            self._assert_open()
            return self._conn.cursor()

    def _query(self, sql: str, args: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._cursor().execute(sql, list(args)).fetchall()

    @staticmethod
    def _assert_status(status: str) -> None:
        if status not in RepairStatus.ALL:
            raise ValueError(f"invalid status {status!r}, expect {RepairStatus.ALL}")

    def _encode_snapshot(self, raw: Mapping[str, Any]) -> str:
        if not isinstance(raw, Mapping):
            raise TypeError(f"raw record must be a mapping, got {type(raw)}")
        extra = [str(k) for k in raw.keys() if k not in self._input_set]
        if extra:
            raise ValueError(
                f"raw record has fields not in input_schema: {sorted(extra)}"
            )
        values = {name: raw.get(name) for name in self._input_names}
        return _encode_snapshot(values, self._input_names)

    def _pipeline(
        self,
        raw: Dict[str, Any],
        cleaner: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[str, List[str], Optional[str]]:
        try:
            input_row = self._snapshot_row(raw)
            outputs = list(_iter_outputs(cleaner(input_row)))
            payloads = [self._encode_output(x) for x in outputs]
            return RepairStatus.SUCCESS, payloads, None
        except Exception as e:
            return RepairStatus.FAILED, [], f"{type(e).__name__}: {e}"

    def _snapshot_row(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        values = [raw.get(name) for name in self._input_names]
        converted = self._convert_row(self._input_schema, values)
        return dict(zip(self._input_names, converted))

    def _encode_output(self, output: Mapping[str, Any]) -> str:
        if not isinstance(output, Mapping):
            raise _RepairError(f"expect output rows as mappings, got {type(output)}")
        extra = [str(k) for k in output.keys() if k not in self._output_set]
        if extra:
            raise _RepairError(
                f"output row has fields not in output_schema: {sorted(extra)}"
            )
        values = [output.get(name) for name in self._output_names]
        converted = self._convert_row(self._output_schema, values)
        self._assert_storable(converted)
        try:
            return _encode_row(converted)
        except Exception as e:
            raise _RepairError(f"unable to store output row: {e}")

    def _assert_storable(self, values: List[Any]) -> None:
        # ``apply_schema`` does not check integer widths, so a value like 200 in
        # an ``int8`` column would be accepted here but break ``read`` later.
        # Building the row with arrow now keeps every loaded row readable.
        columns: List[pa.Array] = []
        for i, field in enumerate(self._output_schema):
            try:
                columns.append(pa.array([values[i]], type=field.type))
            except Exception as e:
                raise ValueError(
                    f"output field {field.name!r} is not valid for {field.type}: {e}"
                ) from e
        try:
            pa.Table.from_arrays(columns, schema=self._output_schema)
        except Exception as e:
            raise ValueError(f"output row is not valid for the output schema: {e}")

    @staticmethod
    def _convert_row(schema: pa.Schema, values: List[Any]) -> List[Any]:
        try:
            return list(apply_schema(schema, [values]))[0]
        except Exception as e:
            fields = _failed_fields(schema, values)
            if fields:
                raise ValueError(f"{e} (fields: {', '.join(fields)})") from e
            raise

    def _complete(
        self,
        task: RepairTask,
        status: str,
        payloads: List[str],
        reason: Optional[str],
    ) -> RepairResult:
        key = _record_key(task.record_id)
        with self._txn() as cur:
            current = self._select_task(cur, task.task_id)
            if current is None:  # pragma: no cover
                raise KeyError(f"task {task.task_id} does not exist")
            duplicate = current["status"] != RepairStatus.PENDING
            stale = False
            record = self._select_record(cur, key)
            if record is None:  # pragma: no cover
                stale = True
            elif int(record["revision"]) != task.revision:
                stale = True
            applies = (
                not duplicate
                and not stale
                and record is not None
                and record["status"] == RepairStatus.PENDING
            )
            if not duplicate:
                applied = applies
                cur.execute(
                    "UPDATE triad_repair_tasks SET status = ?, row_count = ?, "
                    "reason = ?, applied = ?, completed_at = ? WHERE task_id = ?",
                    (
                        status,
                        len(payloads),
                        reason,
                        int(applied),
                        time.time(),
                        task.task_id,
                    ),
                )
            else:
                applied = False
            if applies and record is not None:
                cur.execute(
                    "DELETE FROM triad_repair_rows WHERE record_key = ?", (key,)
                )
                cur.executemany(
                    "INSERT INTO triad_repair_rows (record_key, seq, payload) "
                    "VALUES (?, ?, ?)",
                    [(key, i, p) for i, p in enumerate(payloads)],
                )
                cur.execute(
                    "UPDATE triad_repair_records SET status = ?, reason = ?, "
                    "row_count = ?, updated_at = ? WHERE record_key = ?",
                    (status, reason, len(payloads), time.time(), key),
                )
        return RepairResult(
            record_id=task.record_id,
            revision=task.revision,
            status=status,
            rows=len(payloads),
            reason=reason,
            applied=applied,
            stale=stale,
            duplicate=duplicate,
        )

    def _to_record(self, row: sqlite3.Row) -> RepairRecord:
        return RepairRecord(
            record_id=_decode_record_key(row["record_key"]),
            revision=int(row["revision"]),
            status=row["status"],
            rows=int(row["row_count"]),
            reason=row["reason"],
            raw=_decode_snapshot(row["raw"]),
        )

    def _to_task(self, row: sqlite3.Row) -> RepairTask:
        return RepairTask(
            task_id=int(row["task_id"]),
            record_id=_decode_record_key(row["record_key"]),
            revision=int(row["revision"]),
            raw=_decode_snapshot(row["raw"]),
            status=row["status"],
            rows=int(row["row_count"]),
            reason=row["reason"],
            applied=bool(row["applied"]),
        )

    def _to_table(self, values: List[List[Any]]) -> pa.Table:
        columns = []
        for i, field in enumerate(self._output_schema):
            try:
                columns.append(pa.array([x[i] for x in values], type=field.type))
            except Exception as e:
                raise ValueError(f"unable to build column {field.name!r}: {e}") from e
        return pa.Table.from_arrays(columns, schema=self._output_schema)

    @staticmethod
    def _select_record(cur: sqlite3.Cursor, key: str) -> Optional[sqlite3.Row]:
        return cur.execute(
            "SELECT * FROM triad_repair_records WHERE record_key = ?", (key,)
        ).fetchone()

    @staticmethod
    def _select_task(cur: sqlite3.Cursor, task_id: int) -> Optional[sqlite3.Row]:
        return cur.execute(
            "SELECT * FROM triad_repair_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    @staticmethod
    def _select_payloads(cur: sqlite3.Cursor, key: str) -> List[str]:
        rows = cur.execute(
            "SELECT payload FROM triad_repair_rows WHERE record_key = ? ORDER BY seq",
            (key,),
        ).fetchall()
        return [r["payload"] for r in rows]
