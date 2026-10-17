"""A local agentstore for the storage tests: SQLite behind the public API's agentstore routes.

``agentdb_fixture.py`` points ``AgentDBStorage`` here when the environment names no project and
sets ``LIVEKIT_LOCAL_AGENTSTORE=1``. It serves, on a loopback port, what the Rust client
(``lk-agents-storage``) dials:

* ``/v0/agentstore/db``: the data plane, ``AgentDB.Wire`` over a WebSocket. Every statement is
  the SQL the client sent, run unchanged on one SQLite file per database: ``Exec``, ``Query``,
  ``Batch`` (one transaction, rolled back whole on an error), ``Ping``. A test double, not agent-db:
  it reimplements no lease, fence, waiter or scheduling rule, since those are the client's SQL. It
  needs SQLite 3.45 for ``jsonb``.
* ``/v0/projects/p_/threads`` and ``/v0/projects/p_/databases``: the threads service and the
  database lifecycle, in memory; a thread's database is created with it.
* ``/v0/agentstore/streams``: the answer streams, in memory, catch-up and ``live=sse``.

Where it is known to differ from agent-db: tokens are not verified; every frame of a WebSocket runs
on that socket's one SQLite connection, so a ``Query`` always reads the latest commit (agent-db
answers a ``Query`` from a read pool); a connection is never severed; ``Credit`` is not enforced;
thread and database TTLs are not enforced; interactive transactions (``Begin``/``Commit``/
``Rollback``) are refused, since the client sends none. ``statements`` keeps every statement it
ran, for tests that count them.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

MIN_SQLITE = (3, 45, 0)
"""``jsonb()``, which the thread schema and the client's statements use."""


# ---------------------------------------------------------------------------------------------
# the wire: livekit/protocol `protobufs/livekit_agentdb.proto`, message `AgentDB.Wire`
# ---------------------------------------------------------------------------------------------

# Field numbers and types as the .proto declares them; built here because the protocol's Python
# package does not ship this file. (name, number, type, oneof); a leading "*" is repeated.
_WIRE: dict[str, list[tuple[str, int, str, str | None]]] = {
    "ClientMessage": [
        ("request_id", 1, "uint32", None),
        ("hello", 2, "Hello", "message"),
        ("exec", 3, "Statement", "message"),
        ("query", 4, "Statement", "message"),
        ("batch", 5, "Batch", "message"),
        ("begin", 6, "Begin", "message"),
        ("commit", 7, "Commit", "message"),
        ("rollback", 8, "Rollback", "message"),
        ("cancel", 9, "Cancel", "message"),
        ("credit", 10, "Credit", "message"),
        ("ping", 11, "Ping", "message"),
    ],
    "ServerMessage": [
        ("request_id", 1, "uint32", None),
        ("hello_ok", 2, "HelloOk", "message"),
        ("columns", 3, "Columns", "message"),
        ("column_batch", 4, "ColumnBatch", "message"),
        ("exec_result", 5, "ExecResult", "message"),
        ("done", 6, "Done", "message"),
        ("error", 7, "Error", "message"),
        ("pong", 8, "Pong", "message"),
    ],
    "Value": [
        ("null_value", 1, "bool", "value"),
        ("int_value", 2, "int64", "value"),
        ("double_value", 3, "double", "value"),
        ("text_value", 4, "string", "value"),
        ("blob_value", 5, "bytes", "value"),
    ],
    "Statement": [
        ("sql", 1, "string", None),
        ("params", 2, "*Value", None),
        ("lang", 3, "QueryLang", None),
    ],
    "Batch": [("statements", 1, "*Statement", None)],
    "Begin": [],
    "Commit": [],
    "Rollback": [],
    "Cancel": [],
    "Credit": [("batches", 1, "uint32", None)],
    "Ping": [("timestamp_ms", 1, "int64", None)],
    "Pong": [("last_ping_timestamp_ms", 1, "int64", None), ("timestamp_ms", 2, "int64", None)],
    "Hello": [("token", 1, "string", None), ("database_id", 2, "string", None)],
    "HelloOk": [
        ("tip", 1, "int64", None),
        ("ping_interval_ms", 2, "uint32", None),
        ("ping_timeout_ms", 3, "uint32", None),
    ],
    "Columns": [("names", 1, "*string", None), ("statement", 2, "uint32", None)],
    "Column": [
        ("types", 1, "bytes", None),
        ("ints", 2, "*sint64", None),
        ("doubles", 3, "*double", None),
        ("text_data", 4, "bytes", None),
        ("text_ends", 5, "*uint32", None),
        ("blob_data", 6, "bytes", None),
        ("blob_ends", 7, "*uint32", None),
    ],
    "ColumnBatch": [
        ("columns", 1, "*Column", None),
        ("rows", 2, "uint32", None),
        ("statement", 3, "uint32", None),
    ],
    "ExecResult": [
        ("rows_affected", 1, "int64", None),
        ("last_insert_id", 2, "int64", None),
        ("tip", 3, "int64", None),
        ("statement", 4, "uint32", None),
    ],
    "Done": [("tip", 1, "int64", None), ("total_rows", 2, "uint64", None)],
    "Error": [("code", 1, "string", None), ("message", 2, "string", None)],
}
_ENUMS = {"QueryLang": ["SQL"], "ValueType": ["NULL", "INT", "DOUBLE", "TEXT", "BLOB"]}
NULL, INT, DOUBLE, TEXT, BLOB = range(5)
"""``ValueType``: the tag of one cell of a ``Column``."""


def _wire_classes() -> dict[str, Any]:
    F = descriptor_pb2.FieldDescriptorProto
    scalars = {
        "bool": F.TYPE_BOOL,
        "int64": F.TYPE_INT64,
        "uint32": F.TYPE_UINT32,
        "uint64": F.TYPE_UINT64,
        "sint64": F.TYPE_SINT64,
        "double": F.TYPE_DOUBLE,
        "string": F.TYPE_STRING,
        "bytes": F.TYPE_BYTES,
    }
    file = descriptor_pb2.FileDescriptorProto(
        name="local_agentstore/livekit_agentdb.proto", package="livekit", syntax="proto3"
    )
    wire = file.message_type.add(name="AgentDB").nested_type.add(name="Wire")
    for name, values in _ENUMS.items():
        enum = wire.enum_type.add(name=name)
        for number, value in enumerate(values):
            enum.value.add(name=value, number=number)
    for name, fields in _WIRE.items():
        message = wire.nested_type.add(name=name)
        oneofs: list[str] = []
        for fname, number, ftype, oneof in fields:
            repeated = ftype.startswith("*")
            ftype = ftype.lstrip("*")
            f = message.field.add(name=fname, number=number)
            f.label = F.LABEL_REPEATED if repeated else F.LABEL_OPTIONAL
            if ftype in scalars:
                f.type = scalars[ftype]
            else:
                f.type = F.TYPE_ENUM if ftype in _ENUMS else F.TYPE_MESSAGE
                f.type_name = f".livekit.AgentDB.Wire.{ftype}"
            if oneof is not None:
                if oneof not in oneofs:
                    oneofs.append(oneof)
                    message.oneof_decl.add(name=oneof)
                f.oneof_index = oneofs.index(oneof)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return {
        name: message_factory.GetMessageClass(
            pool.FindMessageTypeByName(f"livekit.AgentDB.Wire.{name}")
        )
        for name in _WIRE
    }


W = _wire_classes()


def _param(value: Any) -> Any:
    kind = value.WhichOneof("value")
    return None if kind in (None, "null_value") else getattr(value, kind)


def _column(cells: list[Any]) -> Any:
    """One ``Column`` of a ``ColumnBatch``, each cell tagged with its SQLite storage class."""
    col = W["Column"]()
    tags = bytearray()
    for cell in cells:
        if cell is None:
            tags.append(NULL)
        elif isinstance(cell, int):
            tags.append(INT)
            col.ints.append(cell)
        elif isinstance(cell, float):
            tags.append(DOUBLE)
            col.doubles.append(cell)
        elif isinstance(cell, bytes):
            tags.append(BLOB)
            col.blob_data += cell
            col.blob_ends.append(len(col.blob_data))
        else:
            tags.append(TEXT)
            col.text_data += str(cell).encode()
            col.text_ends.append(len(col.text_data))
    col.types = bytes(tags)
    return col


# ---------------------------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------------------------


@dataclass
class Statement:
    """One statement the data plane ran: what a test counts renewals or writes by."""

    at: float
    """``time.monotonic()`` when it ran."""
    conn: int
    """The WebSocket it came on, numbered from 1."""
    frame: str
    """``exec``, ``query`` or ``batch``."""
    database_id: str
    sql: str
    params: list[Any]
    rows: int
    """Rows it returned (``RETURNING`` or a ``SELECT``)."""
    error: str | None = None


@dataclass
class _Stream:
    records: list[bytes] = field(default_factory=list)
    closed: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def notify(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()


def _rfc3339(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds(duration: str | None) -> int:
    """A protobuf JSON ``Duration`` (``"3600s"``) in whole seconds."""
    return int(float(duration.removesuffix("s"))) if duration else 0


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"code": code, "message": message}, status=status)


class LocalAgentStore:
    """The server, on its own thread and event loop so the test's loop (and virtual time) never
    paces it. ``start()`` returns the base URL, the ``LIVEKIT_URL`` the storage is built with."""

    SSE_ROTATE = 5.0
    """How long a ``live=sse`` response waits for the next record before it ends, as agent-db
    rotates the connection; the client reconnects from its offset."""

    def __init__(self, directory: Path, *, latency: float = 0.0) -> None:
        self.directory = directory
        self.latency = latency
        """Seconds the data plane waits before it answers each frame, a network round trip's
        stand-in; 0 answers at once."""
        self.statements: list[Statement] = []
        """Every statement the data plane ran, in order; also ``statements.jsonl`` in
        ``directory``."""
        self.threads: dict[str, dict[str, Any]] = {}
        self.databases: dict[str, dict[str, Any]] = {}
        self._tips: dict[str, int] = {}
        self._streams: dict[str, _Stream] = {}
        self._conns = 0
        self._sockets: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self.url = ""

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self) -> str:
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise RuntimeError(
                f"the local agentstore needs SQLite {'.'.join(map(str, MIN_SQLITE))} for jsonb; "
                f"this Python has {sqlite3.sqlite_version}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        ready = threading.Event()
        self._loop = asyncio.new_event_loop()

        def serve() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve(sock))
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=serve, name="local-agentstore", daemon=True)
        self._thread.start()
        ready.wait(10)
        return self.url

    async def _serve(self, sock: socket.socket) -> None:
        app = web.Application()
        app.router.add_get("/v0/agentstore/db", self._ws)
        app.router.add_post("/v0/agentstore/streams", self._create_stream)
        app.router.add_post("/v0/agentstore/streams/{id}", self._append)
        app.router.add_get("/v0/agentstore/streams/{id}", self._read)
        app.router.add_delete("/v0/agentstore/streams/{id}", self._delete_stream)
        app.router.add_post("/v0/projects/{project}/threads", self._create_thread)
        app.router.add_get("/v0/projects/{project}/threads", self._list_threads)
        app.router.add_get("/v0/projects/{project}/threads/{id}", self._get_thread)
        app.router.add_patch("/v0/projects/{project}/threads/{id}", self._update_thread)
        app.router.add_delete("/v0/projects/{project}/threads/{id}", self._delete_thread)
        app.router.add_post("/v0/projects/{project}/databases", self._create_database_route)
        app.router.add_get("/v0/projects/{project}/databases/{id}", self._get_database)
        app.router.add_delete("/v0/projects/{project}/databases/{id}", self._delete_database)
        self._runner = web.AppRunner(app, handle_signals=False, shutdown_timeout=1.0)
        await self._runner.setup()
        await web.SockSite(self._runner, sock).start()

    def stop(self) -> None:
        if self._loop is None or self._runner is None:
            return
        runner, loop = self._runner, self._loop

        async def shutdown() -> None:
            # the client parks idle connections for the thread's next turn
            await asyncio.gather(*(ws.close() for ws in list(self._sockets)))
            await runner.cleanup()

        asyncio.run_coroutine_threadsafe(shutdown(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(10)
        loop.close()
        self._loop = None

    # -- reading it back ------------------------------------------------------------------------

    def database_path(self, database_id: str) -> Path:
        return self.directory / f"{database_id}.sqlite"

    @contextlib.contextmanager
    def read(self, database_id: str) -> Iterator[sqlite3.Connection]:
        """A read-only connection to a database, for a test to look at its rows directly."""
        db = sqlite3.connect(f"file:{self.database_path(database_id)}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    # -- databases and threads ------------------------------------------------------------------

    def _create_database(self, ttl_seconds: int) -> dict[str, Any]:
        database_id = "DB_" + uuid.uuid4().hex[:12]
        now = time.time()
        self.databases[database_id] = {
            "databaseId": database_id,
            "region": "local",
            "createdAt": _rfc3339(now),
            **({"expiresAt": _rfc3339(now + ttl_seconds)} if ttl_seconds else {}),
        }
        self._tips[database_id] = 0
        return self.databases[database_id]

    def _drop_database(self, database_id: str) -> None:
        self.databases.pop(database_id, None)
        self._tips.pop(database_id, None)
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path(database_id)}{suffix}").unlink(missing_ok=True)

    async def _create_database_route(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        info = self._create_database(_seconds(body.get("ttl")))
        return web.json_response({k: info[k] for k in ("databaseId", "expiresAt") if k in info})

    async def _get_database(self, request: web.Request) -> web.Response:
        database_id = request.match_info["id"]
        if database_id not in self.databases:
            return _error(404, "not_found", f"database {database_id} not found")
        return web.json_response(
            {"database": {**self.databases[database_id], "tip": str(self._tips[database_id])}}
        )

    async def _delete_database(self, request: web.Request) -> web.Response:
        database_id = request.match_info["id"]
        if database_id not in self.databases:
            return _error(404, "not_found", f"database {database_id} not found")
        self._drop_database(database_id)
        return web.json_response({})

    def _touch(self, thread: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        thread["lastActive"] = _rfc3339(now)
        ttl = _seconds(thread.get("idleTtl"))
        if ttl:
            thread["expiresAt"] = _rfc3339(now + ttl)
        else:
            thread.pop("expiresAt", None)
        return thread

    async def _create_thread(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        thread_id = "AT_" + uuid.uuid4().hex[:12]
        database = self._create_database(0)
        thread = {
            "threadId": thread_id,
            "scope": body.get("scope", ""),
            "attributes": dict(body.get("attributes") or {}),
            "createdAt": database["createdAt"],
            "databaseId": database["databaseId"],
            **({"idleTtl": body["idleTtl"]} if body.get("idleTtl") else {}),
        }
        self.threads[thread_id] = self._touch(thread)
        return web.json_response({"thread": thread})

    def _find_thread(self, request: web.Request) -> dict[str, Any] | None:
        return self.threads.get(request.match_info["id"])

    async def _get_thread(self, request: web.Request) -> web.Response:
        if (thread := self._find_thread(request)) is None:
            return _error(404, "not_found", f"thread {request.match_info['id']} not found")
        return web.json_response({"thread": thread})

    async def _update_thread(self, request: web.Request) -> web.Response:
        if (thread := self._find_thread(request)) is None:
            return _error(404, "not_found", f"thread {request.match_info['id']} not found")
        body = await request.json() if request.can_read_body else {}
        for key, value in (body.get("attributes") or {}).items():
            if value:
                thread["attributes"][key] = value
            else:
                thread["attributes"].pop(key, None)
        if body.get("idleTtl"):
            thread["idleTtl"] = body["idleTtl"]
        return web.json_response({"thread": self._touch(thread)})

    async def _delete_thread(self, request: web.Request) -> web.Response:
        thread = self.threads.pop(request.match_info["id"], None)
        if thread is None:
            return _error(404, "not_found", f"thread {request.match_info['id']} not found")
        self._drop_database(thread["databaseId"])
        return web.json_response({})

    async def _list_threads(self, request: web.Request) -> web.Response:
        scope = request.query.get("scope")
        threads = [t for t in self.threads.values() if not scope or t["scope"] == scope]
        return web.json_response({"items": threads[::-1], "pageInfo": {}})

    # -- streams --------------------------------------------------------------------------------

    async def _create_stream(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        stream_id = "AST_" + uuid.uuid4().hex[:12]
        self._streams[stream_id] = _Stream()
        ttl = int(body.get("ttl_seconds") or 86_400)
        return web.json_response({"stream_id": stream_id, "expires_at": int(time.time()) + ttl})

    def _stream(self, request: web.Request) -> _Stream | None:
        return self._streams.get(request.match_info["id"])

    async def _append(self, request: web.Request) -> web.Response:
        if (stream := self._stream(request)) is None:
            return _error(404, "not_found", f"stream {request.match_info['id']} not found")
        body = await request.read()
        if body:
            if stream.closed:
                return _error(409, "stream_closed", "the stream is closed")
            stream.records.append(body)
        if request.headers.get("Stream-Closed", "").lower() == "true":
            stream.closed = True
        stream.notify()
        return web.Response(headers={"Stream-Offset": str(len(stream.records))})

    async def _delete_stream(self, request: web.Request) -> web.Response:
        if self._streams.pop(request.match_info["id"], None) is None:
            return _error(404, "not_found", f"stream {request.match_info['id']} not found")
        return web.Response()

    async def _read(self, request: web.Request) -> web.StreamResponse:
        if (stream := self._stream(request)) is None:
            return _error(404, "not_found", f"stream {request.match_info['id']} not found")
        offset = request.query.get("offset", "-1")
        start = 0 if offset == "-1" else int(offset)
        if request.query.get("live") != "sse":
            return web.Response(
                body=b"[" + b",".join(stream.records[start:]) + b"]",
                content_type="application/json",
                headers={
                    "Stream-Next-Offset": str(len(stream.records)),
                    "Stream-Up-To-Date": "true",
                    "Stream-Closed": str(stream.closed).lower(),
                },
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        while True:
            if start < len(stream.records):
                data = b"[" + b",".join(stream.records[start:]) + b"]"
                await response.write(b"event: data\ndata: " + data + b"\n\n")
                start = len(stream.records)
            control = {"streamNextOffset": str(start), "upToDate": True}
            if stream.closed:
                control["streamClosed"] = True
            await response.write(f"event: control\ndata: {json.dumps(control)}\n\n".encode())
            if stream.closed:
                break
            try:
                await asyncio.wait_for(stream.changed.wait(), self.SSE_ROTATE)
            except asyncio.TimeoutError:
                break  # rotation
        await response.write_eof()
        return response

    # -- the data plane -------------------------------------------------------------------------

    def _log(self, record: Statement) -> None:
        line = dataclasses.asdict(record)
        line["params"] = [
            f"<blob {len(p)}>" if isinstance(p, bytes) else p[:200] if isinstance(p, str) else p
            for p in record.params
        ]
        with (self.directory / "statements.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")

    def _run(
        self,
        db: sqlite3.Connection,
        stmt: Any,
        index: int,
        tip: int,
        conn: int,
        frame: str,
        database_id: str,
    ) -> list[tuple[str, Any]]:
        """One statement, answered the way agent-db answers it: ``Columns`` then one
        ``ColumnBatch`` for a statement that returns rows (``Columns`` alone when it returned
        none), an ``ExecResult`` for one that cannot; an ``Exec`` is always an ``ExecResult``."""
        params = [_param(p) for p in stmt.params]
        record = Statement(time.monotonic(), conn, frame, database_id, stmt.sql, params, 0)
        self.statements.append(record)
        try:
            cursor = db.execute(stmt.sql, params)
            rows = cursor.fetchall() if cursor.description else []
        except sqlite3.Error as e:
            record.error = str(e)
            self._log(record)
            raise
        record.rows = len(rows)
        self._log(record)
        if cursor.description and frame != "exec":
            names = [c[0] for c in cursor.description]
            frames: list[tuple[str, Any]] = [
                ("columns", W["Columns"](names=names, statement=index))
            ]
            if rows:
                columns = [_column([row[i] for row in rows]) for i in range(len(names))]
                frames.append(
                    (
                        "column_batch",
                        W["ColumnBatch"](columns=columns, rows=len(rows), statement=index),
                    )
                )
            return frames
        return [
            (
                "exec_result",
                W["ExecResult"](
                    rows_affected=max(0, cursor.rowcount),
                    last_insert_id=cursor.lastrowid or 0,
                    tip=tip,
                    statement=index,
                ),
            )
        ]

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(timeout=1.0)
        await ws.prepare(request)
        self._sockets.add(ws)
        self._conns += 1
        conn = self._conns
        db: sqlite3.Connection | None = None
        database_id = ""
        try:
            async for incoming in ws:
                if incoming.type != WSMsgType.BINARY:
                    continue
                msg = W["ClientMessage"].FromString(incoming.data)
                kind = msg.WhichOneof("message")
                frames: list[tuple[str, Any]] = []
                try:
                    if kind == "hello":
                        database_id = msg.hello.database_id
                        if database_id not in self.databases:
                            frames = [
                                (
                                    "error",
                                    W["Error"](
                                        code="not_found",
                                        message=f"database {database_id} not found",
                                    ),
                                )
                            ]
                        else:
                            db = sqlite3.connect(
                                self.database_path(database_id), isolation_level=None
                            )
                            db.execute("PRAGMA journal_mode=WAL")
                            frames = [("hello_ok", W["HelloOk"](tip=self._tips[database_id]))]
                    elif db is None:
                        frames = [("error", W["Error"](code="protocol", message="Hello first"))]
                    elif kind in ("query", "exec"):
                        # a statement that wrote (a `Query` may: `UPDATE .. RETURNING`) commits
                        changes = db.total_changes
                        tip = self._tips[database_id] + 1
                        frames = self._run(db, getattr(msg, kind), 0, tip, conn, kind, database_id)
                        if db.total_changes != changes:
                            self._tips[database_id] = tip
                        tip = self._tips[database_id]
                        if kind == "query":
                            frames.append(("done", W["Done"](tip=tip)))
                        else:
                            frames[0][1].tip = tip
                    elif kind == "batch":
                        tip = self._tips[database_id] + 1
                        db.execute("BEGIN IMMEDIATE")
                        try:
                            for index, stmt in enumerate(msg.batch.statements):
                                frames.extend(
                                    self._run(db, stmt, index, tip, conn, "batch", database_id)
                                )
                            db.execute("COMMIT")
                        except BaseException:
                            db.execute("ROLLBACK")
                            raise
                        self._tips[database_id] = tip
                        frames.append(("done", W["Done"](tip=tip)))
                    elif kind == "ping":
                        frames = [
                            (
                                "pong",
                                W["Pong"](
                                    last_ping_timestamp_ms=msg.ping.timestamp_ms,
                                    timestamp_ms=int(time.time() * 1000),
                                ),
                            )
                        ]
                    elif kind in ("credit", "cancel"):
                        continue  # every result is sent at once; there is nothing to pace
                    else:
                        frames = [
                            (
                                "error",
                                W["Error"](
                                    code="unimplemented",
                                    message=f"the local agentstore does not serve {kind}",
                                ),
                            )
                        ]
                except sqlite3.Error as e:
                    frames = [("error", W["Error"](code="sql_error", message=str(e)))]
                if self.latency:
                    await asyncio.sleep(self.latency)
                for name, payload in frames:
                    response = W["ServerMessage"](request_id=msg.request_id)
                    getattr(response, name).CopyFrom(payload)
                    await ws.send_bytes(response.SerializeToString())
        finally:
            self._sockets.discard(ws)
            if db is not None:
                db.close()
        return ws
