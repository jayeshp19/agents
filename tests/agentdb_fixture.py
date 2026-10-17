"""The agentstore the storage tests run against, through LiveKit's public API.

A staging project from the environment: ``LIVEKIT_URL``, with ``LIVEKIT_API_KEY`` and
``LIVEKIT_API_SECRET``. Without ``LIVEKIT_URL`` and with ``LIVEKIT_LOCAL_AGENTSTORE=1`` the tests
run against ``local_agentstore.py`` instead, SQLite behind the same routes on a loopback port, with
a test key pair; with neither, every test skips.
``agentdb_storage`` is one ``AgentDBStorage``; ``agentdb_workers`` two of them with holders of
their own, which is two workers of one project; ``agentdb_worker`` the factory behind both.
Threads are minted by the threads service, never reused across tests.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest

from livekit.agents.a2a import pump
from livekit.agents.storage import AgentDBStorage

from .local_agentstore import LocalAgentStore

LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10"))
"""The lane lease's lifetime in seconds, as ``AgentDBStorage`` reads it from the environment."""
LEASE_MARGIN = 1.0
"""What a test sleeps past ``LEASE_TTL`` before it counts a lease as lapsed."""
PROJECT_URL = os.environ.get("LIVEKIT_URL", "")
"""The project's URL, read at import: a test module may remove ``LIVEKIT_URL`` so that its
servers address themselves locally, and the storage still needs it. Empty: the local store, when
``LOCAL_STORE`` asks for it."""
LOCAL_STORE = os.environ.get("LIVEKIT_LOCAL_AGENTSTORE", "").lower() in ("1", "true", "yes")
"""Opt-in to the local store when there is no project: several tests it would run still fail for
reasons of their own, so without it they skip as they do with no project."""
LOCAL_LATENCY = float(os.environ.get("LIVEKIT_LOCAL_AGENTSTORE_LATENCY", "0"))
"""Seconds the local store waits before each data plane answer, to stand in for the network."""
LOCAL_API_KEY = "APIlocalagentstore"
LOCAL_API_SECRET = "local-agentstore-secret-not-a-real-key"
"""The key pair of the local store, which does not verify tokens."""

_local: LocalAgentStore | None = None


def project_storage(holder: str = "") -> AgentDBStorage:
    """An ``AgentDBStorage`` on ``PROJECT_URL``, which it reads only while it is built; on the
    local store when there is no project and ``LOCAL_STORE`` is set."""
    with pytest.MonkeyPatch.context() as mp:
        if PROJECT_URL:
            mp.setenv("LIVEKIT_URL", PROJECT_URL)
            return AgentDBStorage(holder=holder)
        if _local is not None:
            mp.setenv("LIVEKIT_URL", _local.url)
            return AgentDBStorage(api_key=LOCAL_API_KEY, api_secret=LOCAL_API_SECRET, holder=holder)
        if LOCAL_STORE:
            raise ValueError("the local agentstore is not running")
        # no project and no local store: raises for the missing LIVEKIT_URL
        return AgentDBStorage(holder=holder)


def agentstore_configured() -> str | None:
    """``None`` when the environment names a project or the local store runs, else why the
    tests skip."""
    try:
        project_storage()
    except ValueError as e:
        return str(e)
    return None


async def new_session(storage: AgentDBStorage, agent_name: str) -> str:
    """A fresh thread id from the threads service, its database created with it."""
    return (await storage.threads.create_thread()).thread_id


async def let_the_lease_lapse() -> None:
    await asyncio.sleep(LEASE_TTL + LEASE_MARGIN)


@pytest.fixture(scope="session")
def local_agentstore(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalAgentStore | None]:
    """The local store for the session, started on first use; ``None`` when there is a project or
    ``LIVEKIT_LOCAL_AGENTSTORE`` is not set."""
    global _local
    if PROJECT_URL or not LOCAL_STORE:
        yield None
        return
    store = LocalAgentStore(Path(tmp_path_factory.mktemp("agentstore")), latency=LOCAL_LATENCY)
    try:
        store.start()
    except RuntimeError as e:
        pytest.skip(f"no agentstore project in the environment, and no local store: {e}")
    _local = store
    try:
        yield store
    finally:
        _local = None
        store.stop()


@pytest.fixture
async def agentdb_worker(
    local_agentstore: LocalAgentStore | None,
) -> AsyncIterator[Callable[[str], AgentDBStorage]]:
    """``agentdb_worker("worker-c")`` is one more worker against the same project."""
    if (why := agentstore_configured()) is not None:
        pytest.skip(
            f"no agentstore project in the environment: {why} "
            "(LIVEKIT_LOCAL_AGENTSTORE=1 runs the local store)"
        )

    def make(holder: str) -> AgentDBStorage:
        return project_storage(holder)

    yield make

    # an A2A response ends before its turn has closed the durable stream (a2a/pump.py); that
    # close is a request to agent-db now, so it is given its time before the leak check
    turns = [task for task in pump.running if not task.done()]
    if turns:
        await asyncio.wait(turns, timeout=5)


@pytest.fixture
def agentdb_storage(agentdb_worker: Callable[[str], AgentDBStorage]) -> AgentDBStorage:
    return agentdb_worker("worker-a")


@pytest.fixture
def agentdb_workers(
    agentdb_worker: Callable[[str], AgentDBStorage],
) -> tuple[AgentDBStorage, AgentDBStorage]:
    return agentdb_worker("worker-a"), agentdb_worker("worker-b")
