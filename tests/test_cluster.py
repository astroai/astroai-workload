"""Unit tests for cluster lifecycle + state store + canfar ops (moved from ray/manager)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy/optional deps so tests run without a live canfar client or Ray.
_canfar = types.ModuleType("canfar")
_canfar_models = types.ModuleType("canfar.models")
_canfar_config = types.ModuleType("canfar.models.config")
_canfar_sessions = types.ModuleType("canfar.sessions")


class _Configuration:
    def __init__(self) -> None:
        self.active = MagicMock(authentication=None, server=None)
        self.registry = MagicMock(username=None, secret=None, url=None)

    def get_credential(self, _idp: str) -> None:
        raise KeyError(_idp)


class _Session:
    def __init__(self) -> None:
        self.config = MagicMock(registry=MagicMock(username=None, secret=None))

    def fetch(self, **_kwargs):
        return []

    def create(self, **_kwargs):
        return []

    def info(self, *_a, **_k):
        return []

    def logs(self, *_a, **_k):
        return {}

    def destroy(self, *_a, **_k):
        return {}


_canfar_config.Configuration = _Configuration
_canfar_sessions.Session = _Session
sys.modules.setdefault("canfar", _canfar)
sys.modules.setdefault("canfar.models", _canfar_models)
sys.modules.setdefault("canfar.models.config", _canfar_config)
sys.modules.setdefault("canfar.sessions", _canfar_sessions)
sys.modules.setdefault("ray", MagicMock(__version__="2.56.1"))

from astroai_workload.canfar_ops import (  # noqa: E402
    CanfarOps,
    parse_probe_logs,
)
from astroai_workload.cluster import (  # noqa: E402
    ClusterCreateRequest,
    clean_orphaned_workers,
    gc_terminal_cluster_workers,
    stop_cluster,
    validate_cluster_create,
)
from astroai_workload.reconcile import (  # noqa: E402
    _apply_canfar_phase,
    _refresh_cluster_phase,
    reconcile_cluster,
)
from astroai_workload.settings import ManagerSettings  # noqa: E402
from astroai_workload.state_store import (  # noqa: E402
    ClusterState,
    StateStore,
    WorkerRecord,
)
from astroai_workload.workers import (  # noqa: E402
    build_worker_env,
    destroy_all_workers,
    destroy_worker,
)


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StateStore:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("RAY_CLUSTER_ID", "testcid")
    s = StateStore(cluster_id="testcid")
    s.ensure_dir()
    return s


@pytest.fixture()
def settings() -> ManagerSettings:
    return ManagerSettings(
        cluster_id="testcid",
        worker_image="images.canfar.net/astroai/ray-worker:local",
        probe_image="images.canfar.net/astroai/ray-worker:local",
        ray_version="2.56.1",
        scratch_dir="/scratch",
        ray_head_port=6379,
        heartbeat_timeout_seconds=120,
        worker_launch_timeout_seconds=30,
        preflight_timeout_seconds=60,
    )


def _state(**kwargs) -> ClusterState:
    kwargs.setdefault("cluster_id", "testcid")
    kwargs.setdefault("manager_ip", "10.0.0.1")
    kwargs.setdefault("ray_address", "10.0.0.1:6379")
    return ClusterState(**kwargs)


def _auth_ok(canfar: MagicMock) -> None:
    status = MagicMock()
    status.authenticated = True
    status.message = None
    canfar.auth_status.return_value = status


# ===============================================================
# canfar_ops
# ===============================================================
class TestParseProbeLogs:
    def test_pass(self) -> None:
        logs = (
            "WORKER_IP=10.0.0.5\n"
            "PROBE worker->manager:6379 PASS\n"
            "PROBE worker->manager:6380 PASS\n"
            "PROBE_RESULT PASS\n"
        )
        result = parse_probe_logs(logs)
        assert result["worker_ip"] == "10.0.0.5"
        assert result["result"] == "PASS"
        assert len(result["checks"]) == 2

    def test_fail(self) -> None:
        result = parse_probe_logs("PROBE_RESULT FAIL\n")
        assert result["result"] == "FAIL"

    def test_empty(self) -> None:
        result = parse_probe_logs("")
        assert result["worker_ip"] is None
        assert result["result"] == "UNKNOWN"


class TestCanfarOpsCreateHeadless:
    def test_success_single_replica(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops = CanfarOps()
        with (
            patch("astroai_workload.canfar_ops._registry_configured", return_value=True),
            patch("astroai_workload.canfar_ops._registry_env", return_value={}),
            patch.object(ops, "_fresh_session") as mock_new,
        ):
            mock_sess = MagicMock()
            mock_sess.create.return_value = ["sid-1"]
            mock_new.return_value = mock_sess
            results = ops.create_headless(
                name="ray-w-test", image="img", cores=2, ram=8, gpu=1, replicas=1
            )
            assert len(results) == 1
            assert results[0].session_id == "sid-1"
            assert results[0].name == "ray-w-test"

    def test_multiple_replicas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops = CanfarOps()
        with (
            patch("astroai_workload.canfar_ops._registry_configured", return_value=True),
            patch("astroai_workload.canfar_ops._registry_env", return_value={}),
            patch.object(ops, "_fresh_session") as mock_new,
        ):
            mock_sess = MagicMock()
            mock_sess.create.return_value = ["sid-1", "sid-2", "sid-3"]
            mock_new.return_value = mock_sess
            results = ops.create_headless(name="ray-w-test", image="img", replicas=3)
            assert len(results) == 3
            assert results[1].name == "ray-w-test-2"

    def test_no_registry_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops = CanfarOps()
        with (
            patch("astroai_workload.canfar_ops._registry_configured", return_value=False),
            pytest.raises(RuntimeError, match="Harbor registry credentials"),
        ):
            ops.create_headless(name="x", image="img")

    def test_create_empty_recovers_via_name_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops = CanfarOps()
        with (
            patch("astroai_workload.canfar_ops._registry_configured", return_value=True),
            patch("astroai_workload.canfar_ops._registry_env", return_value={}),
            patch.object(ops, "_fresh_session") as mock_new,
            patch.object(ops, "_resolve_ids_by_name", return_value=["recovered-sid"]) as probe,
        ):
            mock_sess = MagicMock()
            mock_sess.create.return_value = []
            mock_new.return_value = mock_sess
            results = ops.create_headless(name="ray-w-test", image="img")
            assert results[0].session_id == "recovered-sid"
            probe.assert_called_once_with(name="ray-w-test", replicas=1)


# ===============================================================
# state / workers
# ===============================================================
class TestBuildWorkerEnv:
    def test_basic_env(self, settings: ManagerSettings, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("astroai_workload.workers.manager_pod_ip", lambda: "10.0.0.1")
        env = build_worker_env(settings, "/arc/home/u/heartbeat")
        assert env["RAY_CLUSTER_ID"] == "testcid"
        assert env["RAY_HEAD_IP"] == "10.0.0.1"
        assert env["RAY_VERSION_EXPECTED"] == "2.56.1"
        assert env["RAY_SPILL_DIR"] == "/scratch/ray/testcid"
        assert env["RAY_MANAGER_HEARTBEAT_TIMEOUT_SECONDS"] == "120"


class TestDestroyWorker:
    def test_destroys_and_updates_phase(
        self, store: StateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        canfar = MagicMock()
        canfar.destroy.return_value = True
        store.save(
            _state(
                phase="Running",
                workers=[WorkerRecord(session_id="w1", name="ray-w-1", phase="Ray Healthy")],
            )
        )
        with patch("astroai_workload.workers.archive_session_logs"):
            result = destroy_worker(canfar=canfar, store=store, session_id="w1")
        assert result["destroyed"] is True
        state = store.load()
        assert state is not None
        assert state.workers[0].phase == "Stopping"


class TestDestroyAllWorkers:
    def test_skips_stopped_by_default(self, store: StateStore) -> None:
        canfar = MagicMock()
        canfar.destroy.return_value = True
        store.save(
            _state(
                phase="Running",
                workers=[
                    WorkerRecord(session_id="active", name="w-active", phase="Ray Healthy"),
                    WorkerRecord(session_id="done", name="w-done", phase="Stopped"),
                ],
            )
        )
        with (
            patch("astroai_workload.workers.archive_session_logs"),
            patch("astroai_workload.workers.destroy_worker", wraps=destroy_worker),
        ):
            results = destroy_all_workers(canfar=canfar, store=store)
        destroyed_ids = {r["session_id"] for r in results}
        assert "active" in destroyed_ids
        assert "done" not in destroyed_ids


# ===============================================================
# reconcile
# ===============================================================
class TestApplyCanfarPhase:
    def test_pending(self) -> None:
        w = WorkerRecord(session_id="w1", name="w", phase="Requested", canfar_status="Pending")
        _apply_canfar_phase(w)
        assert w.phase == "CANFAR Pending"

    def test_failed(self) -> None:
        w = WorkerRecord(session_id="w1", name="w", phase="Requested", canfar_status="Failed")
        _apply_canfar_phase(w)
        assert w.phase == "CANFAR Failed"


class TestRefreshClusterPhase:
    def test_all_joined_running(self) -> None:
        from datetime import UTC, datetime, timedelta

        since = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
        state = _state(
            phase="Creating",
            worker_count=2,
            min_joined=2,
            setup_ready=True,
            setup_ready_since=since,
            workers=[
                WorkerRecord(session_id="w1", name="w1", phase="Ray Healthy", ray_joined=True),
                WorkerRecord(session_id="w2", name="w2", phase="Ray Healthy", ray_joined=True),
            ],
        )
        _refresh_cluster_phase(state)
        assert state.phase == "Running"

    def test_partial_degraded(self) -> None:
        state = _state(
            phase="Creating",
            worker_count=3,
            min_joined=1,
            workers=[
                WorkerRecord(session_id="w1", name="w1", phase="Ray Healthy", ray_joined=True),
                WorkerRecord(session_id="w2", name="w2", phase="Ray Joining"),
                WorkerRecord(session_id="w3", name="w3", phase="CANFAR Failed"),
            ],
        )
        _refresh_cluster_phase(state)
        assert state.phase == "Degraded"

    def test_stopping_all_terminal_becomes_stopped(self) -> None:
        state = _state(
            phase="Stopping",
            worker_count=2,
            workers=[
                WorkerRecord(session_id="w1", name="w1", phase="Stopped"),
                WorkerRecord(session_id="w2", name="w2", phase="Orphaned"),
            ],
        )
        _refresh_cluster_phase(state)
        assert state.phase == "Stopped"


class TestReconcileCluster:
    def test_marks_workers_joined(self, store: StateStore, monkeypatch: pytest.MonkeyPatch) -> None:
        canfar = MagicMock()
        _auth_ok(canfar)
        canfar.session_info.return_value = {"status": "Running"}
        store.save(
            _state(
                phase="Creating",
                workers=[
                    WorkerRecord(
                        session_id="w1", name="w1", phase="CANFAR Pending", worker_ip="10.0.0.5"
                    ),
                ],
            )
        )
        monkeypatch.setattr("astroai_workload.reconcile.manager_pod_ip", lambda: "10.0.0.1")
        monkeypatch.setattr("astroai_workload.reconcile.ray_address", lambda: "10.0.0.1:6379")
        monkeypatch.setattr("astroai_workload.reconcile.list_ray_nodes", lambda *a, **k: [])
        monkeypatch.setattr(
            "astroai_workload.reconcile.live_worker_node_ips", lambda *a, **k: {"10.0.0.5"}
        )
        monkeypatch.setattr(
            "astroai_workload.reconcile.node_ip_to_id", lambda *a, **k: {"10.0.0.5": "node-1"}
        )
        with patch("astroai_workload.reconcile.archive_session_logs"):
            result = reconcile_cluster(canfar=canfar, store=store)
        assert result is not None
        w = result.workers[0]
        assert w.ray_joined is True
        assert w.ray_node_id == "node-1"
        assert w.phase == "Ray Healthy"


# ===============================================================
# cluster lifecycle
# ===============================================================
class TestValidateClusterCreate:
    def test_rejects_no_auth(self, store: StateStore) -> None:
        canfar = MagicMock()
        status = MagicMock()
        status.authenticated = False
        status.message = "not authed"
        canfar.auth_status.return_value = status
        req = ClusterCreateRequest(name="x")
        with pytest.raises(RuntimeError, match="not authed"):
            validate_cluster_create(canfar=canfar, store=store, req=req)

    def test_rejects_active_cluster(self, store: StateStore) -> None:
        canfar = MagicMock()
        _auth_ok(canfar)
        store.save(_state(phase="Running"))
        req = ClusterCreateRequest(name="x", require_preflight=False)
        with pytest.raises(RuntimeError, match="already active"):
            validate_cluster_create(canfar=canfar, store=store, req=req)

    def test_rejects_no_preflight(self, store: StateStore) -> None:
        canfar = MagicMock()
        _auth_ok(canfar)
        req = ClusterCreateRequest(name="x", require_preflight=True)
        with pytest.raises(RuntimeError, match="preflight"):
            validate_cluster_create(canfar=canfar, store=store, req=req)

    def test_accepts_idle_without_preflight_when_optional(self, store: StateStore) -> None:
        canfar = MagicMock()
        _auth_ok(canfar)
        req = ClusterCreateRequest(name="x", require_preflight=False)
        validate_cluster_create(canfar=canfar, store=store, req=req)


class TestStopCluster:
    def test_full_stop_flow(
        self, store: StateStore, settings: ManagerSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        canfar = MagicMock()
        canfar.destroy.return_value = True
        canfar.session_info.return_value = {}
        canfar.session_logs.return_value = ""
        store.save(
            _state(
                phase="Running",
                worker_count=2,
                workers=[
                    WorkerRecord(
                        session_id="w1", name="w1", phase="Ray Healthy", canfar_status="Running"
                    ),
                    WorkerRecord(
                        session_id="w2", name="w2", phase="Ray Healthy", canfar_status="Running"
                    ),
                ],
            )
        )
        with (
            patch("astroai_workload.cluster.archive_session_logs"),
            patch("astroai_workload.cluster.reconcile_cluster") as mock_reconcile,
        ):

            def reconcile_side(canfar=None, store=None, state=None, nodes=None):
                s = store.load() if store else state
                if s:
                    for w in s.workers:
                        w.phase = "Stopped"
                    s.phase = "Stopping"
                    store.save(s)
                return s

            mock_reconcile.side_effect = reconcile_side
            result = stop_cluster(canfar=canfar, store=store)

        assert result is not None
        assert result.phase == "Stopped"
        assert all(w.phase == "Stopped" for w in result.workers)

    def test_stop_empty_state(self, store: StateStore) -> None:
        canfar = MagicMock()
        assert stop_cluster(canfar=canfar, store=store) is None


class TestCleanOrphans:
    def test_destroys_preflight_when_idle(
        self, store: StateStore, settings: ManagerSettings
    ) -> None:
        canfar = MagicMock()
        store.save(_state(phase="Failed", workers=[]))

        def list_sessions(name_prefix: str):
            if name_prefix.startswith("ray-preflight"):
                return [{"id": "pf1", "name": f"ray-preflight-{settings.cluster_id}-x"}]
            return []

        canfar.list_headless_sessions.side_effect = list_sessions
        canfar.destroy.return_value = True
        destroyed = clean_orphaned_workers(settings=settings, canfar=canfar, store=store)
        assert any(d["session_id"] == "pf1" for d in destroyed)

    def test_destroys_tracked_when_terminal(
        self, store: StateStore, settings: ManagerSettings
    ) -> None:
        canfar = MagicMock()
        canfar.list_headless_sessions.return_value = []
        canfar.destroy.return_value = True
        store.save(
            _state(
                phase="Stopped",
                workers=[
                    WorkerRecord(
                        session_id="ghost",
                        name="ray-w-testcid-ghost",
                        phase="Stopped",
                        canfar_status="Running",
                    )
                ],
            )
        )
        destroyed = clean_orphaned_workers(settings=settings, canfar=canfar, store=store)
        assert any(d.get("session_id") == "ghost" for d in destroyed)


class TestGcTerminalClusterWorkers:
    def test_handles_no_saved_state(self, store: StateStore, settings: ManagerSettings) -> None:
        canfar = MagicMock()
        canfar.list_headless_sessions.return_value = []
        canfar.destroy.return_value = True
        with patch("astroai_workload.cluster.reconcile_cluster", return_value=None):
            result = gc_terminal_cluster_workers(settings=settings, canfar=canfar, store=store)
        assert result is None
