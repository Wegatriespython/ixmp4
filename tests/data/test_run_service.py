import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import sqlalchemy as sa

from ixmp4.base_exceptions import Forbidden
from ixmp4.data.run.exceptions import NoDefaultRunVersion, RunIsLocked, RunNotFound
from ixmp4.data.run.service import RunService
from ixmp4.transport import DirectTransport, Session, Transport
from tests import auth, backends
from tests.data.base import ServiceTest

transport = backends.get_transport_fixture(scope="class")


class RunServiceTest(ServiceTest[RunService]):
    service_class = RunService


class TestRunCreate(RunServiceTest):
    def test_run_create(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        run = service.create("Model", "Scenario")
        assert run.model.name == "Model"
        assert run.scenario.name == "Scenario"
        assert run.version == 1
        assert not run.is_default

    def test_run_create_versioning(
        self, versioning_service: RunService, fake_time: datetime.datetime
    ) -> None:
        expected_versions = pd.DataFrame(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    False,
                    fake_time.replace(tzinfo=None),
                    "@unknown",
                    pd.NaT,
                    None,
                    None,
                    None,
                    3,
                    None,
                    0,
                ],
            ],
            columns=[
                "id",
                "model__id",
                "scenario__id",
                "version",
                "is_default",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
                "lock_transaction",
                "lock_token",
                "transaction_id",
                "end_transaction_id",
                "operation_type",
            ],
        )

        vdf = versioning_service.versions.tabulate()
        pdt.assert_frame_equal(expected_versions, vdf, check_like=True)

    def test_create_run_increment_version(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        inc_run = service.create("Model", "Scenario")
        assert inc_run.model.name == "Model"
        assert inc_run.scenario.name == "Scenario"
        assert inc_run.version == 2
        assert not inc_run.is_default

    def test_create_run_increment_version_versioning(
        self, versioning_service: RunService, fake_time: datetime.datetime
    ) -> None:
        expected_versions = pd.DataFrame(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    False,
                    fake_time.replace(tzinfo=None),
                    "@unknown",
                    pd.NaT,
                    None,
                    None,
                    None,
                    3,
                    None,
                    0,
                ],
                [
                    2,
                    1,
                    1,
                    2,
                    False,
                    fake_time.replace(tzinfo=None),
                    "@unknown",
                    pd.NaT,
                    None,
                    None,
                    None,
                    4,
                    None,
                    0,
                ],
            ],
            columns=[
                "id",
                "model__id",
                "scenario__id",
                "version",
                "is_default",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
                "lock_transaction",
                "lock_token",
                "transaction_id",
                "end_transaction_id",
                "operation_type",
            ],
        )

        vdf = versioning_service.versions.tabulate()
        pdt.assert_frame_equal(expected_versions, vdf, check_like=True)


class TestRunGetRunVersions(RunServiceTest):
    def test_run_versions(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        run1 = service.create("Model", "Scenario")
        run2 = service.create("Model", "Scenario")
        service.set_as_default_version(run2.id)
        run3 = service.create("Model", "Scenario")

        assert run1 == service.get("Model", "Scenario", 1)

        assert run2.id == service.get("Model", "Scenario", 2).id
        assert run2.id == service.get_default_version("Model", "Scenario").id

        assert run3 == service.get("Model", "Scenario", 3)

    def test_run_versions_versioning(
        self, versioning_service: RunService, fake_time: datetime.datetime
    ) -> None:
        expected_versions = pd.DataFrame(
            [
                [1, 1, 1, 1, False, None, None, 3, None, 0],
                [2, 1, 1, 2, False, None, None, 4, 6, 0],
                [2, 1, 1, 2, True, None, None, 6, None, 1],
                [3, 1, 1, 3, False, None, None, 7, None, 0],
            ],
            columns=[
                "id",
                "model__id",
                "scenario__id",
                "version",
                "is_default",
                "lock_transaction",
                "lock_token",
                "transaction_id",
                "end_transaction_id",
                "operation_type",
            ],
        )

        vdf = versioning_service.versions.tabulate(
            columns=expected_versions.columns.to_list()
        )
        pdt.assert_frame_equal(expected_versions, vdf, check_like=True)


class TestRunGetRunNoDefaultVersion(RunServiceTest):
    def test_run_no_default_version(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        service.create("Model", "Scenario")
        with pytest.raises(NoDefaultRunVersion):
            service.get_default_version("Model", "Scenario")


class TestRunGetOrCreate(RunServiceTest):
    def test_run_get_or_create(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        run1 = service.create("Model", "Scenario")
        run2 = service.get_or_create("Model", "Scenario")

        assert run1 != run2
        assert run2.version == 2

        service.set_as_default_version(run1.id)

        run3 = service.get_or_create("Model", "Scenario")

        # is_default has changed, so the exact equality check will fail
        assert run1.id == run3.id


class TestRunGetById(RunServiceTest):
    def test_run_get_by_id(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        expected = service.create("Model", "Scenario")
        result = service.get_by_id(expected.id)
        assert expected == result


class TestRunNotFound(RunServiceTest):
    def test_run_not_found(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        with pytest.raises(RunNotFound):
            service.get_by_id(1)
        with pytest.raises(RunNotFound):
            service.lock(1)
        with pytest.raises(RunNotFound):
            service.unlock(1)


class TestRunList(RunServiceTest):
    def test_run_list(self, service: RunService, fake_time: datetime.datetime) -> None:
        run = service.create("Model", "Scenario")
        service.set_as_default_version(run.id)

        service.create("Model", "Scenario")
        service.create("Other Model", "Other Scenario")

        runs = service.list(default_only=False)

        assert runs[0].id == 1
        assert runs[0].model.name == "Model"
        assert runs[0].scenario.name == "Scenario"
        assert runs[0].version == 1
        assert runs[0].lock_transaction is None
        assert runs[0].is_default
        assert runs[0].created_by == "@unknown"
        assert runs[0].created_at == fake_time.replace(tzinfo=None)
        assert runs[0].updated_by == "@unknown"
        assert runs[0].updated_at == fake_time.replace(tzinfo=None)

        assert runs[1].id == 2
        assert runs[1].model.name == "Model"
        assert runs[1].scenario.name == "Scenario"
        assert runs[1].version == 2
        assert runs[1].lock_transaction is None
        assert not runs[1].is_default
        assert runs[1].created_by == "@unknown"
        assert runs[1].created_at == fake_time.replace(tzinfo=None)
        assert runs[1].updated_by is None
        assert runs[1].updated_at is None

        assert runs[2].id == 3
        assert runs[2].model.name == "Other Model"
        assert runs[2].scenario.name == "Other Scenario"
        assert runs[2].version == 1
        assert runs[2].lock_transaction is None
        assert not runs[2].is_default
        assert runs[2].created_by == "@unknown"
        assert runs[2].created_at == fake_time.replace(tzinfo=None)
        assert runs[2].updated_by is None
        assert runs[2].updated_at is None


class TestRunTabulate(RunServiceTest):
    def test_run_tabulate(
        self, service: RunService, fake_time: datetime.datetime
    ) -> None:
        run = service.create("Model", "Scenario")
        service.set_as_default_version(run.id)

        service.create("Model", "Scenario")
        service.create("Other Model", "Other Scenario")

        expected_runs = pd.DataFrame(
            [
                [
                    1,
                    "Model",
                    "Scenario",
                    1,
                    None,
                    True,
                    "@unknown",
                    fake_time.replace(tzinfo=None),
                    "@unknown",
                    fake_time.replace(tzinfo=None),
                ],
                [
                    2,
                    "Model",
                    "Scenario",
                    2,
                    None,
                    False,
                    "@unknown",
                    fake_time.replace(tzinfo=None),
                    None,
                    pd.NaT,
                ],
                [
                    3,
                    "Other Model",
                    "Other Scenario",
                    1,
                    None,
                    False,
                    "@unknown",
                    fake_time.replace(tzinfo=None),
                    None,
                    pd.NaT,
                ],
            ],
            columns=[
                "id",
                "model",
                "scenario",
                "version",
                "lock_transaction",
                "is_default",
                "created_by",
                "created_at",
                "updated_by",
                "updated_at",
            ],
        )

        runs = service.tabulate(
            include_audit_info=True,
            include_internal_columns=True,
            default_only=False,
        ).drop(columns=["model__id", "scenario__id"])
        pdt.assert_frame_equal(expected_runs, runs, check_like=True)


class TestRunLock(RunServiceTest):
    def test_run_lock(self, service: RunService) -> None:
        run = service.create("Model", "Scenario")
        locked = service.lock(run.id)
        assert locked.lock_transaction is not None

        with pytest.raises(RunIsLocked):
            service.lock(run.id)
        assert service.get_by_id(run.id).lock_token == locked.lock_token

    @pytest.mark.parametrize("with_token", [True, False])
    def test_run_unlock(self, service: RunService, with_token: bool) -> None:
        run = service.create("Model", "Scenario")
        locked = service.lock(run.id)
        lock_token = locked.lock_token if with_token else None

        unlocked = service.unlock(run.id, lock_token=lock_token)
        assert unlocked.lock_transaction is None
        assert unlocked.lock_token is None
        # unlocking an unlocked run is a no-op
        assert service.unlock(run.id, lock_token=lock_token).lock_transaction is None

    def test_run_unlock_stale_token(self, service: RunService) -> None:
        run = service.create("Model", "Scenario")
        stale = service.lock(run.id).lock_token
        service.unlock(run.id, lock_token=stale)
        current = service.lock(run.id).lock_token

        # a late or repeated unlock by the previous holder
        with pytest.raises(RunIsLocked):
            service.unlock(run.id, lock_token=stale)
        assert service.get_by_id(run.id).lock_token == current

    def test_run_lock_concurrent(self, transport: Transport, tmp_path: Path) -> None:
        """Of several callers on their own connections, exactly one acquires
        the lock in each round."""
        if not isinstance(transport, DirectTransport):
            self.skip_transport(transport, "cannot open further connections")
        engine = racing_engine(transport, tmp_path)
        transports = [
            DirectTransport(Session(bind=engine), check_alembic_version=False)
            for _ in range(6)
        ]
        services = [RunService(t) for t in transports]
        try:
            run = services[0].create("Model", "Scenario")
            # on SQLite the first lock creates the only transaction row, which
            # must not itself be raced
            services[0].unlock(services[0].lock(run.id).id)
            holders = []
            for _ in range(10):
                holders.append(race_for_lock(services, run.id))
                services[0].unlock(run.id)
        finally:
            for t in transports:
                t.close()

        assert holders == [1] * 10


def racing_engine(transport: DirectTransport, tmp_path: Path) -> sa.Engine:
    """An engine for the transport's database that gives each session its own
    connection, unlike the transport's own engine."""
    assert transport.session.bind is not None
    url = transport.session.bind.engine.url
    if url.get_backend_name() != "sqlite":
        return sa.create_engine(url, max_identifier_length=63)

    # an in-memory SQLite database is private to its connection
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'race.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    backends.create_model_tables(engine)
    return engine


def race_for_lock(services: list[RunService], run_id: int) -> int:
    """Let all services call `lock` at once and return how many acquired it."""
    start = threading.Barrier(len(services), timeout=60)

    def racer(service: RunService) -> bool:
        start.wait()
        try:
            service.lock(run_id)
        except RunIsLocked:
            return False
        return True

    with ThreadPoolExecutor(len(services)) as pool:
        return sum(pool.map(racer, services))


class TestRunAuthSarahPrivate(auth.SarahTest, auth.PrivatePlatformTest, RunServiceTest):
    def test_run_create(self, service: RunService) -> None:
        run = service.create("Model", "Scenario")
        assert run.id == 1
        assert run.version == 1
        assert run.created_by == "superuser_sarah"

    def test_run_get(self, service: RunService) -> None:
        run = service.get("Model", "Scenario", 1)
        assert run.id == 1

    def test_run_set_as_default_version(self, service: RunService) -> None:
        service.set_as_default_version(1)

    def test_run_get_default_version(self, service: RunService) -> None:
        run = service.get_default_version("Model", "Scenario")
        assert run.id == 1

    def test_run_unset_as_default_version(self, service: RunService) -> None:
        service.unset_as_default_version(1)

    def test_run_get_by_id(self, service: RunService) -> None:
        run = service.get_by_id(1)
        assert run.id == 1

    def test_run_lock(self, service: RunService) -> None:
        service.lock(1)

    def test_run_unlock(self, service: RunService) -> None:
        service.unlock(1)

    def test_run_revert(self, versioning_service: RunService) -> None:
        versioning_service.revert(1, 1)

    def test_run_list(self, service: RunService) -> None:
        results = service.list(default_only=False)
        assert len(results) == 1

    def test_run_tabulate(self, service: RunService) -> None:
        results = service.tabulate(default_only=False)
        assert len(results) == 1

    def test_run_delete(self, service: RunService) -> None:
        service.delete_by_id(1)


class TestRunAuthAlicePrivate(auth.AliceTest, auth.PrivatePlatformTest, RunServiceTest):
    def test_run_create(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.create("Model", "Scenario")
        run = unauthorized_service.create("Model", "Scenario")
        assert run.id == 1

    def test_run_get(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get("Model", "Scenario", 1)

    def test_run_set_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.set_as_default_version(1)
        unauthorized_service.set_as_default_version(1)

    def test_run_get_default_version(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get_default_version("Model", "Scenario")

    def test_run_unset_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.unset_as_default_version(1)
        unauthorized_service.unset_as_default_version(1)

    def test_run_get_by_id(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get_by_id(1)

    def test_run_lock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.lock(1)
        unauthorized_service.lock(1)

    def test_run_unlock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.unlock(1)
        unauthorized_service.unlock(1)

    def test_run_revert(self, versioning_service: RunService) -> None:
        with pytest.raises(RunNotFound):
            versioning_service.revert(1, 12)

    def test_run_list(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.list(default_only=False)

    def test_run_tabulate(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.tabulate(default_only=False)

    def test_run_delete(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.delete_by_id(1)


class TestRunAuthBobPrivate(auth.BobTest, auth.PrivatePlatformTest, RunServiceTest):
    def test_run_create(self, service: RunService) -> None:
        run = service.create("Model", "Scenario")
        assert run.id == 1
        assert run.version == 1
        assert run.created_by == "staffuser_bob"

    def test_run_get(self, service: RunService) -> None:
        run = service.get("Model", "Scenario", 1)
        assert run.id == 1

    def test_run_set_as_default_version(self, service: RunService) -> None:
        service.set_as_default_version(1)

    def test_run_get_default_version(self, service: RunService) -> None:
        run = service.get_default_version("Model", "Scenario")
        assert run.id == 1

    def test_run_unset_as_default_version(self, service: RunService) -> None:
        service.unset_as_default_version(1)

    def test_run_get_by_id(self, service: RunService) -> None:
        run = service.get_by_id(1)
        assert run.id == 1

    def test_run_lock(self, service: RunService) -> None:
        service.lock(1)

    def test_run_unlock(self, service: RunService) -> None:
        service.unlock(1)

    def test_run_revert(self, versioning_service: RunService) -> None:
        versioning_service.revert(1, 1)

    def test_run_list(self, service: RunService) -> None:
        results = service.list(default_only=False)
        assert len(results) == 1

    def test_run_tabulate(self, service: RunService) -> None:
        results = service.tabulate(default_only=False)
        assert len(results) == 1

    def test_run_delete(self, service: RunService) -> None:
        service.delete_by_id(1)


class TestRunAuthCarinaPrivate(
    auth.CarinaTest, auth.PrivatePlatformTest, RunServiceTest
):
    def test_run_create(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.create("Model", "Scenario")

        run1 = unauthorized_service.create("Model", "Scenario")
        run2 = unauthorized_service.create("Other Model", "Scenario")

        run3 = service.create("Model 10", "Scenario")

        assert run1.id == 1
        assert run2.id == 2
        assert run3.id == 3

    def test_run_get(self, service: RunService) -> None:
        run1 = service.get("Model", "Scenario", 1)
        assert run1.id == 1

        with pytest.raises(Forbidden):
            service.get("Other Model", "Scenario", 1)

        run3 = service.get("Model 10", "Scenario", 1)
        assert run3.id == 3

    def test_run_set_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.set_as_default_version(1)
        unauthorized_service.set_as_default_version(1)

        service.set_as_default_version(3)

    def test_run_get_default_version(self, service: RunService) -> None:
        run1 = service.get_default_version("Model", "Scenario")
        assert run1.id == 1

        with pytest.raises(Forbidden):
            service.get_default_version("Other Model", "Scenario")

        run3 = service.get_default_version("Model 10", "Scenario")
        assert run3.id == 3

    def test_run_unset_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.unset_as_default_version(1)
        unauthorized_service.unset_as_default_version(1)

        service.unset_as_default_version(3)

    def test_run_get_by_id(self, service: RunService) -> None:
        run1 = service.get_by_id(1)
        assert run1.id == 1

        with pytest.raises(RunNotFound):
            service.get_by_id(2)

        run3 = service.get_by_id(3)
        assert run3.id == 3

    def test_run_lock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.lock(1)
        unauthorized_service.lock(1)

        service.lock(3)

    def test_run_unlock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.unlock(1)
        unauthorized_service.unlock(1)

        service.unlock(3)

    def test_run_revert(self, versioning_service: RunService) -> None:
        with pytest.raises(Forbidden):
            versioning_service.revert(1, 1)

        with pytest.raises(RunNotFound):
            versioning_service.revert(2, 1)

        versioning_service.revert(3, 1)

    def test_run_list(self, service: RunService) -> None:
        results = service.list(default_only=False)
        assert len(results) == 2

    def test_run_tabulate(self, service: RunService) -> None:
        results = service.tabulate(default_only=False)
        assert len(results) == 2

    def test_run_delete(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.delete_by_id(1)

        with pytest.raises(Forbidden):
            service.delete_by_id(2)

        with pytest.raises(Forbidden):
            service.delete_by_id(3)


class TestRunAuthNonePrivate(auth.NoneTest, auth.PrivatePlatformTest, RunServiceTest):
    def test_run_create(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(Forbidden):
            service.create("Model", "Scenario")
        unauthorized_service.create("Model", "Scenario")

    def test_run_get(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get("Model", "Scenario", 1)

    def test_run_set_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.set_as_default_version(1)
        unauthorized_service.set_as_default_version(1)

    def test_run_get_default_version(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get_default_version("Model", "Scenario")

    def test_run_unset_as_default_version(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.unset_as_default_version(1)
        unauthorized_service.unset_as_default_version(1)

    def test_run_get_by_id(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.get_by_id(1)

    def test_run_lock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.lock(1)
        unauthorized_service.lock(1)

    def test_run_unlock(
        self, service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            service.unlock(1)
        unauthorized_service.unlock(1)

    def test_run_revert(
        self, versioning_service: RunService, unauthorized_service: RunService
    ) -> None:
        with pytest.raises(RunNotFound):
            versioning_service.revert(1, 1)
        unauthorized_service.revert(1, 1)

    def test_run_list(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.list(default_only=False)

    def test_run_tabulate(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.tabulate(default_only=False)

    def test_run_delete(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.delete_by_id(1)


class TestRunAuthDaveGated(auth.DaveTest, auth.GatedPlatformTest, RunServiceTest):
    def test_run_create(self, service: RunService) -> None:
        run = service.create("Model", "Scenario")
        assert run.id == 1
        assert run.version == 1
        assert run.created_by == "user_dave"

    def test_run_get(self, service: RunService) -> None:
        run = service.get("Model", "Scenario", 1)
        assert run.id == 1

    def test_run_set_as_default_version(self, service: RunService) -> None:
        service.set_as_default_version(1)

    def test_run_get_default_version(self, service: RunService) -> None:
        run = service.get_default_version("Model", "Scenario")
        assert run.id == 1

    def test_run_unset_as_default_version(self, service: RunService) -> None:
        service.unset_as_default_version(1)

    def test_run_get_by_id(self, service: RunService) -> None:
        run = service.get_by_id(1)
        assert run.id == 1

    def test_run_lock(self, service: RunService) -> None:
        service.lock(1)

    def test_run_unlock(self, service: RunService) -> None:
        service.unlock(1)

    def test_run_revert(self, versioning_service: RunService) -> None:
        versioning_service.revert(1, 1)

    def test_run_list(self, service: RunService) -> None:
        results = service.list(default_only=False)
        assert len(results) == 1

    def test_run_tabulate(self, service: RunService) -> None:
        results = service.tabulate(default_only=False)
        assert len(results) == 1

    def test_run_delete(self, service: RunService) -> None:
        with pytest.raises(Forbidden):
            service.delete_by_id(1)
