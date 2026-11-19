"""Group-coordinated candidate races using the preparation sampler."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from ._cache import digest
from ._efficiency import AutotuneEffort
from ._ledger import RaceLedger
from ._measurement import capture_benchmark_call, no_compilation
from .tuning import CollectiveBucket
from .types import (
    CollectiveTuningRequirement,
    FinalistRecord,
    FrozenMapping,
    call_scope,
)


if TYPE_CHECKING:
    from .session import _Obligation


@dataclass(frozen=True)
class _CollectiveCalibration:
    role: Literal["full", "refined", "reused"]
    parents: tuple[_Obligation, ...] = ()


@dataclass(frozen=True)
class _CollectiveCalibrationRecord:
    bucket: CollectiveBucket
    ranks: tuple[int, ...]
    key: str
    coverage: FrozenMapping
    role: Literal["full", "refined", "reused"]
    parents: tuple[tuple[str, tuple[FrozenMapping, ...]], ...]


def _parent_records(calibration):
    return tuple(
        (
            parent.key,
            tuple(
                FrozenMapping(choice) for choice in parent.coverage["bucket_choices"]
            ),
        )
        for parent in calibration.parents
    )


def validate_retained_calibration(obligation):
    calibration = obligation.collective_calibration
    if calibration is None:
        return
    record = obligation.collective_record
    if record.role == "full":
        return
    if any(parent.selection.source == "default" for parent in calibration.parents):
        return
    if calibration.role != record.role or record.parents != _parent_records(
        calibration
    ):
        raise ValueError(
            f"retained collective {obligation.request.name} has a {record.role} calibration "
            f"that cannot supply the declared {calibration.role} calibration"
        )


class _Cancelled(Exception):
    pass


def _projection(assignment, parameters):
    return FrozenMapping({name: assignment[name] for name in parameters})


def tuning_bucket(obligation):
    request = obligation.request
    contract = request.plan.contract
    callback = contract.collective_bucket
    if (
        request.collective is None
        or callback is None
        or request.finalist_policy is not None
    ):
        return None
    space = obligation.configuration.space
    if (
        space is None
        or space.autotune_effort is not AutotuneEffort.DEFAULT
        or obligation.configuration.pinned is not None
        or request.dependencies
        or request.benchmark_call is not request.plan._benchmark_call
    ):
        return None
    if obligation.ready:
        record = obligation.collective_record
        return None if record is None else record.bucket
    bucket = callback(request.plan.query)
    if bucket is None:
        return None
    if not isinstance(bucket, CollectiveBucket):
        raise TypeError("collective_bucket must return a CollectiveBucket or None")
    parameters = {knob.name for knob in space.knobs}
    if set(bucket.parameters) - parameters:
        raise ValueError(
            "collective bucket parameters are absent from its parameter space"
        )
    if bucket.grid is not None and set(bucket.parameters) != parameters:
        raise ValueError("collective grids require every assignment parameter")
    projections = tuple(
        sorted(
            {
                _projection(assignment, bucket.parameters)
                for assignment, _ in obligation.candidates
            },
            key=digest,
        )
    )
    if len(projections) <= bucket.limit:
        return None
    return replace(
        bucket,
        identity=FrozenMapping(
            dict(
                contract=(
                    contract.component_id,
                    contract.query_schema_version,
                    contract.config_schema_version,
                    contract.semantic_version,
                    contract.candidate_contract_version,
                ),
                invocation=request.plan.invocation.to_dict(),
                bucket=bucket.identity,
                parameters=bucket.parameters,
                limit=bucket.limit,
                eligible=projections,
            )
        ),
    )


def calibration_order(obligations):
    """Order full calibrations, refined intervals, and exact-query validation."""
    representatives = {}
    grids = {}
    for index, obligation in enumerate(obligations):
        if obligation.selection is not None and not obligation.ready:
            continue
        bucket = tuning_bucket(obligation)
        if bucket is None:
            continue
        key = digest(
            FrozenMapping(
                dict(
                    ranks=obligation.request.collective.ranks,
                    identity=bucket.identity,
                )
            )
        )
        if bucket.grid is not None:
            grids.setdefault(key, []).append((index, obligation, bucket))
            continue
        previous = representatives.get(key)
        if previous is None or bucket.size_bytes > previous[0]:
            representatives[key] = (bucket.size_bytes, index)

    ordered = {"full": [], "refined": [], "reused": []}
    grid_indices = set()
    for family, members in sorted(grids.items()):
        intervals = {}
        for member in members:
            intervals.setdefault(member[2].grid.upper_bytes, []).append(member)
        fine = [
            min(
                group,
                key=lambda item: (-item[2].size_bytes, item[1].request.collective.key),
            )
            for group in intervals.values()
        ]
        fine.sort(key=lambda item: (item[2].size_bytes, item[1].request.collective.key))
        bands = {}
        for member in fine:
            bands[member[2].grid.band] = member
        anchors = {member[0]: member for member in (fine[0], *bands.values())}
        full = sorted(anchors.values(), key=lambda item: item[2].size_bytes)
        for position, representative, bucket in fine:
            if position in anchors:
                calibration = _CollectiveCalibration("full")
            else:
                lower = max(
                    (item for item in full if item[2].size_bytes < bucket.size_bytes),
                    key=lambda item: item[2].size_bytes,
                )
                upper = min(
                    (item for item in full if item[2].size_bytes > bucket.size_bytes),
                    key=lambda item: item[2].size_bytes,
                )
                calibration = _CollectiveCalibration("refined", (lower[1], upper[1]))
            representative.collective_calibration = calibration
            ordered[calibration.role].append(
                (family, bucket.size_bytes, representative)
            )
            for index, obligation, member_bucket in intervals[bucket.grid.upper_bytes]:
                grid_indices.add(index)
                if obligation is not representative:
                    obligation.collective_calibration = _CollectiveCalibration(
                        "reused", (representative,)
                    )
                    ordered["reused"].append(
                        (family, member_bucket.size_bytes, obligation)
                    )
    first = {index for _, index in representatives.values()}
    result = [
        item
        for role in ("full", "refined", "reused")
        for _, _, item in sorted(
            ordered[role],
            key=lambda entry: (entry[0], entry[1], entry[2].request.collective.key),
        )
    ]
    result.extend(item for index, item in enumerate(obligations) if index in first)
    result.extend(
        item
        for index, item in enumerate(obligations)
        if index not in first and index not in grid_indices
    )
    return result


def exchange(requirement, step, **payload):
    peers = yield CollectiveTuningRequirement(
        requirement.key, requirement.ranks, step, FrozenMapping(payload)
    )
    return tuple(item.payload for item in peers)


def _same(peers, field):
    value = peers[0][field]
    if any(peer[field] != value for peer in peers[1:]):
        raise ValueError(f"collective tuning ranks disagree on {field}")
    return value


def _load(job, obligation, record):
    contract = obligation.request.plan.contract
    configuration = obligation.configuration
    assignment = FrozenMapping(record["assignment"])
    configuration.space.validate(assignment)
    config = contract._lower(configuration.query, configuration.device, assignment)
    if contract.config_payload(config) != record["config"]:
        raise ValueError("collective cached assignment differs from its configuration")
    obligation.finalists = job._validate_finalists(
        obligation, record.get("finalists", ())
    )
    obligation.coverage = dict(record["coverage"])
    job._cache_hits += 1
    return job._selection(obligation, config, "cached", assignment)


def _bucket_choices(bucket, choices, assignment):
    if not isinstance(choices, (tuple, list)) or not 0 < len(choices) <= bucket.limit:
        raise ValueError(
            "collective bucket choices must be nonempty and within the limit"
        )
    eligible = set(bucket.identity["eligible"])
    selected = []
    for choice in choices:
        if not isinstance(choice, Mapping) or set(choice) != set(bucket.parameters):
            raise ValueError(
                "collective bucket choice has different projection parameters"
            )
        projection = FrozenMapping(choice)
        if projection not in eligible:
            raise ValueError(
                "collective bucket choice is outside its eligible projections"
            )
        if projection in selected:
            raise ValueError("collective bucket choices must be unique")
        selected.append(projection)
    if _projection(assignment, bucket.parameters) not in selected:
        raise ValueError("collective selection is outside its calibrated choices")
    return tuple(choice.to_dict() for choice in selected)


def select(job, obligation, selections):
    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    if contract.collective_signature is None:
        raise ValueError(
            "collective tuning requires a group-consistent query signature"
        )
    bucket = tuning_bucket(obligation)
    bucket_payload = (
        None
        if bucket is None
        else dict(
            identity=bucket.identity.to_dict(),
            size_bytes=bucket.size_bytes,
            parameters=bucket.parameters,
            limit=bucket.limit,
        )
    )
    grid = obligation.collective_calibration
    if bucket is not None and bucket.grid is not None:
        if grid is None:
            grid = _CollectiveCalibration("full")
        bucket_payload["grid"] = dict(
            upper_bytes=bucket.grid.upper_bytes, band=bucket.grid.band
        )
    grid_payload = (
        None
        if grid is None
        else dict(
            role=grid.role,
            parents=tuple(parent.request.collective.key for parent in grid.parents),
        )
    )
    candidates = tuple(
        (assignment.to_dict(), contract.config_payload(config).to_dict())
        for assignment, config in obligation.candidates
    )
    policy = obligation.request.finalist_policy
    policy_payload = (
        None
        if policy is None
        else (policy.family_fields, policy.per_family, policy.max_total)
    )
    peers = yield from exchange(
        requirement,
        "catalog",
        finalist_policy=policy_payload,
        candidates=candidates,
        signature=contract.collective_signature(obligation.request.plan.query),
        local_key=job._choice_key(obligation, selections),
        device=job.session.device.uuid,
        cache_only=job.session.cache_only,
        rounds=job.session.rounds,
        samples=job.session.samples,
        cancel=job.session._stop.is_set(),
        bucket=bucket_payload,
        grid=grid_payload,
    )
    for field in (
        "signature",
        "candidates",
        "cache_only",
        "rounds",
        "samples",
        "finalist_policy",
        "bucket",
        "grid",
    ):
        _same(peers, field)
    if any(peer["cancel"] for peer in peers):
        return job._selection(obligation, obligation.configuration.default, "default")
    bucket_key = None
    calibration = None
    if bucket is not None and grid is None:
        bucket_key = digest(
            FrozenMapping(
                dict(
                    identity=bucket.identity,
                    devices=tuple(peer["device"] for peer in peers),
                    ranks=requirement.ranks,
                )
            )
        )
        calibration = job.session._collective_buckets.get(bucket_key)
    grid_parents = ()
    grid_choices = None
    if grid is not None:
        if any(parent.selection.source == "default" for parent in grid.parents):
            return job._selection(
                obligation, obligation.configuration.default, "default"
            )
        grid_parents = tuple(
            (key, tuple(choice.to_dict() for choice in choices))
            for key, choices in _parent_records(grid)
        )
        if grid.role == "refined":
            grid_choices = tuple(
                dict.fromkeys(
                    FrozenMapping(choice)
                    for _, choices in grid_parents
                    for choice in choices
                )
            )
        elif grid.role == "reused":
            grid_choices = (FrozenMapping(grid_parents[0][1][0]),)
    cache_identity = {
        "measurement": (
            "collective_max_stream_gated_graph_v7_aligned_sample_finalists"
            if policy is not None
            else "collective_max_stream_gated_graph_v7_aligned_sample_token_scaled"
        ),
        "repetitions": {"reduction_token_target": 100, "default": 1},
        "rounds": job.session.rounds,
        "samples": job.session.samples,
        "members": tuple((peer["device"], peer["local_key"]) for peer in peers),
    }
    if bucket is not None:
        cache_identity["collective_bucket"] = (2, bucket_payload)
        cache_identity["bucket_calibration"] = None
    if grid is not None:
        cache_identity["collective_grid"] = (1, grid.role, grid_parents)
    obligation.key = digest(cache_identity)
    if calibration is not None and obligation.key != calibration[0]:
        obligation.key = digest({**cache_identity, "bucket_calibration": calibration})
    cache = job.session._selection_cache()
    peers = yield from exchange(
        requirement,
        "cache",
        record=cache.get(obligation.key),
        calibration=calibration,
        grid_parents=grid_parents,
    )
    _same(peers, "calibration")
    _same(peers, "grid_parents")
    record = next(
        (peer["record"] for peer in peers if peer["record"] is not None), None
    )
    if record is not None and (policy is None or record.get("finalists")):
        if bucket is not None:
            choices = _bucket_choices(
                bucket, record["coverage"].get("bucket_choices"), record["assignment"]
            )
            if calibration is not None and choices != calibration[1]:
                raise ValueError(
                    "cached collective choices differ from its calibration"
                )
            if grid is not None and FrozenMapping(choices[0]) != FrozenMapping(
                record["assignment"]
            ):
                raise ValueError(
                    "cached collective grid winner differs from its first choice"
                )
            if grid_choices is not None and any(
                FrozenMapping(choice) not in grid_choices for choice in choices
            ):
                raise ValueError(
                    "cached collective choices differ from its grid parents"
                )
        selection = _load(job, obligation, record)
        if grid is not None:
            obligation.collective_record = _CollectiveCalibrationRecord(
                bucket,
                requirement.ranks,
                obligation.key,
                FrozenMapping(obligation.coverage),
                grid.role,
                _parent_records(grid),
            )
        if bucket_key is not None and calibration is None:
            job.session._collective_buckets[bucket_key] = (obligation.key, choices)
        return selection
    if job.session.cache_only:
        raise LookupError(
            f"no completed collective selection for {obligation.request.name}"
        )
    if obligation.request.benchmark_call is None:
        raise ValueError("collective tuning requires a representative benchmark call")
    if grid_choices is not None and grid.role == "reused":
        try:
            selection = yield from _reuse_calibration(job, obligation, grid_choices[0])
            obligation.collective_record = _CollectiveCalibrationRecord(
                bucket,
                requirement.ranks,
                obligation.key,
                FrozenMapping(obligation.coverage),
                grid.role,
                _parent_records(grid),
            )
            return selection
        except _Cancelled:
            return job._selection(
                obligation, obligation.configuration.default, "default"
            )
    if calibration is not None or grid_choices is not None:
        choices = (
            set(grid_choices)
            if grid_choices is not None
            else {FrozenMapping(choice) for choice in calibration[1]}
        )
        before = len(obligation.candidates)
        obligation.candidates = [
            (assignment, config)
            for assignment, config in obligation.candidates
            if _projection(assignment, bucket.parameters) in choices
        ]
        if not obligation.candidates:
            raise ValueError(
                "collective bucket choices have no eligible configurations"
            )
        obligation.coverage["bucket_pruned_count"] = before - len(obligation.candidates)
        if calibration is not None:
            obligation.coverage["bucket_choices"] = calibration[1]
    try:
        selection = yield from race(
            job,
            obligation,
            bucket=bucket if calibration is None else None,
        )
        if grid is not None:
            obligation.collective_record = _CollectiveCalibrationRecord(
                bucket,
                requirement.ranks,
                obligation.key,
                FrozenMapping(obligation.coverage),
                grid.role,
                _parent_records(grid),
            )
        if bucket_key is not None and calibration is None:
            job.session._collective_buckets[bucket_key] = (
                obligation.key,
                tuple(obligation.coverage["bucket_choices"]),
            )
        return selection
    except _Cancelled:
        return job._selection(obligation, obligation.configuration.default, "default")


def _prepare_trial(
    job, obligation, index, assignment, config, live, *, ledger=None, capture=True
):
    from .session import _CallGuard, _Trial

    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    job._phase = "preparing candidates"
    if ledger is not None:
        ledger.note_precompiled(
            index,
            obligation.programs.get(contract.config_payload(config), frozenset()),
        )
    programs = yield from job._compile(obligation, config, required=True)
    yield from job._wait_programs(programs, obligation)
    compile_assignment = obligation.configuration.space.compile_assignment(assignment)
    previous = obligation.compile_assignments.setdefault(compile_assignment, programs)
    if previous != programs:
        raise ValueError("collective runtime parameter changed actual compile keys")
    selected = job._selection(obligation, config, "tuned", assignment)
    state, call, guard, retained = job._call(
        obligation,
        selected,
        obligation.request.benchmark_call,
        synchronize=False,
        prime=False,
    )
    trial = _Trial(index, assignment, config, call, guard, retained, state)
    live.append(trial)
    job._temporaries.remove(guard.finish)
    peers = yield from exchange(
        requirement,
        f"bound/{index}",
        cancel=job.session._stop.is_set(),
        producers=len(call.benchmark_producers) or 1,
        has_producer=call.produce is not None,
        repetitions=call.benchmark_repetitions,
    )
    _same(peers, "producers")
    repetitions = _same(peers, "repetitions")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("benchmark repetitions must be a positive integer")
    if not all(peer["has_producer"] for peer in peers):
        raise ValueError(
            "collective benchmark requires an input producer on every rank"
        )
    if any(peer["cancel"] for peer in peers):
        raise _Cancelled
    # Every rank has bound its operands before collective priming starts.
    for producer_index, producer in enumerate(
        call.benchmark_producers or (call.produce,)
    ):
        if producer is None:
            raise ValueError("collective benchmark requires an input producer")
        with job.session._gpu_scope(), no_compilation():
            with call_scope():
                if call.reset is not None:
                    call.reset()
                producer()
                call.invoke()
            job.session._synchronize()
            validation_error = None
            try:
                if call.validate is not None:
                    call.validate()
            except BaseException as error:
                validation_error = error
        peers = yield from exchange(
            requirement,
            f"validated/{index}/{producer_index}",
            error=None
            if validation_error is None
            else f"{type(validation_error).__name__}: {validation_error}",
            cancel=job.session._stop.is_set(),
        )
        if validation_error is not None:
            raise validation_error
        failures = [
            f"rank {rank}: {peer['error']}"
            for rank, peer in zip(requirement.ranks, peers, strict=True)
            if peer["error"] is not None
        ]
        if failures:
            raise RuntimeError("collective validation failed; " + "; ".join(failures))
        if any(peer["cancel"] for peer in peers):
            raise _Cancelled
    if capture:
        with job.session._gpu_scope(), no_compilation():
            captured = capture_benchmark_call(
                replace(call, restore=guard.restore, close=guard.close, validate=None),
                repetitions=repetitions,
            )
            trial.call = captured
            trial.guard = _CallGuard(captured)
    peers = yield from exchange(
        requirement, f"validated/{index}", cancel=job.session._stop.is_set()
    )
    if any(peer["cancel"] for peer in peers):
        raise _Cancelled
    return trial


def _reuse_calibration(job, obligation, choice):
    live = []
    try:
        index, (assignment, config) = next(
            (index, candidate)
            for index, candidate in enumerate(obligation.candidates)
            if candidate[0] == choice
        )
        yield from _prepare_trial(
            job, obligation, index, assignment, config, live, capture=False
        )
        obligation.coverage.update(
            measured_count=0,
            bucket_pruned_count=len(obligation.candidates) - 1,
            bucket_choices=(choice.to_dict(),),
        )
        obligation.cache_pending = True
        return job._selection(obligation, config, "tuned", assignment)
    finally:
        job.session._synchronize()
        for trial in reversed(live):
            trial.close()


def race(job, obligation, *, bucket=None):
    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    champion = None
    live = []
    measurements = {}
    policy = obligation.request.finalist_policy

    def family_of(trial):
        return FrozenMapping(
            {name: trial.assignment[name] for name in policy.family_fields}
        )

    job._candidate_sharded = False
    job._candidate_count = len(obligation.candidates)
    ledger = RaceLedger(
        job=job,
        obligation=obligation,
        kind="collective",
        shard_index=0,
        shard_count=1,
        local_indices=range(len(obligation.candidates)),
        restored={},
    )

    def sample_exchange(indices, turn, repetition, groups):
        peers = yield from exchange(
            requirement,
            f"samples/{job._batch_index}/{turn}/{repetition}",
            indices=indices,
            samples=groups,
            cancel=job.session._stop.is_set(),
        )
        _same(peers, "indices")
        if any(peer["cancel"] for peer in peers):
            raise _Cancelled
        if any(
            len(peer["samples"]) != len(groups)
            or any(
                len(values) != len(expected)
                for values, expected in zip(peer["samples"], groups, strict=True)
            )
            for peer in peers
        ):
            raise ValueError("collective ranks measured different sample counts")
        if any(
            not math.isfinite(value) or value <= 0
            for peer in peers
            for values in peer["samples"]
            for value in values
        ):
            raise ValueError("collective rank produced an invalid latency")
        result = tuple(
            tuple(
                max(peer["samples"][i][j] for peer in peers) for j in range(len(values))
            )
            for i, values in enumerate(groups)
        )
        if any(
            not math.isfinite(value) or value <= 0 for row in result for value in row
        ):
            raise ValueError("collective race produced an invalid latency")
        return result

    def launch_exchange(indices, turn, repetition, sample):
        peers = yield from exchange(
            requirement,
            f"queued/{job._batch_index}/{turn}/{repetition}/{sample}",
            indices=indices,
        )
        _same(peers, "indices")

    try:
        for index, (assignment, config) in enumerate(obligation.candidates):
            trial_started = time.perf_counter()
            trial = yield from _prepare_trial(
                job, obligation, index, assignment, config, live, ledger=ledger
            )
            repetitions = trial.call.benchmark_repetitions
            job._candidates_prepared += 1
            job._batch_index += 1
            ledger.prepared(
                trial,
                seconds=time.perf_counter() - trial_started,
                batch=job._batch_index,
            )
            trials = ([] if champion is None else [champion]) + [trial]
            if any(item.call.benchmark_repetitions != repetitions for item in trials):
                raise ValueError(
                    "collective candidates require the same repetition count"
                )
            job._batch_candidates = len(trials)
            measure_started = time.perf_counter()
            measurement = yield from job._measure(
                trials,
                champion=len(trials) == 2,
                adaptive_collective_pair=len(trials) == 2,
                repetitions=repetitions,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
                sample_observer=ledger.observer(trials),
            )
            best = min(
                range(len(trials)),
                key=lambda i: (measurement.latencies_us[i], trials[i].index),
            )
            measurements.update(
                (item.index, latency)
                for item, latency in zip(trials, measurement.latencies_us, strict=True)
            )
            ledger.batch(
                job._batch_index,
                trials,
                measurement,
                champion=champion,
                winner=trials[best].index,
                seconds=time.perf_counter() - measure_started,
            )
            for position, item in enumerate(trials):
                if position != best:
                    item.close()
                    live.remove(item)
            champion = trials[best]
            champion.latency_us = measurement.latencies_us[best]
            job._benchmarked += 1
            job._timing.record(
                "collective_batch_end",
                request=obligation.request.name,
                candidates=[item.index for item in trials],
                latencies_us=measurement.latencies_us,
                winner=champion.index,
            )
        if champion is None:
            raise ValueError("collective race has no eligible candidates")
        if policy is not None:
            nominees = {}
            for index, latency in measurements.items():
                assignment, _ = obligation.candidates[index]
                family = FrozenMapping(
                    {name: assignment[name] for name in policy.family_fields}
                )
                previous = nominees.get(family)
                if previous is None or (latency, index) < previous:
                    nominees[family] = (latency, index)
            nominees[family_of(champion)] = (champion.latency_us, champion.index)
            if len(nominees) > policy.max_total:
                raise ValueError("finalist family coverage exceeds max_total")
            trials = []
            for index in sorted(index for _, index in nominees.values()):
                if index == champion.index:
                    trial = champion
                else:
                    assignment, config = obligation.candidates[index]
                    trial = yield from _prepare_trial(
                        job, obligation, index, assignment, config, live, ledger=ledger
                    )
                trials.append(trial)
            job._batch_index += 1
            measurement = yield from job._measure(
                trials,
                champion=False,
                eliminate=False,
                adaptive_repeats=False,
                repetitions=trials[0].call.benchmark_repetitions,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
            )
            obligation.finalists = job._validate_finalists(
                obligation,
                tuple(
                    FinalistRecord(
                        assignment=FrozenMapping(item.assignment),
                        family=family_of(item),
                        candidate_index=item.index,
                        latency_us=float(latency),
                    )
                    for item, latency in zip(
                        trials, measurement.latencies_us, strict=True
                    )
                ),
            )
            obligation.coverage["finalist_count"] = len(obligation.finalists)
        peers = yield from exchange(
            requirement, "selected", assignment=champion.assignment
        )
        _same(peers, "assignment")
        obligation.coverage["measured_count"] = len(measurements)
        if bucket is not None:
            best = {}
            for index, latency in measurements.items():
                projection = _projection(
                    obligation.candidates[index][0], bucket.parameters
                )
                best[projection] = min(best.get(projection, math.inf), latency)
            winner = _projection(champion.assignment, bucket.parameters)
            choices = sorted(
                best,
                key=lambda choice: (choice != winner, best[choice], digest(choice)),
            )
            obligation.coverage["bucket_choices"] = tuple(
                choice.to_dict() for choice in choices[: bucket.limit]
            )
        obligation.cache_pending = True
        if obligation.key is not None:
            ledger.commit(
                job.session._race_progress_cache,
                measurements=measurements,
                champion_index=champion.index,
                rejected_count=0,
            )
        return job._selection(obligation, champion.config, "tuned", champion.assignment)
    finally:
        job.session._synchronize()
        for trial in reversed(live):
            trial.close()
