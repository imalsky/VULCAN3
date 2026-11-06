"""The job-queue runner: a batched loop that refills finished lanes and compacts its tail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ._carry import _ATM_ARRAY_FIELDS, _ATM_STATIC_BATCH_AXES, _PHOTO_FIELDS, _QUEUE_STOP_FIELDS
from .jax_step import AtmStatic

if TYPE_CHECKING:
    from ._loop_env import RunnerEnv

# run_queue's refill cadence: refill once this many lanes are free (or every
# lane still holding a job is), and at least every _QUEUE_REFILL_EVERY loop
# iterations. Untuned, picked by hand; a caller may pass its own `chunk`
# (a batched retrieval does, through the forward model).
_QUEUE_REFILL_CHUNK = 8
_QUEUE_REFILL_EVERY = 100


def _queue_compact_rungs(spec: Any, n_lanes: int) -> tuple:
    """run_queue's tail-compaction ladder (config `queue_compact`).

    The distinct lane counts of `spec` below `n_lanes`, widest first. A rung at or above the lane count is dropped,
    so one ladder serves every queue width; None or empty is no compaction.
    """
    if spec is None:
        return ()
    if isinstance(spec, (int, np.integer, str, bytes)):
        raise ValueError(  # noqa: TRY004 (a bad config value)
            f"queue_compact must be a list of lane counts, refilled {spec!r}"
        )
    rungs = []
    for r in spec:
        if isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or r < 1:
            raise ValueError(f"queue_compact: every rung must be an int >= 1, refilled {r!r} in {spec!r}")
        rungs.append(int(r))
    return tuple(sorted({r for r in rungs if r < int(n_lanes)}, reverse=True))


# size-limit-ok: one loop program; its closures share the queue's shapes
def runner_queue(  # noqa: C901 (one loop program)
    env: RunnerEnv, lane_step: Any, jobs: Any, n_lanes: int, chunk: int, init_fn: Any, out_fn: Any, compact: Any = ()
) -> Any:
    """Run the loop behind `OuterLoop.run_queue` (contract there).

    Finished lanes are written out and refilled inside this one while
    loop, so wall time follows total work / n_lanes, not the slowest job.
    The static atmosphere rides the carry so a refill can swap a lane's.
    Every lane takes its own photolysis fields exactly once before its
    first chemistry step: on its photo cadence (`it % update_photo_frq ==
    0`, always true at tick 0) from `body_fn_batch`'s gate inside that
    step, as in `runner_batch`; off its cadence at entry, or it would
    step on the previous occupant's RT state.

    `compact` (`_queue_compact_rungs`: descending widths below
    `n_lanes`) adds tail compaction; `()` is one loop to the end.
    Once every job has been handed to a lane and the live lanes (holding
    a job, not done) fit the next rung, the loop exits and the live lanes
    are gathered into a carry of that width, which runs the same loop
    body from the same tick (nothing is pending, so it never refills),
    and so on down the ladder.
    The done lanes still holding a job are written out at the gather with
    the frozen state the final flush would write.
    """
    sizes = {jnp.shape(x)[0] for x in jax.tree_util.tree_leaves(jobs) if jnp.ndim(x) > 0}
    if len(sizes) != 1:
        raise ValueError(
            f"run_queue: the array leaves of `jobs` must share one leading job axis; found lengths {sorted(sizes)}."
        )
    n_jobs = int(sizes.pop())
    n_lanes = int(n_lanes)
    if n_jobs < 1:
        raise ValueError("run_queue: the job queue is empty (n_jobs = 0).")
    if n_lanes < 1:
        raise ValueError(
            f"run_queue: n_lanes must be >= 1; refilled {n_lanes}. With no lane the queue is never drained."
        )
    chunk = int(max(1, min(int(chunk), n_lanes)))

    def job_at(j: Any) -> Any:
        """Job `j` with no leading axis.

        Leaves that carry no job axis (the AtmStatic toggles are Python scalars) pass through.
        """
        j = jnp.clip(j, 0, n_jobs - 1)
        return jax.tree_util.tree_map(lambda x: x[j] if jnp.ndim(x) > 0 else x, jobs)

    def fresh(j: Any, *, first_tick: Any) -> Any:
        """(k,) job ids -> k fresh lane states, their atmosphere arrays and their AtmStatic.

        The toggles of the AtmStatic are unmapped.

        `first_tick` is the tick of the lanes' first step; a lane off its
        photo cadence there is put on its own photolysis fields here (see
        the docstring). The initial fill passes 0, which is on every
        cadence.
        """
        st, atm = jax.vmap(
            lambda jj: init_fn(job_at(jj)),
            out_axes=(0, _ATM_STATIC_BATCH_AXES),
        )(j)
        if env.photo_branch is not None and not isinstance(first_tick, int):
            off = jnp.mod(first_tick, st.update_photo_frq) != jnp.int32(0)
            lit = jax.vmap(env.photo_branch)(st)
            st = st._replace(
                **{
                    f: jnp.where(
                        off.reshape((-1,) + (1,) * (jnp.ndim(getattr(lit, f)) - 1)),
                        getattr(lit, f),
                        getattr(st, f),
                    )
                    for f in _PHOTO_FIELDS
                }
            )
        return st, {f: getattr(atm, f) for f in _ATM_ARRAY_FIELDS}, atm

    j0 = jnp.arange(n_lanes, dtype=jnp.int32)
    lanes, atm_l, atm_tmpl = fresh(j0, first_tick=0)
    # Only the toggles are kept from the template: the array fields ride
    # the carry, so holding the initial ones here would bake n_lanes
    # columns into the program as constants.
    toggles = {f: getattr(atm_tmpl, f) for f in AtmStatic._fields if f not in _ATM_ARRAY_FIELDS}
    # Lanes past the last job are idle: frozen from the start, holding no
    # job (-1), never written out.
    have = j0 < jnp.int32(n_jobs)
    lanes = lanes._replace(is_done=lanes.is_done | jnp.logical_not(have))
    lane_job = jnp.where(have, j0, jnp.int32(-1))
    out = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_jobs, *x.shape[1:]), x.dtype),
        jax.vmap(out_fn)(lanes),
    )
    nxt = jnp.int32(min(n_lanes, n_jobs))

    def with_atm(al: Any) -> Any:
        return AtmStatic(**al, **toggles)

    def write_out(out: Any, lanes: Any, dst_lane: Any, dst_job: Any, valid: Any) -> Any:
        """Write `dst_lane`'s summaries to their jobs.

        An invalid slot is sent to the out-of-bounds index `n_jobs`, which `mode="drop"` discards.
        """
        summ = jax.vmap(out_fn)(jax.tree_util.tree_map(lambda x: x[dst_lane], lanes))
        dst = jnp.where(valid, dst_job, jnp.int32(n_jobs))
        return jax.tree_util.tree_map(lambda o, v: o.at[dst].set(v, mode="drop"), out, summ)

    def refill(c: Any) -> Any:
        it, lanes, atm_l, lane_job, nxt, out = c
        # The carry's width: n_lanes, or a compaction rung's (the tail
        # loops run `body` too, see there; with nothing pending this
        # branch never runs in them).
        w = int(lane_job.shape[0])
        lane_ix = j0 if w == n_lanes else jnp.arange(w, dtype=jnp.int32)
        free = lanes.is_done & (lane_job >= 0)
        # Free lanes first, then lane order: a fixed-size selection of
        # `chunk` slots whose leading entries are the refillable ones.
        order = jnp.logical_not(free).astype(jnp.int32) * jnp.int32(w) + lane_ix
        slots = jnp.argsort(order)[: min(chunk, w)]
        slot_is_free = free[slots]
        out = write_out(out, lanes, slots, lane_job[slots], slot_is_free)
        job_ids = nxt + jnp.cumsum(slot_is_free, dtype=jnp.int32) - jnp.int32(1)
        refilled = slot_is_free & (job_ids < jnp.int32(n_jobs))
        # `init_fn` runs on every slot, valid or not: shapes are fixed.
        new_st, new_atm, _ = fresh(job_ids, first_tick=it)

        def put(lane_arr: Any, new: Any) -> Any:
            g = refilled.reshape((-1,) + (1,) * (new.ndim - 1))
            return lane_arr.at[slots].set(jnp.where(g, new, lane_arr[slots]))

        lanes = jax.tree_util.tree_map(put, lanes, new_st)
        atm_l = jax.tree_util.tree_map(put, atm_l, new_atm)
        # A lane written out without a new job drops to -1, so `free`
        # never selects it again and it is not written twice.
        lane_job = lane_job.at[slots].set(
            jnp.where(refilled, job_ids, jnp.where(slot_is_free, jnp.int32(-1), lane_job[slots]))
        )
        return (it, lanes, atm_l, lane_job, nxt + jnp.sum(refilled, dtype=jnp.int32), out)

    def body(c: Any) -> Any:
        it, lanes, atm_l, lane_job, nxt, out = c
        lanes = lane_step(it, lanes, with_atm(atm_l))
        lanes = lanes._replace(**{f: jax.lax.stop_gradient(getattr(lanes, f)) for f in _QUEUE_STOP_FIELDS})
        it = it + jnp.int32(1)
        c = (it, lanes, atm_l, lane_job, nxt, out)
        n_free = jnp.sum(lanes.is_done & (lane_job >= 0), dtype=jnp.int32)
        n_held = jnp.sum(lane_job >= 0, dtype=jnp.int32)
        remaining = jnp.int32(n_jobs) - nxt
        want = jnp.minimum(jnp.int32(chunk), remaining)
        # Refill on a full chunk, or when every lane still holding a job
        # is free (the tail, fewer than `chunk` left), or on the cadence
        # so a single straggler does not hold the queue.
        trigger = (
            (remaining > 0)
            & (n_free > 0)
            & ((n_free >= want) | (n_free == n_held) | (jnp.mod(it, jnp.int32(_QUEUE_REFILL_EVERY)) == jnp.int32(0)))
        )
        return jax.lax.cond(trigger, refill, lambda cc: cc, c)

    def cond(c: Any) -> Any:
        _it, lanes, _atm_l, lane_job, nxt, _out = c
        live = jnp.logical_not(lanes.is_done) & (lane_job >= 0)
        return (nxt < jnp.int32(n_jobs)) | jnp.any(live)

    # ---- tail compaction (see the docstring)
    def n_live(lanes: Any, lane_job: Any) -> Any:
        return jnp.sum(jnp.logical_not(lanes.is_done) & (lane_job >= 0), dtype=jnp.int32)

    def until(width: int) -> Any:
        """`cond`, also ending once no job is pending and the live lanes fit `width`.

        `width` None is the last rung, run to the end.
        """
        if width is None:
            return cond

        def f(c: Any) -> Any:
            _it, lanes, _atm_l, lane_job, nxt, _out = c
            fits = (nxt >= jnp.int32(n_jobs)) & (n_live(lanes, lane_job) <= jnp.int32(width))
            return cond(c) & jnp.logical_not(fits)

        return f

    def gather(c: Any, width_from: Any, width_to: Any) -> Any:
        """Write out the done lanes holding a job.

        Keep the live lanes (lane order) in a carry of `width_to` lanes; the slots past them take
        done lanes, frozen and holding no job.
        """
        it, lanes, atm_l, lane_job, nxt, out = c
        idx = jnp.arange(width_from, dtype=jnp.int32)
        live = jnp.logical_not(lanes.is_done) & (lane_job >= 0)
        out = write_out(out, lanes, idx, lane_job, (lane_job >= 0) & lanes.is_done)
        order = jnp.logical_not(live).astype(jnp.int32) * jnp.int32(width_from) + idx
        slots = jnp.argsort(order)[:width_to]
        keep = live[slots]
        lanes = jax.tree_util.tree_map(lambda x: x[slots], lanes)
        atm_l = jax.tree_util.tree_map(lambda x: x[slots], atm_l)
        lanes = lanes._replace(is_done=lanes.is_done | jnp.logical_not(keep))
        lane_job = jnp.where(keep, lane_job[slots], jnp.int32(-1))
        return (it, lanes, atm_l, lane_job, nxt, out)

    ladder = (*tuple(compact), None)
    c = jax.lax.while_loop(until(ladder[0]), body, (jnp.int32(0), lanes, atm_l, lane_job, nxt, out))
    width = n_lanes
    for rung, nxt_rung in zip(compact, ladder[1:], strict=False):
        c = gather(c, width, rung)
        width = rung
        # The tail loops run `body` itself. With nothing pending its
        # refill cond is never taken, but it is what lets the lane
        # step's carry alias the loop's: a body that hands the lane
        # step's photolysis-cond outputs straight back to the carry makes
        # XLA copy every lane leaf each tick (20% of a 264-lane
        # tick).
        c = jax.lax.while_loop(until(nxt_rung), body, c)
    it, lanes, _atm_l, lane_job, _nxt, out = c
    # Flush the lanes still holding a job when the last loop ended.
    lane_ix = j0 if width == n_lanes else jnp.arange(width, dtype=jnp.int32)
    out = write_out(out, lanes, lane_ix, lane_job, lane_job >= 0)
    return out, it
