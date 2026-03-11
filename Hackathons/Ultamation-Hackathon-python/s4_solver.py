"""
Ultamation 2026 Programming Challenge
Stage 4 solver: prerequisites + type matching + switching penalty minimization.
"""

from scheduling_utils import get_data, send_solution


def solve_stage_4():
    data = get_data(stage=4)
    servers = data["servers"]
    tasks = data["tasks"]

    # ---- Key helpers (handle possible casing differences) ----
    def s_id(s): return s.get("ID", s.get("id"))
    def s_speed(s): return float(s.get("Speed", s.get("speed")))
    def s_types(s): return s.get("type", s.get("Type", s.get("types", s.get("Types"))))

    def t_id(t): return t.get("ID", t.get("id"))
    def t_len(t): return float(t.get("Length", t.get("length")))
    def t_type(t): return t.get("type", t.get("Type"))
    def t_pre(t): return t.get("Prerequisite", t.get("prerequisite"))

    # ---- Precompute server info ----
    server_ids = [s_id(s) for s in servers]
    server_speed = {s_id(s): s_speed(s) for s in servers}
    server_types = {s_id(s): s_types(s) for s in servers}
    server_load = {sid: 0.0 for sid in server_ids}
    last_type = {sid: None for sid in server_ids}

    # Helpful: identify "single-type" servers (great for no-switch runs)
    single_type_servers = {sid for sid in server_ids if len(server_types[sid]) == 1}

    # ---- Precompute task info ----
    task_ids = [t_id(t) for t in tasks]
    task_len = {t_id(t): t_len(t) for t in tasks}
    task_type = {t_id(t): t_type(t) for t in tasks}
    task_pre = {t_id(t): t_pre(t) for t in tasks}

    # ---- Compatible servers per task (type constraint) ----
    compat = {}
    for tid in task_ids:
        typ = task_type[tid]
        candidates = [sid for sid in server_ids if typ in server_types[sid]]
        if not candidates:
            raise ValueError(f"No compatible server exists for task {tid} type={typ}")
        compat[tid] = candidates

    # ---- Build dependency graph ----
    children = {tid: [] for tid in task_ids}
    indegree = {tid: 0 for tid in task_ids}

    for tid in task_ids:
        pre = task_pre[tid]
        if pre is not None:
            if pre not in indegree:
                raise ValueError(f"Task {tid} prerequisite {pre} not found in tasks")
            children[pre].append(tid)
            indegree[tid] += 1

    ready = {tid for tid in task_ids if indegree[tid] == 0}
    out_degree = {tid: len(children[tid]) for tid in task_ids}

    # ---- Switch-aware scoring knobs ----
    # You don't know the real penalty. These weights tell the heuristic to "care a lot"
    # about switching while still respecting makespan/load.
    SWITCH_WEIGHT = 8.0          # bigger => avoid switching more aggressively
    SINGLE_TYPE_BONUS = 3.0      # prefer keeping single-type servers doing their one type
    SCARCITY_WEIGHT = 1.5        # prefer tasks with fewer compatible servers

    # ---- Scoring functions ----
    def projected_finish(sid, tid):
        """Time server sid would have after taking task tid (ignores switching penalty)."""
        return server_load[sid] + task_len[tid] / server_speed[sid]

    def switch_cost(sid, tid):
        """Penalty proxy: cost if type changes on same server."""
        typ = task_type[tid]
        lt = last_type[sid]
        if lt is None or lt == typ:
            return 0.0
        return SWITCH_WEIGHT

    def single_type_bonus(sid, tid):
        """Reward staying on a server's single type (helps avoid future switches)."""
        if sid in single_type_servers:
            # If this server can do exactly one type, reward assigning that type to it
            only = server_types[sid]
            if task_type[tid] == only:
                return -SINGLE_TYPE_BONUS
        return 0.0

    def pick_best_server_for_task(tid):
        """Pick server for task tid using switch-aware cost."""
        candidates = compat[tid]
        # minimize: projected_finish + switch_cost + (bonus)
        return min(
            candidates,
            key=lambda sid: (
                projected_finish(sid, tid)
                + switch_cost(sid, tid)
                + single_type_bonus(sid, tid)
            )
        )

    def task_priority_tuple(tid, chosen_sid):
        """
        We choose a task by minimizing this tuple.
        Order is important:
          1) Prefer no-switch assignments
          2) Prefer smaller projected finish time (helps makespan)
          3) Prefer scarce tasks (few compatible servers)
          4) Prefer larger tasks earlier
          5) Prefer tasks that unlock more (optional)
        """
        no_switch = 0 if (last_type[chosen_sid] is None or last_type[chosen_sid] == task_type[tid]) else 1
        return (
            no_switch,
            projected_finish(chosen_sid, tid),
            len(compat[tid]) * SCARCITY_WEIGHT,
            -task_len[tid],
            -out_degree[tid],
            tid
        )

    # ---- Main scheduling loop ----
    solution = []
    scheduled = set()

    while ready:
        # Evaluate best (task, server) choice among READY tasks
        best_tid = None
        best_sid = None
        best_key = None

        for tid in ready:
            sid = pick_best_server_for_task(tid)
            key = task_priority_tuple(tid, sid)
            if best_key is None or key < best_key:
                best_key = key
                best_tid = tid
                best_sid = sid

        # Assign best choice found
        tid = best_tid
        sid = best_sid

        solution.append((tid, sid))
        scheduled.add(tid)

        # Update server load + last_type for switching logic
        server_load[sid] += task_len[tid] / server_speed[sid]
        last_type[sid] = task_type[tid]

        # Update readiness (topological progression)
        ready.remove(tid)
        for ch in children[tid]:
            indegree[ch] -= 1
            if indegree[ch] == 0:
                ready.add(ch)

    if len(solution) != len(task_ids):
        remaining = [tid for tid in task_ids if tid not in scheduled]
        raise RuntimeError(
            f"Could not schedule all tasks (possible cycle). Remaining sample: {remaining[:20]}"
        )

    # ---- Submit ----
    resp = send_solution(stage=4, solution=solution)
    print("Stage 4 response:", resp)

    # Optional: local diagnostics (not official score)
    print("Estimated server loads:", server_load)
    print("Estimated makespan:", max(server_load.values()))


if __name__ == "__main__":
    solve_stage_4()