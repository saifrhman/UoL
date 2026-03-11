"""
Ultamation 2026 Programming Challenge
Stage 3 solver: prerequisites + type matching + load balancing.
"""

from scheduling_utils import get_data, send_solution


def solve_stage_3():
    data = get_data(stage=3)
    servers = data["servers"]
    tasks = data["tasks"]

    # --------- Helpers to tolerate key casing differences ----------
    def s_id(s): return s.get("ID", s.get("id"))
    def s_speed(s): return float(s.get("Speed", s.get("speed")))
    def s_types(s): return s.get("type", s.get("Type", s.get("types", s.get("Types"))))

    def t_id(t): return t.get("ID", t.get("id"))
    def t_len(t): return float(t.get("Length", t.get("length")))
    def t_type(t): return t.get("type", t.get("Type"))
    def t_pre(t):
        # Your sample showed "Prerequisite" (capital P)
        return t.get("Prerequisite", t.get("prerequisite"))

    # --------- Precompute server info ----------
    server_ids = [s_id(s) for s in servers]
    server_speed = {s_id(s): s_speed(s) for s in servers}
    server_types = {s_id(s): s_types(s) for s in servers}
    server_load = {sid: 0.0 for sid in server_ids}  # accumulated predicted seconds

    # Identify any "c-only" fast server (helpful special-case)
    c_only_servers = [sid for sid in server_ids if server_types[sid] == "c"]
    c_only_sid = c_only_servers[0] if c_only_servers else None

    # --------- Precompute task info ----------
    task_ids = [t_id(t) for t in tasks]
    task_len = {t_id(t): t_len(t) for t in tasks}
    task_type = {t_id(t): t_type(t) for t in tasks}
    task_pre = {t_id(t): t_pre(t) for t in tasks}

    # Compatible servers per task (type constraint)
    compat = {}
    for tid in task_ids:
        typ = task_type[tid]
        compat_servers = [sid for sid in server_ids if typ in server_types[sid]]
        if not compat_servers:
            raise ValueError(f"No compatible server exists for task {tid} type={typ}")
        compat[tid] = compat_servers

    # --------- Build dependency graph ----------
    children = {tid: [] for tid in task_ids}
    indegree = {tid: 0 for tid in task_ids}

    # Note: spec says single prerequisite, but we handle generally
    for tid in task_ids:
        pre = task_pre[tid]
        if pre is not None:
            if pre not in indegree:
                raise ValueError(f"Task {tid} prerequisite {pre} not found in tasks")
            children[pre].append(tid)
            indegree[tid] += 1

    # --------- READY set: tasks whose prerequisites satisfied ----------
    ready = {tid for tid in task_ids if indegree[tid] == 0}

    # Precompute "unlock power" (out-degree) for tie-breaking
    out_degree = {tid: len(children[tid]) for tid in task_ids}

    # --------- Scheduling loop ----------
    solution = []
    scheduled = set()

    def pick_next_task(ready_set):
        """
        Heuristic priority:
          1) Scarcity: fewer compatible servers first
          2) Larger tasks first
          3) Unlock more dependents
          4) Stable tie-break by task id
        """
        return min(
            ready_set,
            key=lambda tid: (
                len(compat[tid]),
                -task_len[tid],
                -out_degree[tid],
                tid
            )
        )

    def pick_server_for_task(tid):
        """
        Choose compatible server minimizing projected finish time:
          load[s] + length/speed[s]

        Small special-case: if task is 'c' and there's a c-only server,
        strongly prefer it unless it is clearly becoming a bottleneck.
        """
        length = task_len[tid]
        typ = task_type[tid]
        candidates = compat[tid]

        if typ == "c" and c_only_sid in candidates:
            # Prefer c-only server unless it is much worse than best alternative.
            best_any = min(
                candidates,
                key=lambda sid: server_load[sid] + (length / server_speed[sid])
            )
            # If c-only is within a small margin, force it to keep S3 busy.
            c_only_score = server_load[c_only_sid] + (length / server_speed[c_only_sid])
            best_score = server_load[best_any] + (length / server_speed[best_any])

            # Margin is a heuristic; tweak if you want. This is conservative.
            if c_only_score <= best_score * 1.08:
                return c_only_sid

        return min(
            candidates,
            key=lambda sid: server_load[sid] + (length / server_speed[sid])
        )

    while ready:
        tid = pick_next_task(ready)
        sid = pick_server_for_task(tid)

        # Record assignment
        solution.append((tid, sid))
        scheduled.add(tid)

        # Update server load
        server_load[sid] += task_len[tid] / server_speed[sid]

        # Remove from ready
        ready.remove(tid)

        # Unlock dependents
        for child in children[tid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.add(child)

    if len(solution) != len(task_ids):
        # If this happens, you likely have a cycle in prerequisites (shouldn't in contest data)
        remaining = [tid for tid in task_ids if tid not in scheduled]
        raise RuntimeError(
            f"Could not schedule all tasks (possible cycle). Remaining: {remaining[:20]}..."
        )

    # --------- Submit ----------
    resp = send_solution(stage=3, solution=solution)
    print("Stage 3 response:", resp)


if __name__ == "__main__":
    solve_stage_3()