"""
Ultamation 2026 Programming Challenge
Stage 1 solver (greedy load balancing).
"""

from scheduling_utils import get_data, send_solution


def solve_stage_1():
    data = get_data(stage=1)
    servers = data["servers"]
    tasks = data["tasks"]

    print("servers:", servers)
    print("num tasks:", len(tasks))
    print("first 5 tasks:", tasks[:5])

    # Build lookup tables (handle both "ID"/"Speed" and "id"/"speed" just in case)
    
    def sid(s): return s.get("ID", s.get("id"))
    def spd(s): return s.get("Speed", s.get("speed"))
    def tid(t): return t.get("ID", t.get("id"))
    def leng(t): return t.get("Length", t.get("length"))

    server_ids = [sid(s) for s in servers]
    server_speed = {sid(s): float(spd(s)) for s in servers}
    server_time = {sid(s): 0.0 for s in servers}

    tasks_sorted = sorted(tasks, key=lambda t: float(leng(t)), reverse=True)

    solution = []
    for t in tasks_sorted:
        task_id = tid(t)
        length = float(leng(t))

        best_server = min(
            server_ids,
            key=lambda s_id: server_time[s_id] + (length / server_speed[s_id])
        )

        server_time[best_server] += length / server_speed[best_server]
        solution.append((task_id, best_server))

    resp = send_solution(stage=1, solution=solution)
    print("Stage 1 response:", resp)
    


if __name__ == "__main__":
    solve_stage_1()