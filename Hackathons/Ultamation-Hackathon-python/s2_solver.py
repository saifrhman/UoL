from scheduling_utils import get_data, send_solution

def solve_stage_2():
    data = get_data(stage=2)
    servers = data["servers"]
    tasks = data["tasks"]
    print(data)

    # Helpers for key naming variations
    def sid(s): return s.get("ID", s.get("id"))
    def spd(s): return float(s.get("Speed", s.get("speed")))
    def stypes(s): return s.get("Type", s.get("type"))  # string like "ade"

    def tid(t): return t.get("ID", t.get("id"))
    def leng(t): return float(t.get("Length", t.get("length")))
    def ttype(t): return t.get("Type", t.get("type"))   # single char like "a"

    server_ids = [sid(s) for s in servers]
    server_speed = {sid(s): spd(s) for s in servers}
    server_time = {sid(s): 0.0 for s in servers}
    server_types = {sid(s): stypes(s) for s in servers}

    # Sort tasks by length desc (same good heuristic)
    tasks_sorted = sorted(tasks, key=lambda t: leng(t), reverse=True)

    solution = []
    for t in tasks_sorted:
        task_id = tid(t)
        length = leng(t)
        typ = ttype(t)

        # Filter compatible servers
        compatible = [s_id for s_id in server_ids if typ in server_types[s_id]]
        if not compatible:
            raise ValueError(f"No compatible server for task {task_id} with type {typ}")

        # Pick server that minimizes resulting finish time
        best_server = min(
            compatible,
            key=lambda s_id: server_time[s_id] + (length / server_speed[s_id])
        )

        server_time[best_server] += length / server_speed[best_server]
        solution.append((task_id, best_server))

    resp = send_solution(stage=2, solution=solution)
    print("servers sample:", servers)
    print("tasks sample:", tasks[:5])

    print("Stage 2 response:", resp)

    
if __name__ == "__main__":
    solve_stage_2()