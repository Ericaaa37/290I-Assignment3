from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated
import uvicorn
import json
import heapq

from utils import *
from dijkstra import dijkstra  # Provided in template, not required to call directly

# create FastAPI app
app = FastAPI()

# global variable for active graph
# we will store it as an adjacency dict:
# { "A": {"B": weight_ab, "C": weight_ac}, ... }
active_graph = None


@app.get("/")
async def root():
    return {"message": "Welcome to the Shortest Path Solver!"}


@app.post("/upload_graph_json/")
async def create_upload_file(file: UploadFile):
    """
    1. Only accept `.json` files.
    2. Parse JSON content and normalize it into an adjacency dictionary.
    3. If parsing/format invalid, return Upload Error.
    4. On success, set global active_graph and return Upload Success.
    """
    global active_graph

    filename = file.filename or ""

    # Check extension
    if not filename.lower().endswith(".json"):
        return {"Upload Error": "Invalid file type"}

    # Read and parse JSON
    try:
        raw = await file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return {"Upload Error": "Invalid JSON graph format"}

    # Expect a dict describing the graph
    if not isinstance(data, dict):
        return {"Upload Error": "Invalid JSON graph format"}

    # Normalize into adjacency dict
    normalized_graph = {}

    try:
        for node_id, neighbors in data.items():
            node_id = str(node_id)

            # Case 1: already adjacency dict: { "B": 3, "C": 5 }
            if isinstance(neighbors, dict):
                adj = {}
                for nbr, weight in neighbors.items():
                    adj[str(nbr)] = float(weight)
                normalized_graph[node_id] = adj

            # Case 2: list of edges: [ {"to": "...", "distance"/"weight": ...}, ... ]
            elif isinstance(neighbors, list):
                adj = {}
                for edge in neighbors:
                    if not isinstance(edge, dict):
                        continue
                    if "to" not in edge:
                        continue
                    w = edge.get("distance", edge.get("weight"))
                    if w is None:
                        continue
                    adj[str(edge["to"])] = float(w)
                normalized_graph[node_id] = adj

            # Any other structure: treat as having no outgoing edges
            else:
                normalized_graph[node_id] = {}
    except Exception:
        return {"Upload Error": "Invalid JSON graph format"}

    if not normalized_graph:
        return {"Upload Error": "Invalid JSON graph format"}

    # Store as active graph
    active_graph = normalized_graph

    return {"Upload Success": filename}


@app.get("/solve_shortest_path/start_node_id={start_node_id}&end_node_id={end_node_id}")
async def get_shortest_path(start_node_id: str, end_node_id: str):
    """
    Requirements:

    - If no active graph:
        {"Solver Error": "No active graph, please upload a graph first."}

    - If start or end node ID not in graph:
        {"Solver Error": "Invalid start or end node ID."}

    - Otherwise:
        {
            "shortest_path": <list of node IDs> or None,
            "total_distance": <number> or None
        }
    """
    global active_graph

    # 1. Check that a graph has been uploaded
    if not active_graph:
        return {
            "Solver Error": "No active graph, please upload a graph first."
        }

    # 2. Collect all node IDs (keys plus neighbors)
    nodes = set(active_graph.keys())
    for _, nbrs in active_graph.items():
        if isinstance(nbrs, dict):
            nodes.update(str(v) for v in nbrs.keys())

    if start_node_id not in nodes or end_node_id not in nodes:
        return {
            "Solver Error": "Invalid start or end node ID."
        }

    # 3. Ensure every node has an adjacency dict
    graph = {str(u): {} for u in nodes}
    for u, nbrs in active_graph.items():
        u = str(u)
        if isinstance(nbrs, dict):
            for v, w in nbrs.items():
                graph[u][str(v)] = float(w)

    # 4. Run Dijkstra on this adjacency list
    #    (self-contained implementation for positive weights)
    pq = [(0.0, start_node_id)]
    dist = {start_node_id: 0.0}
    prev = {}

    while pq:
        cur_d, u = heapq.heappop(pq)
        if cur_d > dist[u]:
            continue
        if u == end_node_id:
            break
        for v, w in graph.get(u, {}).items():
            new_d = cur_d + w
            if v not in dist or new_d < dist[v]:
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(pq, (new_d, v))

    # 5. No path found
    if end_node_id not in dist:
        return {
            "shortest_path": None,
            "total_distance": None
        }

    # 6. Reconstruct path
    path = [end_node_id]
    while path[-1] != start_node_id:
        parent = prev.get(path[-1])
        if parent is None:
            # Safety guard, treat as no path
            return {
                "shortest_path": None,
                "total_distance": None
            }
        path.append(parent)
    path.reverse()

    return {
        "shortest_path": path,
        "total_distance": dist[end_node_id]
    }


if __name__ == "__main__":
    print("Server is running at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
