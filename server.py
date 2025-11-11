from fastapi import FastAPI, UploadFile
from typing_extensions import Annotated
import uvicorn
import numpy as np

from dijkstra import dijkstra
from utils import create_graph_from_json

# Create FastAPI app
app = FastAPI()

# Global variable to store the uploaded graph (Graph object)
active_graph = None


@app.get("/")
async def root():
    return {"message": "Welcome to the Shortest Path Solver!"}


@app.post("/upload_graph_json/")
async def create_upload_file(file: UploadFile):
    """
    Upload a JSON file that contains the graph data.
    The graph will be parsed into a Graph object and stored in 'active_graph'.
    """
    global active_graph

    # Check file type
    if not file.filename.lower().endswith(".json"):
        return {"Upload Error": "Invalid file type"}

    try:
        # Use the provided utility to construct the Graph
        # Note: do not call file.read() before this, since create_graph_from_json reads from file.file
        active_graph = create_graph_from_json(file)
        return {"Upload Success": file.filename}
    except Exception:
        # For this assignment, treat any failure as invalid file
        return {"Upload Error": "Invalid file type"}


@app.get("/solve_shortest_path/starting_node_id={starting_node_id}&end_node_id={end_node_id}")
async def get_shortest_path(starting_node_id: str, end_node_id: str):
    """
    Solve the shortest path problem for the given starting and ending node IDs.
    Uses Dijkstra's algorithm on the currently active graph.
    Returns:
        {
            "shortest_path": [list of node IDs] or None,
            "total_distance": float or None
        }
    """
    global active_graph

    # No graph uploaded
    if active_graph is None:
        return {"Solver Error": "No active graph, please upload a graph first."}

    # Check node existence in the Graph
    if (
        starting_node_id not in active_graph.nodes
        or end_node_id not in active_graph.nodes
    ):
        return {"Solver Error": "Invalid start or end node ID."}

    try:
        # Run Dijkstra from the starting node
        start_node = active_graph.nodes[starting_node_id]
        dijkstra(active_graph, start_node)

        end_node = active_graph.nodes[end_node_id]

        # If unreachable, return None for both
        if np.isinf(end_node.dist):
            return {"shortest_path": None, "total_distance": None}

        # Reconstruct path from end_node back to start_node using prev pointers
        path = []
        current = end_node
        while current is not None:
            path.append(current.id)
            current = current.prev
        path.reverse()

        return {
            "shortest_path": path,
            "total_distance": float(end_node.dist),
        }

    except Exception as e:
        return {"Solver Error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    print("Server is running at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
