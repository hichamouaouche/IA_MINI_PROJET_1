
import heapq
import math
import time


# ------------------------------------------------------------------
# Heuristiques
# ------------------------------------------------------------------

def manhattan(p, goal) -> float:
    """Distance de Manhattan (admissible, cohérente, coûts unitaires)."""
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def euclidean(p, goal) -> float:
    """Distance euclidienne (admissible, non cohérente en général)."""
    return math.sqrt((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2)


def zero_h(p, goal) -> float:
    """Heuristique nulle — donne UCS quand utilisée dans generic_search."""
    return 0.0


# ------------------------------------------------------------------
# Reconstruction de chemin
# ------------------------------------------------------------------

def _reconstruct(parent, start, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path if path[0] == start else []


# ------------------------------------------------------------------
# Moteur de recherche générique
# ------------------------------------------------------------------

def generic_search(grid, start, goal, heuristic=manhattan, weight: float = 1.0):
 
    t0 = time.perf_counter()

    # Tas : (f, g, état)
    OPEN = []
    heapq.heappush(OPEN, (0.0, 0.0, start))

    g_cost = {start: 0.0}
    parent = {start: None}
    CLOSED = set()

    nodes_expanded = 0
    max_open = 1

    while OPEN:
        max_open = max(max_open, len(OPEN))
        f_cur, g_cur, cur = heapq.heappop(OPEN)

        if cur in CLOSED:
            continue
        CLOSED.add(cur)
        nodes_expanded += 1

        if cur == goal:
            return {
                'path': _reconstruct(parent, start, goal),
                'cost': g_cost[goal],
                'nodes_expanded': nodes_expanded,
                'time': time.perf_counter() - t0,
                'max_open': max_open,
                'found': True,
            }

        for nb in grid.neighbors(cur):
            new_g = g_cur + 1.0          # coût uniforme = 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                parent[nb] = cur
                h_val = heuristic(nb, goal)
                f_val = new_g + weight * h_val
                heapq.heappush(OPEN, (f_val, new_g, nb))

    return {
        'path': [],
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'time': time.perf_counter() - t0,
        'max_open': max_open,
        'found': False,
    }


# ------------------------------------------------------------------
# API publique
# ------------------------------------------------------------------

def ucs(grid, start, goal):
    """Uniform Cost Search (h = 0, optimal)."""
    return generic_search(grid, start, goal, heuristic=zero_h, weight=1.0)


def greedy(grid, start, goal):
    """Greedy Best-First (h = Manhattan, non optimal)."""
    return generic_search(grid, start, goal, heuristic=manhattan, weight=1e9)


def astar(grid, start, goal, heuristic=manhattan, weight: float = 1.0):
    """A* avec heuristique et poids fournis (optimal si h admissible et w=1)."""
    return generic_search(grid, start, goal, heuristic=heuristic, weight=weight)


def weighted_astar(grid, start, goal, weight: float = 2.0):
    """Weighted A* : f = g + w·h (sous-optimal borné à w·opt)."""
    return generic_search(grid, start, goal, heuristic=manhattan, weight=weight)
