
import numpy as np
import random
from collections import defaultdict


# ------------------------------------------------------------------
# Politique
# ------------------------------------------------------------------

def build_policy_from_path(path: list) -> dict:
   
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i + 1]
    return policy


def gradient_policy(state, goal, grid):

    nbs = grid.neighbors(state)
    if not nbs:
        return None
    return min(nbs, key=lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1]))


# ------------------------------------------------------------------
# Directions
# ------------------------------------------------------------------

def _direction(src, dst):
    return (dst[0] - src[0], dst[1] - src[1])


def _laterals(dr, dc):
    """Deux directions perpendiculaires à (dr, dc)."""
    return [(-dc, dr), (dc, -dr)]


def _try_move(grid, state, d):
    """Applique le déplacement d depuis state ; reste sur place si obstacle."""
    nr, nc = state[0] + d[0], state[1] + d[1]
    if grid.is_free(nr, nc):
        return (nr, nc)
    return state


# ------------------------------------------------------------------
# Construction de la matrice de transition
# ------------------------------------------------------------------

def build_transition_matrix(grid, policy: dict, epsilon: float = 0.1):

    free = grid.free_cells()
    state_list = list(free)
    state_idx = {s: i for i, s in enumerate(state_list)}

    # Indices des états absorbants
    goal_idx = state_idx[grid.goal]           # goal doit être une cellule libre
    fail_idx = len(state_list)
    state_list.append('FAIL')
    state_idx['FAIL'] = fail_idx
    N = len(state_list)

    P = np.zeros((N, N))

    # États absorbants
    P[goal_idx, goal_idx] = 1.0
    P[fail_idx, fail_idx] = 1.0

    for s in free:
        if s == grid.goal:
            continue                              # déjà absorbant

        idx = state_idx[s]

        # Déterminer l'action souhaitée
        intended_next = policy.get(s)
        if intended_next is None:
            # Repli : descente de Manhattan
            intended_next = gradient_policy(s, grid.goal, grid)
        if intended_next is None:
            # Aucun voisin libre → absorber dans FAIL (évite auto-boucle orpheline)
            P[idx, fail_idx] = 1.0
            continue

        dr, dc = _direction(s, intended_next)
        lats = _laterals(dr, dc)

        ns_intended = _try_move(grid, s, (dr, dc))
        ns_lat1 = _try_move(grid, s, lats[0])
        ns_lat2 = _try_move(grid, s, lats[1])

        # Accumulation des probabilités (même destination possible plusieurs fois)
        contrib = defaultdict(float)
        contrib[ns_intended] += 1.0 - epsilon
        contrib[ns_lat1] += epsilon / 2.0
        contrib[ns_lat2] += epsilon / 2.0

        row = np.zeros(N)
        for ns, prob in contrib.items():
            if ns in state_idx:
                row[state_idx[ns]] += prob
            else:
                row[fail_idx] += prob

        # Normalisation de sécurité
        s_sum = row.sum()
        if s_sum > 0:
            row /= s_sum

        P[idx] = row

    return P, state_list, goal_idx, fail_idx


# ------------------------------------------------------------------
# Distribution markovienne
# ------------------------------------------------------------------

def compute_distribution(pi0: np.ndarray, P: np.ndarray, n_steps: int):
  
    pi = pi0.copy()
    history = [pi.copy()]
    for _ in range(n_steps):
        pi = pi @ P
        history.append(pi.copy())
    return history


# ------------------------------------------------------------------
# Analyse d'absorption
# ------------------------------------------------------------------

def absorption_analysis(P: np.ndarray, state_list: list,
                        goal_idx: int, fail_idx: int):
    
    N = len(state_list)
    absorbing = {goal_idx, fail_idx}
    transient = [i for i in range(N) if i not in absorbing]

    if not transient:
        return None

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, [goal_idx, fail_idx])]

    try:
        I_mat = np.eye(len(transient))
        IQ = I_mat - Q
        # Utiliser lstsq pour robustesse face aux matrices mal conditionnées
        N_fund, _, rank, _ = np.linalg.lstsq(IQ, I_mat, rcond=1e-12)
        B = N_fund @ R           # colonnes : GOAL, FAIL
        t = N_fund @ np.ones(len(transient))
    except (np.linalg.LinAlgError, ValueError):
        return None

    return {
        'transient_indices': transient,
        'fundamental_matrix': N_fund,
        'absorption_probs': B,   # shape (n_transient, 2)
        'expected_time': t,      # shape (n_transient,)
    }


# ------------------------------------------------------------------
# Simulation Monte-Carlo
# ------------------------------------------------------------------

def monte_carlo_simulation(grid, policy: dict, start, goal,
                           epsilon: float = 0.1,
                           n_episodes: int = 1000,
                           max_steps: int = 500,
                           seed: int = 42):
   
    rng = random.Random(seed)

    goal_count = 0
    fail_count = 0
    step_counts = []

    for _ in range(n_episodes):
        state = start
        reached = False

        for step in range(max_steps):
            if state == goal:
                reached = True
                step_counts.append(step)
                break

            # Choisir l'action souhaitée
            intended = policy.get(state)
            if intended is None:
                intended = gradient_policy(state, goal, grid)
            if intended is None:
                # Bloqué sans voisin
                break

            dr, dc = _direction(state, intended)
            lats = _laterals(dr, dc)

            r = rng.random()
            if r < 1.0 - epsilon:
                chosen_d = (dr, dc)
            elif r < 1.0 - epsilon / 2.0:
                chosen_d = lats[0]
            else:
                chosen_d = lats[1]

            state = _try_move(grid, state, chosen_d)

        else:
            # max_steps dépassé
            pass

        if not reached:
            fail_count += 1
        else:
            goal_count += 1

    total = n_episodes
    return {
        'reach_rate': goal_count / total,
        'fail_rate': fail_count / total,
        'avg_steps': float(np.mean(step_counts)) if step_counts else float('inf'),
        'step_distribution': step_counts,
        'n_episodes': n_episodes,
    }


# ------------------------------------------------------------------
# Analyse des classes de communication (via BFS/DFS sur graphe orienté)
# ------------------------------------------------------------------

def markov_classes(P: np.ndarray, state_list: list, threshold: float = 1e-10):

    N = len(state_list)
    adj = [[] for _ in range(N)]
    radj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if P[i, j] > threshold:
                adj[i].append(j)
                radj[j].append(i)

    # Première passe Kosaraju
    visited = [False] * N
    order = []

    def dfs1(u):
        stack = [(u, False)]
        while stack:
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            if visited[v]:
                continue
            visited[v] = True
            stack.append((v, True))
            for w in adj[v]:
                if not visited[w]:
                    stack.append((w, False))

    for i in range(N):
        if not visited[i]:
            dfs1(i)

    # Seconde passe Kosaraju
    comp = [-1] * N
    comp_id = 0
    visited2 = [False] * N

    def dfs2(u, cid):
        stack = [u]
        while stack:
            v = stack.pop()
            if visited2[v]:
                continue
            visited2[v] = True
            comp[v] = cid
            for w in radj[v]:
                if not visited2[w]:
                    stack.append(w)

    for u in reversed(order):
        if not visited2[u]:
            dfs2(u, comp_id)
            comp_id += 1

    # Reconstruction des classes
    classes = defaultdict(list)
    for i, c in enumerate(comp):
        classes[c].append(i)
    classes_list = list(classes.values())



    persistent_ids = set()
    for c_id, members in classes.items():
        member_set = set(members)
        exits = False
        for m in members:
            for nb in adj[m]:
                if nb not in member_set:
                    exits = True
                    break
            if exits:
                break
        if not exits:
            persistent_ids.add(c_id)

    persistent = [i for i, c in enumerate(comp) if c in persistent_ids]
    transient_st = [i for i, c in enumerate(comp) if c not in persistent_ids]

    return {
        'classes': classes_list,
        'persistent': persistent,
        'transient': transient_st,
        'comp': comp,
    }
