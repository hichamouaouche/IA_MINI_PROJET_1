
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from grid import Grid, make_grid_easy, make_grid_medium, make_grid_hard
from astar import ucs, greedy, astar, weighted_astar, manhattan, zero_h
from markov import (build_policy_from_path, build_transition_matrix,
                    compute_distribution, absorption_analysis,
                    monte_carlo_simulation, markov_classes)

# ------------------------------------------------------------------
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def _draw_grid_on_ax(ax, grid, path=None, title=""):
    """Dessine une grille + chemin dans un ax matplotlib."""
    display = grid.cells.copy().astype(float)
    ax.imshow(display, cmap='Greys', vmin=0, vmax=1, origin='upper')
    for i in range(grid.width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.4)
    for i in range(grid.height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.4)
    if path:
        for r, c in path:
            if (r, c) != grid.start and (r, c) != grid.goal:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color='royalblue', alpha=0.55, zorder=2))
    sr, sc = grid.start
    gr, gc = grid.goal
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                               color='limegreen', alpha=0.9, zorder=3))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                               color='crimson', alpha=0.9, zorder=3))
    ax.text(sc, sr, 'S', ha='center', va='center',
            color='white', fontweight='bold', fontsize=8, zorder=4)
    ax.text(gc, gr, 'G', ha='center', va='center',
            color='white', fontweight='bold', fontsize=8, zorder=4)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


# ==================================================================
# EXPÉRIENCE 1 — UCS vs Greedy vs A* sur 3 grilles
# ==================================================================

def experiment_1():

    print("\n" + "=" * 60)
    print("EXPÉRIENCE 1 : UCS vs Greedy vs A* — 3 grilles")
    print("=" * 60)

    grids = [
        ('Facile (5×5)',      make_grid_easy()),
        ('Moyenne (8×8)',     make_grid_medium()),
        ('Difficile (12×12)', make_grid_hard()),
    ]
    methods = [
        ('UCS',    lambda g, s, t: ucs(g, s, t)),
        ('Greedy', lambda g, s, t: greedy(g, s, t)),
        ('A*',     lambda g, s, t: astar(g, s, t)),
    ]

    results = {}

    # --- Figure : grilles + chemins ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for gi, (gname, grid) in enumerate(grids):
        results[gname] = {}
        for mi, (mname, method) in enumerate(methods):
            res = method(grid, grid.start, grid.goal)
            results[gname][mname] = res
            status = (f"Coût={int(res['cost']) if res['cost']!=float('inf') else '∞'} "
                      f"| Nœuds={res['nodes_expanded']}")
            _draw_grid_on_ax(axes[gi][mi], grid, path=res['path'],
                             title=f"{gname}\n{mname}\n{status}")
            print(f"  [{gname}] {mname:6s} : "
                  f"coût={res['cost']:.0f}  nœuds={res['nodes_expanded']:4d}  "
                  f"t={res['time']*1000:.2f}ms  OPEN_max={res['max_open']}")

    plt.suptitle("Expérience 1 : Chemins planifiés — UCS / Greedy / A*",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    p1 = _fig_path("exp1_grilles.png")
    plt.savefig(p1, bbox_inches='tight', dpi=150)
    plt.close()

    # --- Figure : métriques en barres ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    metric_keys   = ['cost', 'nodes_expanded', 'time']
    metric_labels = ['Coût du chemin', 'Nœuds développés', 'Temps (ms)']

    for mi, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes2[mi]
        x = np.arange(len(grids))
        width = 0.25
        for ki, (mname, _) in enumerate(methods):
            vals = []
            for gname, _ in grids:
                v = results[gname][mname][mk]
                if mk == 'time':
                    v *= 1000
                if v == float('inf'):
                    v = 0
                vals.append(v)
            ax.bar(x + ki * width, vals, width, label=mname,
                   color=colors[ki], alpha=0.85, edgecolor='white')
        ax.set_xticks(x + width)
        ax.set_xticklabels([g[0] for g in grids], rotation=12, fontsize=8)
        ax.set_title(ml, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(ml, fontsize=8)

    plt.suptitle("Expérience 1 : Comparaison des métriques", fontsize=13)
    plt.tight_layout()
    p2 = _fig_path("exp1_metriques.png")
    plt.savefig(p2, bbox_inches='tight', dpi=150)
    plt.close()

    return results, [p1, p2]


# ==================================================================
# EXPÉRIENCE 2 — Impact de ε
# ==================================================================

def experiment_2(n_episodes: int = 600):

    print("\n" + "=" * 60)
    print("EXPÉRIENCE 2 : Impact de ε sur la robustesse du plan A*")
    print("=" * 60)

    grid = make_grid_medium()
    epsilons = [0.0, 0.1, 0.2, 0.3]
    N_STEPS = 60

    res_astar = astar(grid, grid.start, grid.goal)
    path = res_astar['path']
    policy = build_policy_from_path(path)

    results = {}
    goal_probs_markov = []
    goal_probs_mc = []
    avg_steps_mc = []

    for eps in epsilons:
        P, state_list, goal_idx, fail_idx = build_transition_matrix(
            grid, policy, epsilon=eps)

        N = len(state_list)
        pi0 = np.zeros(N)
        start_idx = next(i for i, s in enumerate(state_list)
                         if s == grid.start)
        pi0[start_idx] = 1.0

        dists = compute_distribution(pi0, P, N_STEPS)
        p_goal = dists[-1][goal_idx]

        mc = monte_carlo_simulation(grid, policy, grid.start, grid.goal,
                                    epsilon=eps, n_episodes=n_episodes)

        results[eps] = {
            'astar_cost':        res_astar['cost'],
            'markov_reach':      p_goal,
            'mc_reach_rate':     mc['reach_rate'],
            'mc_avg_steps':      mc['avg_steps'],
            'mc_fail_rate':      mc['fail_rate'],
            'P':                 P,
            'state_list':        state_list,
            'goal_idx':          goal_idx,
            'fail_idx':          fail_idx,
            'distributions':     dists,
            'pi0':               pi0,
        }
        goal_probs_markov.append(p_goal)
        goal_probs_mc.append(mc['reach_rate'])
        avg_steps_mc.append(mc['avg_steps'] if mc['avg_steps'] != float('inf') else 0)

        print(f"  ε={eps:.1f} | A* coût={int(res_astar['cost'])} "
              f"| Markov P(GOAL@{N_STEPS})={p_goal:.3f} "
              f"| MC P(GOAL)={mc['reach_rate']:.3f} "
              f"| MC moy_étapes={mc['avg_steps']:.1f}")

    # --- Figure 1 : courbes P(GOAL) et étapes ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors4 = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    axes[0].plot(epsilons, goal_probs_markov, 'b-o', lw=2, label=f'Markov P(GOAL@{N_STEPS})')
    axes[0].plot(epsilons, goal_probs_mc, 'r--s', lw=2, label='Monte-Carlo P(GOAL)')
    axes[0].set_xlabel('ε (niveau d\'incertitude)', fontsize=9)
    axes[0].set_ylabel('Probabilité d\'atteindre GOAL', fontsize=9)
    axes[0].set_title('P(GOAL) vs ε', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.15)

    axes[1].bar(epsilons, avg_steps_mc, width=0.07,
                color=colors4, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('ε', fontsize=9)
    axes[1].set_ylabel('Nombre moyen d\'étapes', fontsize=9)
    axes[1].set_title('Étapes moyennes pour atteindre GOAL', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)

    ax3 = axes[2]
    for eps, color in zip(epsilons, colors4):
        dists = results[eps]['distributions']
        gi = results[eps]['goal_idx']
        gp = [d[gi] for d in dists]
        ax3.plot(gp, color=color, lw=1.8, label=f'ε={eps}')
    ax3.set_xlabel('Étape n', fontsize=9)
    ax3.set_ylabel('π(n)[GOAL]', fontsize=9)
    ax3.set_title('Convergence vers GOAL par ε', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.suptitle("Expérience 2 : Impact de ε — Markov vs Monte-Carlo", fontsize=13)
    plt.tight_layout()
    p_eps = _fig_path("exp2_epsilon.png")
    plt.savefig(p_eps, bbox_inches='tight', dpi=150)
    plt.close()

    # --- Figure 2 : heatmap absorption (ε = 0.1) ---
    P_ref = results[0.1]['P']
    sl = results[0.1]['state_list']
    gi_ref = results[0.1]['goal_idx']
    fi_ref = results[0.1]['fail_idx']

    abs_res = absorption_analysis(P_ref, sl, gi_ref, fi_ref)
    p_absorb = _fig_path("exp2_absorption_heatmap.png")
    if abs_res is not None:
        B = abs_res['absorption_probs']
        t_arr = abs_res['expected_time']
        trans_idx = abs_res['transient_indices']

        goal_heat = np.full((grid.height, grid.width), np.nan)
        time_heat = np.full((grid.height, grid.width), np.nan)

        for ti, si in enumerate(trans_idx):
            s = sl[si]
            if isinstance(s, tuple):
                r, c = s
                goal_heat[r, c] = B[ti, 0]
                time_heat[r, c] = t_arr[ti]

        # Masquer les obstacles
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.cells[r, c] == 1:
                    goal_heat[r, c] = np.nan
                    time_heat[r, c] = np.nan

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes2[0].imshow(goal_heat, cmap='YlGn', vmin=0, vmax=1, origin='upper')
        plt.colorbar(im1, ax=axes2[0], fraction=0.046)
        axes2[0].set_title('P(atteindre GOAL) par état initial\n(ε = 0.1)', fontsize=10)

        im2 = axes2[1].imshow(time_heat, cmap='plasma', origin='upper')
        plt.colorbar(im2, ax=axes2[1], fraction=0.046)
        axes2[1].set_title('Temps moyen avant absorption\n(ε = 0.1)', fontsize=10)

        for ax_h in axes2:
            for r in range(grid.height):
                for c in range(grid.width):
                    if grid.cells[r, c] == 1:
                        ax_h.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1, color='black'))
            sr, sc = grid.start
            gr2, gc2 = grid.goal
            ax_h.plot(sc, sr, 'cs', markersize=8, label='Start')
            ax_h.plot(gc2, gr2, 'r*', markersize=10, label='Goal')
            ax_h.legend(fontsize=8)

        plt.suptitle("Analyse d'absorption Markov (grille moyenne, ε=0.1)", fontsize=12)
        plt.tight_layout()
        plt.savefig(p_absorb, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        p_absorb = None

    # --- Figure 3 : distribution MC des étapes ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    axes3 = axes3.flatten()
    for i, (eps, color) in enumerate(zip(epsilons, colors4)):
        mc2 = monte_carlo_simulation(grid, policy, grid.start, grid.goal,
                                     epsilon=eps, n_episodes=n_episodes, seed=99)
        if mc2['step_distribution']:
            axes3[i].hist(mc2['step_distribution'], bins=20,
                          color=color, alpha=0.8, edgecolor='white')
        axes3[i].set_title(f'ε = {eps}  |  P(GOAL)={mc2["reach_rate"]:.2f}', fontsize=10)
        axes3[i].set_xlabel('Nombre d\'étapes', fontsize=8)
        axes3[i].set_ylabel('Fréquence', fontsize=8)
        axes3[i].grid(alpha=0.3)

    plt.suptitle("Distribution Monte-Carlo du temps d'atteinte de GOAL", fontsize=12)
    plt.tight_layout()
    p_mc = _fig_path("exp2_mc_distribution.png")
    plt.savefig(p_mc, bbox_inches='tight', dpi=150)
    plt.close()

    return results, path, policy, [p_eps, p_absorb, p_mc]


# ==================================================================
# EXPÉRIENCE 3 — h=0 vs Manhattan
# ==================================================================

def experiment_3():
    """
    Compare A*(h=0) vs A*(Manhattan) — admissibilité, dominance.
    """
    print("\n" + "=" * 60)
    print("EXPÉRIENCE 3 : h=0 vs Manhattan — dominance heuristique")
    print("=" * 60)

    grids = [
        ('Facile (5×5)',      make_grid_easy()),
        ('Moyenne (8×8)',     make_grid_medium()),
        ('Difficile (12×12)', make_grid_hard()),
    ]
    heuristics = [
        ('h = 0 (UCS)',  zero_h),
        ('Manhattan',    manhattan),
    ]

    results = {}

    for gname, grid in grids:
        results[gname] = {}
        for hname, h in heuristics:
            res = astar(grid, grid.start, grid.goal, heuristic=h)
            results[gname][hname] = res
            print(f"  [{gname}] {hname:15s} : "
                  f"nœuds={res['nodes_expanded']:5d}  "
                  f"coût={res['cost']:.0f}  "
                  f"t={res['time']*1000:.3f}ms")

    # --- Figure : barres comparatives ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(grids))
    width = 0.35
    colors_h = ['#4C72B0', '#DD8452']
    metric_keys   = ['nodes_expanded', 'cost', 'time']
    metric_labels = ['Nœuds développés', 'Coût optimal', 'Temps (ms)']

    for mi, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[mi]
        for hi, (hname, _) in enumerate(heuristics):
            vals = []
            for gname, _ in grids:
                v = results[gname][hname][mk]
                if mk == 'time':
                    v *= 1000
                if v == float('inf'):
                    v = 0
                vals.append(v)
            ax.bar(x + hi * width, vals, width, label=hname,
                   color=colors_h[hi], alpha=0.85, edgecolor='white')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([g[0] for g in grids], rotation=10, fontsize=8)
        ax.set_title(ml, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Expérience 3 : h=0 vs Manhattan — Admissibilité & Dominance",
                 fontsize=13)
    plt.tight_layout()
    p = _fig_path("exp3_heuristiques.png")
    plt.savefig(p, bbox_inches='tight', dpi=150)
    plt.close()

    return results, [p]


# ==================================================================
# EXPÉRIENCE 4 — Weighted A*
# ==================================================================

def experiment_4():

    print("\n" + "=" * 60)
    print("EXPÉRIENCE 4 : Weighted A* — Compromis vitesse / optimalité")
    print("=" * 60)

    grid = make_grid_hard()
    weights = [1.0, 1.5, 2.0, 3.0, 5.0]

    results = {}
    nodes_list, cost_list, time_list = [], [], []

    for w in weights:
        res = weighted_astar(grid, grid.start, grid.goal, weight=w)
        results[w] = res
        nodes_list.append(res['nodes_expanded'])
        cost_list.append(res['cost'] if res['cost'] != float('inf') else 0)
        time_list.append(res['time'] * 1000)
        print(f"  w={w:.1f} : nœuds={res['nodes_expanded']:4d}  "
              f"coût={res['cost']:.0f}  t={res['time']*1000:.3f}ms")

    opt_cost = cost_list[0]

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(weights, nodes_list, 'b-o', lw=2, ms=8)
    axes[0].set_xlabel('Poids w', fontsize=9)
    axes[0].set_ylabel('Nœuds développés', fontsize=9)
    axes[0].set_title('Vitesse (nœuds) vs w', fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(weights, cost_list, 'r-s', lw=2, ms=8, label='Coût trouvé')
    if opt_cost > 0:
        for w, c in zip(weights, cost_list):
            bound = w * opt_cost
            axes[1].plot(w, bound, 'g^', ms=8)
        axes[1].axhline(y=opt_cost, color='green', ls='--',
                        label=f'Coût optimal ({opt_cost:.0f})')
    axes[1].set_xlabel('Poids w', fontsize=9)
    axes[1].set_ylabel('Coût du chemin', fontsize=9)
    axes[1].set_title('Qualité du chemin vs w', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Chemins sur la grille pour w=1 et w=5
    for wi, w in enumerate([1.0, 5.0]):
        ax_g = axes[2]
        _draw_grid_on_ax(ax_g, grid,
                         path=results[w]['path'],
                         title=f'Chemin w={w}  (coût={results[w]["cost"]:.0f})')
        break  # N'afficher que w=1 pour laisser de la place
    _draw_grid_on_ax(axes[2], grid,
                     path=results[5.0]['path'],
                     title=f'Chemin A*(w=5, coût={results[5.0]["cost"]:.0f})'
                           f' vs optimal={opt_cost:.0f}')

    plt.suptitle("Expérience 4 : Weighted A* — Vitesse vs Optimalité", fontsize=13)
    plt.tight_layout()
    p = _fig_path("exp4_weighted.png")
    plt.savefig(p, bbox_inches='tight', dpi=150)
    plt.close()

    return results, [p]
