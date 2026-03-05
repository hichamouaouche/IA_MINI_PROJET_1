
import os
import sys
import time

# Assure que le répertoire courant est le bon
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Création anticipée des dossiers de sortie
os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

print("=" * 65)
print("  Mini-Projet IA : Planification Robuste sur Grille")
print("  A* + Chaînes de Markov à temps discret")
print("=" * 65)


def run_experiments():
    from src.experiments import experiment_1, experiment_2, experiment_3, experiment_4

    t0 = time.time()

    print("\n[1/4] Expérience 1 — UCS vs Greedy vs A* (3 grilles)...")
    exp1, figs1 = experiment_1()
    _ok("Expérience 1", t0)

    t1 = time.time()
    print("\n[2/4] Expérience 2 — Impact de ε (Markov + Monte-Carlo)...")
    exp2, path2, policy2, figs2 = experiment_2(n_episodes=600)
    _ok("Expérience 2", t1)

    t2 = time.time()
    print("\n[3/4] Expérience 3 — h=0 vs Manhattan...")
    exp3, figs3 = experiment_3()
    _ok("Expérience 3", t2)

    t3 = time.time()
    print("\n[4/4] Expérience 4 — Weighted A*...")
    exp4, figs4 = experiment_4()
    _ok("Expérience 4", t3)

    return exp1, exp2, exp3, exp4


def _ok(label, t_start):
    elapsed = time.time() - t_start
    print(f"  ✓ {label} terminée en {elapsed:.2f}s")


def main():
    GLOBAL_START = time.time()

    # ----------------------------------------------------------------
    # 1. Expériences
    # ----------------------------------------------------------------
    try:
        exp1, exp2, exp3, exp4 = run_experiments()
    except Exception as e:
        print(f"\n  ERREUR lors des expériences : {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ----------------------------------------------------------------
    # Résumé final
    # ----------------------------------------------------------------
    elapsed = time.time() - GLOBAL_START
    print("\n" + "=" * 65)
    print(f"  PROJET TERMINÉ EN {elapsed:.1f}s")
    print("=" * 65)
    print("\n  Figures générées :")
    for f in sorted(os.listdir(os.path.join("outputs", "figures"))):
        print(f"    • outputs/figures/{f}")
    print()


if __name__ == "__main__":
    main()
