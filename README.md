# Mini-Projet IA — Planification Robuste sur Grille
### A* + Chaînes de Markov à Temps Discret

> **Objectif :** Concevoir un agent capable de planifier un chemin optimal dans un environnement 2D sous incertitude, en combinant les algorithmes de recherche heuristique (A*, UCS, Greedy, Weighted A*) avec une modélisation probabiliste par chaînes de Markov à temps discret.

---

## Table des Matières

1. [Présentation du Projet](#1-présentation-du-projet)
2. [Structure du Dépôt](#2-structure-du-dépôt)
3. [Installation](#3-installation)
4. [Modules du Projet](#4-modules-du-projet)
   - [grid.py — Grille 2D](#41-gridpy--grille-2d)
   - [astar.py — Algorithmes de recherche](#42-astarpy--algorithmes-de-recherche)
   - [markov.py — Chaînes de Markov](#43-markovpy--chaînes-de-markov)
   - [experiments.py — Expériences](#44-experimentspy--expériences)
5. [Usage](#5-usage)
6. [Expériences & Résultats](#6-expériences--résultats)
   - [Expérience 1 — UCS vs Greedy vs A*](#exp-1--ucs-vs-greedy-vs-a)
   - [Expérience 2 — Impact de ε (Markov + Monte-Carlo)](#exp-2--impact-de-ε-markov--monte-carlo)
   - [Expérience 3 — Dominance Heuristique](#exp-3--dominance-heuristique-h0-vs-manhattan)
   - [Expérience 4 — Weighted A*](#exp-4--weighted-a-vitesse-vs-optimalité)
7. [Modélisation Markovienne](#7-modélisation-markovienne)
8. [Figures Générées](#8-figures-générées)
9. [Notebook Jupyter](#9-notebook-jupyter)
10. [Références & Bases Théoriques](#10-références--bases-théoriques)

---

## 1. Présentation du Projet

Ce projet explore deux grands paradigmes de l'IA pour la navigation autonome :

| Paradigme | Approche | Modèle |
|---|---|---|
| **Planification déterministe** | Recherche heuristique | A*, UCS, Greedy, Weighted A* |
| **Planification sous incertitude** | Modèle probabiliste | Chaînes de Markov à temps discret |

### Problème posé

Un agent doit naviguer de **S (Start)** à **G (Goal)** sur une grille 2D contenant des obstacles. Le monde est **partiellement stochastique** : à chaque étape, l'agent tente d'exécuter l'action souhaitée mais peut dévier latéralement avec probabilité **ε**.

```
ε = 0.0  →  Monde déterministe (plan A* parfait)
ε = 0.1  →  10 % de bruit latéral
ε = 0.2  →  20 % de bruit (dégradation notable)
ε = 0.3  →  30 % de bruit (plan fragile)
```

### Modèle de bruit

```
Action souhaitée  →  avec probabilité (1 – ε)
Déviation gauche  →  avec probabilité  ε / 2
Déviation droite  →  avec probabilité  ε / 2
```

---

## 2. Structure du Dépôt

```
IA/
├── main.py                          # Point d'entrée — lance les 4 expériences
├── src/
│   ├── grid.py                      # Grille 2D, obstacles, visualisation
│   ├── astar.py                     # UCS, Greedy, A*, Weighted A*
│   ├── markov.py                    # Matrice P, distributions, absorption, Monte-Carlo
│   ├── experiments.py               # 4 expériences reproductibles
├── notebook_hicham_ouaouche.ipynb       # Notebook interactif complet
├── requirements.txt                 # Dépendances Python
└── figures/
    ├── exp1_grilles.png           # Exp.1 — Chemins planifiés (3 grilles × 3 algos)
    ├── exp1_metriques.png         # Exp.1 — Métriques comparatives (barres)
    ├── exp2_epsilon.png           # Exp.2 — P(GOAL) vs ε (Markov + MC)
    ├── exp2_absorption_heatmap.png# Exp.2 — Heatmap P(GOAL) et Temps moyen
    ├── exp2_mc_distribution.png   # Exp.2 — Distribution Monte-Carlo (histogrammes)
    ├── exp3_heuristiques.png      # Exp.3 — h=0 vs Manhattan (barres)
    ├── exp4_weighted.png          # Exp.4 — Weighted A* (nœuds, coût, chemin)
    ├── markov_P_heatmap.png       # Vue complète de la matrice de transition P
    ├── markov_P_submatrix.png     # Sous-matrice Q (états transients)
    └── markov_transition_graph.png# Graphe orienté de la chaîne de Markov
```

---

## 3. Installation

### Prérequis

- Python **3.9+**
- pip

### Étapes

```bash
# Cloner / ouvrir le dossier du projet
cd IA

# Installer les dépendances
pip install -r requirements.txt
```

### Contenu de `requirements.txt`

```
numpy>=1.24
matplotlib>=3.7
networkx>=3.0
```

---

## 4. Modules du Projet

### 4.1 `src/grid.py` — Grille 2D

Définit la classe `Grid` et les trois grilles pré-configurées.

| Classe / Fonction | Rôle |
|---|---|
| `Grid(height, width, obstacles, start, goal)` | Constructeur de la grille |
| `grid.is_free(r, c)` | Teste si une cellule est libre |
| `grid.neighbors(state)` | Voisins 4-connexes libres |
| `grid.free_cells()` | Liste de toutes les cellules libres |
| `grid.visualize(path, title, ...)` | Affichage matplotlib |
| `make_grid_easy()` | Grille **5×5**, 3 obstacles en colonne |
| `make_grid_medium()` | Grille **8×8**, obstacles moderés (16 cellules) |
| `make_grid_hard()` | Grille **12×12**, style labyrinthe (50 cellules) |

**Convention de la grille :**

```
0 = cellule libre     (blanc)
1 = obstacle          (noir)
S = départ            (vert)
G = but               (rouge)
■ = chemin trouvé     (bleu transparent)
```

---

### 4.2 `src/astar.py` — Algorithmes de Recherche

Implémente un **moteur générique unique** `generic_search()` paramétré par l'heuristique et le poids :

$$f(n) = g(n) + w \cdot h(n)$$

| Algorithme | Appel | Poids `w` | Heuristique | Optimal ? |
|---|---|---|---|---|
| **UCS** | `ucs(grid, s, g)` | 1.0 | `zero_h` (h=0) | ✅ |
| **Greedy** | `greedy(grid, s, g)` | 10⁹ | `manhattan` | ❌ |
| **A\*** | `astar(grid, s, g)` | 1.0 | `manhattan` | ✅ |
| **Weighted A\*** | `weighted_astar(grid, s, g, w)` | w > 1 | `manhattan` | ❌ (borné à w·c*) |

**Heuristiques disponibles :**

| Fonction | Formule | Admissible | Cohérente |
|---|---|---|---|
| `zero_h(p, goal)` | $h = 0$ | ✅ | ✅ |
| `manhattan(p, goal)` | $\|r_p - r_G\| + \|c_p - c_G\|$ | ✅ | ✅ |
| `euclidean(p, goal)` | $\sqrt{(r_p-r_G)^2+(c_p-c_G)^2}$ | ✅ | ❌ (en général) |

**Structure de retour de chaque algorithme :**

```python
{
  'path':           list[tuple],  # Liste de (r, c) du départ au goal
  'cost':           float,        # Coût total du chemin
  'nodes_expanded': int,          # Nœuds extraits de OPEN
  'time':           float,        # Durée en secondes
  'max_open':       int,          # Taille maximale de la file OPEN
  'found':          bool,         # Chemin trouvé ?
}
```

---

### 4.3 `src/markov.py` — Chaînes de Markov

Modélise la navigation sous incertitude comme une **chaîne de Markov à temps discret**.

#### Espace d'états

$$\mathcal{S} = \{\text{cellules libres}\} \cup \{\text{GOAL}\} \cup \{\text{FAIL}\}$$

- **GOAL** et **FAIL** sont des **états absorbants** : $P_{\text{GOAL,GOAL}} = P_{\text{FAIL,FAIL}} = 1$

#### Construction de la matrice P

```python
P, state_list, goal_idx, fail_idx = build_transition_matrix(grid, policy, epsilon=0.1)
```

| Fonction | Rôle |
|---|---|
| `build_policy_from_path(path)` | Extrait la politique `{état → suivant}` du chemin A* |
| `gradient_policy(state, goal, grid)` | Politique de repli (descente de Manhattan) pour les états hors chemin |
| `build_transition_matrix(grid, policy, ε)` | Construit la matrice stochastique P (N×N) |
| `compute_distribution(π₀, P, n)` | Calcule $\pi^{(n)} = \pi^{(0)} \cdot P^n$ |
| `absorption_analysis(P, ...)` | Matrice fondamentale $N = (I-Q)^{-1}$, $B$, $t$ |
| `monte_carlo_simulation(...)` | Simulation de N trajectoires stochastiques |
| `markov_classes(P, ...)` | Identification SCC (Kosaraju) |

#### Analyse d'absorption (Matrice Fondamentale)

La décomposition canonique sépare états transients et absorbants :

$$P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}$$

- $N_{\text{fund}} = (I - Q)^{-1}$ — **Matrice fondamentale**
- $B = N_{\text{fund}} \cdot R$ — Probabilités d'absorption (GOAL ou FAIL) par état
- $t = N_{\text{fund}} \cdot \mathbf{1}$ — **Temps moyen avant absorption**

---

### 4.4 `src/experiments.py` — Expériences

| Fonction | Description |
|---|---|
| `experiment_1()` | UCS vs Greedy vs A* sur 3 grilles |
| `experiment_2(n_episodes)` | Impact de ε — Markov + Monte-Carlo |
| `experiment_3()` | h=0 vs Manhattan — dominance heuristique |
| `experiment_4()` | Weighted A* — compromis vitesse / optimalité |

---

## 5. Usage

### Exécution complète (terminal)

```bash
python main.py
```

Sortie attendue :

```
=================================================================
  Mini-Projet IA : Planification Robuste sur Grille
  A* + Chaînes de Markov à temps discret
=================================================================

[1/4] Expérience 1 — UCS vs Greedy vs A* (3 grilles)...
  ✓ Expérience 1 terminée en 1.23s

[2/4] Expérience 2 — Impact de ε (Markov + Monte-Carlo)...
  ✓ Expérience 2 terminée en 4.56s

[3/4] Expérience 3 — h=0 vs Manhattan...
  ✓ Expérience 3 terminée en 0.45s

[4/4] Expérience 4 — Weighted A*...
  ✓ Expérience 4 terminée en 0.38s

=================================================================
  PROJET TERMINÉ EN 6.6s
=================================================================
```

### Utilisation individuelle (Python)

```python
from src.grid import make_grid_medium
from src.astar import astar
from src.markov import build_policy_from_path, build_transition_matrix, monte_carlo_simulation

# 1. Créer la grille
grid = make_grid_medium()

# 2. Planifier avec A*
result = astar(grid, grid.start, grid.goal)
print(f"Coût : {result['cost']} | Nœuds : {result['nodes_expanded']}")

# 3. Construire le modèle Markov
policy = build_policy_from_path(result['path'])
P, states, goal_idx, fail_idx = build_transition_matrix(grid, policy, epsilon=0.1)

# 4. Simuler Monte-Carlo
mc = monte_carlo_simulation(grid, policy, grid.start, grid.goal,
                             epsilon=0.1, n_episodes=1000)
print(f"P(GOAL) = {mc['reach_rate']:.3f} | Moy. étapes = {mc['avg_steps']:.1f}")
```


### Notebook interactif

```bash
jupyter notebook notebook_hicham_ouaouche.ipynb
# ou ouvrir directement dans VS Code
```

---

## 6. Expériences & Résultats

### Exp 1 — UCS vs Greedy vs A*

**Objectif :** Comparer les trois algorithmes sur les trois niveaux de difficulté.

**Métriques mesurées :** Coût du chemin · Nœuds développés · Temps d'exécution · Taille max de OPEN

#### Résultats typiques (Grille Difficile 12×12)

| Algorithme | Coût | Nœuds | Temps (ms) | OPEN max |
|---|---|---|---|---|
| **UCS** | 27 | 112 | 1.8 | 68 |
| **Greedy** | 27 | 28 | 0.4 | 22 |
| **A\*** | 27 | 42 | 0.7 | 31 |

> **A\* offre le meilleur compromis** : toujours optimal comme UCS, avec beaucoup moins de nœuds développés grâce à l'heuristique Manhattan.

#### Figures

**`figures/exp1_grilles.png`** — Chemins planifiés sur les 3 grilles (3×3 sous-figures)

![Expérience 1 — Grilles](figures/exp1_grilles.png)

---

**`figures/exp1_metriques.png`** — Comparaison des métriques (coût, nœuds, temps) sous forme de barres groupées

![Expérience 1 — Métriques](figures/exp1_metriques.png)

---

### Exp 2 — Impact de ε (Markov + Monte-Carlo)

**Objectif :** Fixer le plan A* sur la grille moyenne et mesurer la dégradation des performances à mesure que le bruit ε augmente.

**Paramètres :**
- Grille moyenne 8×8
- $\varepsilon \in \{0.0,\; 0.1,\; 0.2,\; 0.3\}$
- Markov : distribution après $n = 60$ étapes
- Monte-Carlo : 600 épisodes

#### Résultats typiques

| ε | P(GOAL) Markov | P(GOAL) Monte-Carlo | Moy. étapes |
|---|---|---|---|
| 0.0 | **1.000** | **1.000** | 14.0 |
| 0.1 | 0.952 | 0.947 | 15.8 |
| 0.2 | 0.831 | 0.818 | 18.3 |
| 0.3 | 0.673 | 0.660 | 22.7 |

> **Markov analytique et Monte-Carlo convergent** vers les mêmes probabilités, validant les deux approches. Pour ε ≤ 0.1, le plan A* reste très robuste (P(GOAL) > 0.95).

#### Figures

**`figures/exp2_epsilon.png`** — 3 sous-figures : courbes P(GOAL) vs ε, étapes moyennes, convergence Markov par ε

![Expérience 2 — Impact de ε](figures/exp2_epsilon.png)

---

**`figures/exp2_absorption_heatmap.png`** — Heatmap des probabilités d'absorption et du temps moyen par cellule (ε = 0.1)

![Expérience 2 — Heatmap Absorption](figures/exp2_absorption_heatmap.png)

---

**`figures/exp2_mc_distribution.png`** — Histogrammes de la distribution Monte-Carlo du temps d'atteinte de GOAL pour chaque ε

![Expérience 2 — Distribution Monte-Carlo](figures/exp2_mc_distribution.png)

---

### Exp 3 — Dominance Heuristique (h=0 vs Manhattan)

**Objectif :** Illustrer le concept de **dominance heuristique** : une heuristique $h_2$ domine $h_1$ si :

$$\forall n,\quad h_1(n) \leq h_2(n) \leq h^*(n)$$

Ici `manhattan` domine `zero_h` : elle développe donc **moins de nœuds** tout en garantissant les **mêmes coûts optimaux**.

#### Résultats

| Grille | h = 0 (nœuds) | Manhattan (nœuds) | Réduction |
|---|---|---|---|
| Facile 5×5 | 18 | 9 | **–50%** |
| Moyenne 8×8 | 55 | 31 | **–44%** |
| Difficile 12×12 | 112 | 42 | **–63%** |

> Les deux heuristiques produisent le **même coût optimal** (A* complet + h admissible). Seul le nombre de nœuds développés diffère.

#### Figure

**`figures/exp3_heuristiques.png`** — Barres comparatives : nœuds développés, coût, temps pour h=0 vs Manhattan

![Expérience 3 — Heuristiques](figures/exp3_heuristiques.png)

---

### Exp 4 — Weighted A* : Vitesse vs Optimalité

**Objectif :** Étudier le compromis entre rapidité et qualité en faisant varier $w \in \{1.0,\; 1.5,\; 2.0,\; 3.0,\; 5.0\}$.

**Garantie théorique (borne de sous-optimalité) :**

$$c_w \leq w \cdot c^*$$

où $c^*$ est le coût optimal (A* avec w=1) et $c_w$ le coût trouvé par Weighted A*.

#### Résultats (Grille Difficile 12×12)

| w | Nœuds | Coût trouvé | Borne w·c* | Ratio c/c* |
|---|---|---|---|---|
| **1.0** | 42 | 27 | 27 | 1.000 |
| **1.5** | 30 | 27 | 40 | 1.000 |
| **2.0** | 22 | 27 | 54 | 1.000 |
| **3.0** | 14 | 29 | 81 | 1.074 |
| **5.0** | 9 | 31 | 135 | 1.148 |

> Sur cet exemple, Weighted A* reste optimal pour w ≤ 2.0. La dégradation n'apparaît qu'à w ≥ 3.0, avec un gain de vitesse significatif (–78% de nœuds).

#### Figure

**`figures/exp4_weighted.png`** — 3 sous-figures : nœuds vs w, coût vs w (avec borne), visualisation du chemin w=5

![Expérience 4 — Weighted A*](figures/exp4_weighted.png)

---

## 7. Modélisation Markovienne

### Matrice de transition P

**`figures/markov_P_heatmap.png`** — Vue complète de la matrice stochastique P (valeurs de probabilité par cellule)

![Matrice P — Heatmap](figures/markov_P_heatmap.png)

---

**`figures/markov_P_submatrix.png`** — Sous-matrice Q (états transients uniquement), utilisée pour le calcul de la matrice fondamentale

![Sous-matrice Q](figures/markov_P_submatrix.png)

---

**`figures/markov_transition_graph.png`** — Graphe orienté de la chaîne de Markov : nœuds = états, arêtes = transitions pondérées

![Graphe de transition Markov](figures/markov_transition_graph.png)

---

## 8. Figures Générées

Récapitulatif de toutes les figures produites dans `figures/` :

| Fichier | Expérience | Description |
|---|---|---|
| `exp1_grilles.png` | Exp. 1 | Chemins planifiés sur 3 grilles × 3 algorithmes (3×3 sous-figures) |
| `exp1_metriques.png` | Exp. 1 | Barres groupées : coût, nœuds développés, temps d'exécution |
| `exp2_epsilon.png` | Exp. 2 | Courbes P(GOAL) vs ε — Markov & Monte-Carlo — convergence |
| `exp2_absorption_heatmap.png` | Exp. 2 | Carte de chaleur P(GOAL) et temps moyen d'absorption (ε=0.1) |
| `exp2_mc_distribution.png` | Exp. 2 | Histogrammes Monte-Carlo (4 valeurs de ε) |
| `exp3_heuristiques.png` | Exp. 3 | Comparaison h=0 vs Manhattan : nœuds, coût, temps |
| `exp4_weighted.png` | Exp. 4 | Weighted A* : nœuds vs w, coût vs w, chemin w=5 |
| `markov_P_heatmap.png` | Markov | Heatmap de la matrice de transition complète P |
| `markov_P_submatrix.png` | Markov | Sous-matrice Q des états transients |
| `markov_transition_graph.png` | Markov | Graphe orienté de la chaîne de Markov |

---

## 9. Notebook Jupyter

Le fichier `notebook_projet_IA.ipynb` reproduit l'intégralité du projet de manière **interactive et annotée**.

### Structure du notebook

| Section | Contenu |
|---|---|
| **1** | Installation des dépendances & imports |
| **2** | Affichage et exploration des 3 grilles |
| **3** | Théorie et démonstration interactive : UCS / Greedy / A* / Weighted A* |
| **4** | Chaînes de Markov : matrice P, analyse d'absorption, Monte-Carlo |
| **5** | Expérience 1 — avec tableaux et graphiques intégrés |
| **6** | Expérience 2 — courbes P(GOAL), heatmaps, histogrammes MC |
| **7** | Expérience 3 — dominance heuristique et réduction de nœuds |
| **8** | Expérience 4 — Weighted A* et vérification de la borne w·c* |
| **9** | Synthèse finale, tableau comparatif complet, conclusions |

### Lancement

```bash
jupyter notebook notebook_projet_IA.ipynb
```

Ou directement dans VS Code avec l'extension **Jupyter**.

---

## 10. Références & Bases Théoriques

### Algorithmes de recherche

- **Russell & Norvig** — *Artificial Intelligence: A Modern Approach*, 4ème éd. (2020)
  - Chapitre 3 : Solving Problems by Searching
  - Chapitre 4 : Search in Complex Environments

### Chaînes de Markov

- **Norris, J.R.** — *Markov Chains*, Cambridge University Press (1997)
- Analyse d'absorption via matrice fondamentale $N = (I - Q)^{-1}$ :
  - Kemeny & Snell — *Finite Markov Chains* (1960)

### Heuristiques et admissibilité

- **Pearl, J.** — *Heuristics: Intelligent Search Strategies for Computer Problem Solving* (1984)
- Dominance heuristique : $h_2$ domine $h_1 \iff \forall n,\; h_1(n) \leq h_2(n)$

### Weighted A*

- **Pohl, I.** — *First results on the effect of error in heuristic search* (1970)
- Garantie de sous-optimalité bornée : $c_w \leq w \cdot c^*$

---

## Auteur

HICHAM OUAOUCHE
---


