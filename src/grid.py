
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


class Grid:
    FREE = 0
    OBSTACLE = 1

    def __init__(self, height: int, width: int,
                 obstacles=None, start=(0, 0), goal=None):
        self.height = height
        self.width = width
        self.cells = np.zeros((height, width), dtype=np.int8)
        self.start = start
        self.goal = goal if goal is not None else (height - 1, width - 1)
        if obstacles:
            for r, c in obstacles:
                self.cells[r, c] = self.OBSTACLE

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def is_free(self, r: int, c: int) -> bool:
        return (0 <= r < self.height and
                0 <= c < self.width and
                self.cells[r, c] == self.FREE)

    def neighbors(self, state):
        """Renvoie les voisins libres (4-connexité)."""
        r, c = state
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self.is_free(r + dr, c + dc):
                result.append((r + dr, c + dc))
        return result

    def free_cells(self):
        """Renvoie la liste de toutes les cellules libres (r, c)."""
        return [(r, c)
                for r in range(self.height)
                for c in range(self.width)
                if self.cells[r, c] == self.FREE]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize(self, path=None, title="Grille", ax=None, save_path=None):
        """Affiche la grille. Si ax fourni, dessine dedans ; sinon crée une figure."""
        own_fig = ax is None
        if own_fig:
            figw = max(5, self.width * 0.8)
            figh = max(5, self.height * 0.8)
            fig, ax = plt.subplots(figsize=(figw, figh))

        # Fond : blanc = libre, noir = obstacle
        display = self.cells.copy().astype(float)
        ax.imshow(display, cmap='Greys', vmin=0, vmax=1, origin='upper')

        # Lignes de grille
        for i in range(self.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.4)
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.4)

        # Chemin (bleu transparent)
        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='royalblue', alpha=0.55, zorder=2))

        # Départ (vert) / But (rouge)
        sr, sc = self.start
        gr, gc = self.goal
        ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1,
                                   color='limegreen', alpha=0.9, zorder=3))
        ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                                   color='crimson', alpha=0.9, zorder=3))
        ax.text(sc, sr, 'S', ha='center', va='center',
                color='white', fontweight='bold', fontsize=9, zorder=4)
        ax.text(gc, gr, 'G', ha='center', va='center',
                color='white', fontweight='bold', fontsize=9, zorder=4)

        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.tick_params(labelsize=7)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if own_fig:
            plt.tight_layout()
            if not save_path:
                plt.show()
            plt.close()


# ------------------------------------------------------------------
# Grilles pré-définies
# ------------------------------------------------------------------

def make_grid_easy() -> Grid:
    """Grille 5×5, quelques obstacles."""
    return Grid(5, 5,
                obstacles=[(1, 2), (2, 2), (3, 2)],
                start=(0, 0), goal=(4, 4))


def make_grid_medium() -> Grid:
    """Grille 8×8, obstacles modérés."""
    obs = [
        (1, 1), (1, 2), (1, 3), (1, 5),
        (2, 5), (3, 5), (4, 5),
        (3, 1), (3, 2), (3, 3),
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
        (6, 2),
    ]
    return Grid(8, 8, obstacles=obs, start=(0, 0), goal=(7, 7))


def make_grid_hard() -> Grid:
    """Grille 12×12, style labyrinthe."""
    obs = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 1), (5, 2), (5, 3), (5, 4),
        (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
        (7, 1), (7, 2), (7, 3), (7, 4),
        (8, 6), (8, 7), (8, 8), (8, 9), (8, 10),
        (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6),
        (10, 7), (10, 8), (10, 9), (10, 10),
    ]
    return Grid(12, 12, obstacles=obs, start=(0, 0), goal=(11, 11))
