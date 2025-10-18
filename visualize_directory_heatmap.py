import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from init import DIR_CHOICES, time_slots, days_of_week

def load_sum_matrix(directory: str) -> np.ndarray:
    """Load processed_data.pkl from a directory and sum team 5x6 matrices."""
    filepath = os.path.join(directory, "processed_data.pkl")
    with open(filepath, "rb") as f:
        leagues, team_schedules = pickle.load(f)

    total = np.zeros((5, 6), dtype=float)
    for arr in team_schedules.values():
        A = np.asarray(arr, dtype=float)
        if A.shape != (5, 6):
            raise ValueError(f"Found array with shape {A.shape}, expected (5, 6).")
        total += A
    return total

def best_grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    """
    Choose a grid with up to `max_cols` columns.
    For n=3 -> (1,3); n=4-6 -> (2,3); n=7-9 -> (3,3), etc.
    """
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols

def main(selected_names=None, *, max_cols: int = 3):
    """
    selected_names: list of keys from DIR_CHOICES to include (order respected).
                    If None or empty, include all DIR_CHOICES in their defined order.
    max_cols: max columns in the subplot grid (default 3; set to 2 for a strict 2-wide grid).
    """
    # Resolve which (name,dir) pairs to plot
    all_items = list(DIR_CHOICES.items())
    if not selected_names:
        items = all_items
    else:
        # Keep only known names, in the order provided
        known = {k for k, _ in all_items}
        items = [(name, DIR_CHOICES[name]) for name in selected_names if name in known]

    if not items:
        raise SystemExit("No valid directories selected. Check your names in DIR_CHOICES.")

    # Load matrices (skip gracefully on errors)
    names, mats = [], []
    for name, d in items:
        try:
            mat = load_sum_matrix(d)
            names.append(name)
            mats.append(mat)
        except FileNotFoundError:
            print(f"[skip] {name}: processed_data.pkl not found in {d}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    n = len(mats)
    if n == 0:
        raise SystemExit("Nothing to plot after loading. All selections failed or were missing.")

    # Shared color scale
    vmin = min(np.min(m) for m in mats)
    vmax = max(np.max(m) for m in mats)

    rows, cols = best_grid(n, max_cols=max_cols)
    figsize = (4 * cols + 2, 3.5 * rows + 1)  # simple scaling
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    im = None
    for i, (name, mat) in enumerate(zip(names, mats)):
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        ax = axes[i]
        im = ax.imshow(mat, norm=norm, aspect="equal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name)
        ax.set_xticks(range(len(time_slots)))
        ax.set_yticks(range(len(days_of_week)))
        ax.set_xticklabels(time_slots, rotation=45, ha="right")
        ax.set_yticklabels(days_of_week)
        ax.set_xlabel("Time slots")
        ax.set_ylabel("Days")

        # light cell grid
        ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", linestyle=":", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Hide any unused axes (when grid has extras)
    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    cbar = fig.colorbar(im, ax=[a for a in axes if a.get_visible()], shrink=0.9)
    cbar.set_label("Sum of values across teams")

    plt.show()

if __name__ == "__main__":
    main(["fully_random", "optimal", "real_world_like"], max_cols=3)
    #main()
