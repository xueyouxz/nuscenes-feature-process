"""
project_som.py — Planning residual field SOM projection.

Pipeline:
  1. Load GT ego_poses + pred final_plannings for each scene pair
  2. Compute per-frame residuals in ego coordinate system (last 6 planning points)
  3. Discard末尾 frames where < 6 future GT frames are available
  4. Align all scenes to minimum usable frame count (truncate, no padding)
  5. StandardScaler → PCA (n_components from config, or scree mode if null)
  6. Train MiniSom 8×8 on PCA-compressed features
  7. Save som_scene_projection.json + som_node_prototypes.json to output_dir

Two-phase run:
  - Phase 1: set pca_n_components=null in config → generates scree_plot.png and exits
  - Phase 2: set pca_n_components=<k> in config → runs full pipeline
"""

from __future__ import annotations

import json
import sys
from math import cos, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG_PATH = Path("configs/som_config.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {DEFAULT_CONFIG_PATH}")
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


def ego_transform(point_global: list[float], origin: list[float], yaw: float) -> tuple[float, float]:
    """
    Transform a 2D global-frame point into the ego frame at (origin, yaw).

    Rotation: global → ego requires rotating by -yaw.
    R(-yaw) applied to (dx, dy):
      ex =  dx * cos(yaw) + dy * sin(yaw)
      ey = -dx * sin(yaw) + dy * cos(yaw)
    """
    dx = point_global[0] - origin[0]
    dy = point_global[1] - origin[1]
    ex = dx * cos(yaw) + dy * sin(yaw)
    ey = -dx * sin(yaw) + dy * cos(yaw)
    return ex, ey


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def compute_scene_residuals(gt_data: dict, pred_data: dict) -> np.ndarray | None:
    """
    Compute residuals for one scene.

    Returns float32 array of shape (usable_frames, 6, 2), or None if the
    scene cannot produce any usable frames.

    Decisions applied:
    - Take last 6 planning points from final_plannings[t]
    - Transform both pred and GT to ego frame of frame t
    - Discard frames where < 6 future GT frames exist (t >= nbr_samples - 6)
    - Align GT and pred by index (GT[:nbr_samples])
    """
    ego_poses = gt_data.get("ego_poses", [])
    final_plannings = pred_data.get("final_plannings", [])
    nbr_samples = pred_data.get("nbr_samples", len(final_plannings))

    # Align by index: usable frames need GT at t+1..t+6
    n_frames = min(nbr_samples, len(final_plannings), len(ego_poses))
    usable = n_frames - 6
    if usable <= 0:
        return None

    residuals = np.zeros((usable, 6, 2), dtype=np.float32)

    for t in range(usable):
        plan_pts = final_plannings[t]
        # Take last 6 points
        plan_last6 = plan_pts[-6:]
        if len(plan_last6) < 6:
            return None  # This frame has fewer than 6 plan points → skip scene

        ego_origin = ego_poses[t]["translation"][:2]
        ego_yaw = float(ego_poses[t]["yaw"])

        for k in range(6):
            pred_global = plan_last6[k]
            gt_global = ego_poses[t + k + 1]["translation"][:2]

            pred_ego = ego_transform(pred_global, ego_origin, ego_yaw)
            gt_ego = ego_transform(gt_global, ego_origin, ego_yaw)

            residuals[t, k, 0] = pred_ego[0] - gt_ego[0]
            residuals[t, k, 1] = pred_ego[1] - gt_ego[1]

    return residuals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scene_pairs(gt_dir: Path, pred_dir: Path) -> list[tuple[str, dict, dict]]:
    """Load all pred scenes that have a matching GT file."""
    pred_files = sorted(pred_dir.glob("scene-*.json"))
    pairs = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [skip] no GT for {pred_path.name}")
            continue
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        pred_data = json.loads(pred_path.read_text(encoding="utf-8"))
        scene_name = pred_data.get("scene_name", pred_path.stem)
        pairs.append((scene_name, gt_data, pred_data))
    return pairs


def build_residual_matrix(
    pairs: list[tuple[str, dict, dict]],
) -> tuple[list[str], np.ndarray, int]:
    """
    Compute per-scene residuals and assemble into matrix R ∈ R^{N × D}.

    D = min_usable_frames × 6 × 2, where min_usable_frames is the minimum
    usable frame count across all valid scenes (truncate, no padding).

    Returns: (scene_names, R, min_frames)
    """
    scene_names: list[str] = []
    scene_residuals: list[np.ndarray] = []

    for scene_name, gt_data, pred_data in pairs:
        r = compute_scene_residuals(gt_data, pred_data)
        if r is None:
            print(f"  [skip] {scene_name}: insufficient data")
            continue
        scene_names.append(scene_name)
        scene_residuals.append(r)

    if not scene_residuals:
        raise RuntimeError("No valid scene pairs found.")

    usable_counts = [r.shape[0] for r in scene_residuals]
    min_frames = min(usable_counts)
    print(
        f"  Usable frames — min: {min_frames}, "
        f"max: {max(usable_counts)}, "
        f"scenes: {len(scene_residuals)}"
    )

    flat_dim = min_frames * 6 * 2
    R = np.zeros((len(scene_residuals), flat_dim), dtype=np.float32)
    for i, r in enumerate(scene_residuals):
        R[i] = r[:min_frames].flatten()

    return scene_names, R, min_frames


# ---------------------------------------------------------------------------
# PCA scree
# ---------------------------------------------------------------------------


def plot_scree(evr: np.ndarray, output_path: Path) -> None:
    cumulative = np.cumsum(evr)
    n = len(evr)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(1, n + 1), evr * 100, alpha=0.55, color="#4c72b0", label="Per-component")
    ax.plot(range(1, n + 1), cumulative * 100, "r-o", markersize=3, linewidth=1.2, label="Cumulative")
    ax.axhline(90, color="#555", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot — Planning Residual Field")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_cumulative_variance_table(evr: np.ndarray, limit: int = 50) -> None:
    cumulative = np.cumsum(evr)
    threshold_marked = False
    print(f"\n  {'k':>4}  {'cumulative':>12}  {'per-comp':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")
    for k, (cum, per) in enumerate(zip(cumulative, evr), start=1):
        marker = ""
        if not threshold_marked and cum >= 0.90:
            marker = "  ← 90%"
            threshold_marked = True
        print(f"  {k:>4}  {cum*100:>11.2f}%  {per*100:>9.2f}%{marker}")
        if k >= limit:
            remaining = len(evr) - limit
            if remaining > 0:
                print(f"  ... ({remaining} more components not shown)")
            break


# ---------------------------------------------------------------------------
# SOM training
# ---------------------------------------------------------------------------


def train_som(
    Z: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    sigma: float,
    learning_rate: float,
    num_iterations: int,
    random_state: int,
) -> MiniSom:
    som = MiniSom(
        x=grid_rows,
        y=grid_cols,
        input_len=Z.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function="gaussian",
        topology="rectangular",
        random_seed=random_state,
    )
    som.random_weights_init(Z)
    som.train(Z, num_iterations, verbose=True)
    return som


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------


def save_projection(
    scene_names: list[str],
    bmu_coords: list[tuple[int, int]],
    quantization_error: float,
    topographic_error: float,
    config: dict,
    output_dir: Path,
) -> None:
    scenes = [
        {
            "scene_name": name,
            "som_row": int(row),
            "som_col": int(col),
            "node_id": f"{row}_{col}",
        }
        for name, (row, col) in zip(scene_names, bmu_coords)
    ]
    out = {
        "meta": {
            "random_state": config["random_state"],
            "som_grid_rows": config["som_grid_rows"],
            "som_grid_cols": config["som_grid_cols"],
            "som_sigma": config["som_sigma"],
            "som_learning_rate": config["som_learning_rate"],
            "som_num_iterations": config["som_num_iterations"],
            "pca_n_components": config["pca_n_components"],
            "quantization_error": round(quantization_error, 6),
            "topographic_error": round(topographic_error, 6),
        },
        "scenes": scenes,
    }
    path = output_dir / "som_scene_projection.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path}")


def save_node_prototypes(
    som: MiniSom,
    pca: PCA,
    scaler: StandardScaler,
    min_frames: int,
    config: dict,
    output_dir: Path,
) -> None:
    """
    For each SOM node, inverse-transform its prototype vector back through
    PCA and StandardScaler to produce a (min_frames, 6, 2) typical residual field
    in ego coordinates.
    """
    grid_rows = config["som_grid_rows"]
    grid_cols = config["som_grid_cols"]
    weights = som.get_weights()  # (grid_rows, grid_cols, n_components)

    nodes = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            proto_pca = weights[i, j].reshape(1, -1)           # (1, n_components)
            proto_scaled = pca.inverse_transform(proto_pca)     # (1, flat_dim)
            proto_raw = scaler.inverse_transform(proto_scaled)  # (1, flat_dim)
            residual_field = proto_raw[0].reshape(min_frames, 6, 2).tolist()
            nodes.append({
                "row": i,
                "col": j,
                "node_id": f"{i}_{j}",
                "residual_field": residual_field,
            })

    out = {
        "meta": {
            "min_frames": min_frames,
            "planning_steps": 6,
            "xy_dims": 2,
            "coordinate_system": "ego",
            "description": (
                "residual_field[t][k] = [pred_ego_x - gt_ego_x, pred_ego_y - gt_ego_y] "
                "for frame t, planning step k+1 (1..6), in meters"
            ),
        },
        "nodes": nodes,
    }
    path = output_dir / "som_node_prototypes.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Grid density visualization
# ---------------------------------------------------------------------------


def save_density_plot(
    scene_names: list[str],
    bmu_coords: list[tuple[int, int]],
    grid_rows: int,
    grid_cols: int,
    output_dir: Path,
) -> None:
    """
    Render an 8×8 SOM grid heatmap where cell color encodes scene count,
    and each cell shows its count + abbreviated scene IDs.
    """
    # Build (grid_rows, grid_cols) count matrix and per-cell scene lists
    density = np.zeros((grid_rows, grid_cols), dtype=int)
    cell_scenes: dict[tuple[int, int], list[str]] = {}
    for name, (row, col) in zip(scene_names, bmu_coords):
        density[row, col] += 1
        cell_scenes.setdefault((row, col), []).append(name)

    max_count = int(density.max()) or 1

    fig, ax = plt.subplots(figsize=(grid_cols * 1.4, grid_rows * 1.4), dpi=130)
    im = ax.imshow(density, cmap="YlOrRd", vmin=0, vmax=max_count, aspect="equal")

    # Annotate each cell
    for r in range(grid_rows):
        for c in range(grid_cols):
            count = density[r, c]
            # Choose text color for contrast
            text_color = "white" if count > max_count * 0.55 else "black"

            if count == 0:
                ax.text(c, r, "—", ha="center", va="center",
                        fontsize=7, color="#aaa")
                continue

            # Abbreviate scene IDs: "scene-0003" → "0003"
            ids = cell_scenes[(r, c)]
            short_ids = [s.replace("scene-", "") for s in ids]

            # Show count large, scene IDs small below
            ax.text(c, r - 0.18, str(count),
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

            label = ", ".join(short_ids) if count <= 3 else ", ".join(short_ids[:3]) + "…"
            ax.text(c, r + 0.28, label,
                    ha="center", va="center",
                    fontsize=4.5, color=text_color, linespacing=1.2)

    # Grid lines
    for x in np.arange(-0.5, grid_cols, 1):
        ax.axvline(x, color="white", linewidth=0.6)
    for y in np.arange(-0.5, grid_rows, 1):
        ax.axhline(y, color="white", linewidth=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Scene count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.set_xticklabels([str(c) for c in range(grid_cols)], fontsize=8)
    ax.set_yticklabels([str(r) for r in range(grid_rows)], fontsize=8)
    ax.set_xlabel("SOM column", fontsize=9)
    ax.set_ylabel("SOM row", fontsize=9)
    ax.set_title(
        f"SOM Grid Density — {len(scene_names)} scenes on {grid_rows}×{grid_cols} nodes\n"
        f"empty nodes: {int((density == 0).sum())}  |  "
        f"max occupancy: {max_count}",
        fontsize=10,
    )

    fig.tight_layout()
    path = output_dir / "som_grid_density.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    config = load_config()

    gt_dir = Path(config["gt_dir"])
    pred_dir = Path(config["pred_dir"])
    output_dir = Path(config.get("output_dir", "outputs/som"))
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(config.get("random_state", 42))
    som_grid_rows = int(config.get("som_grid_rows", 8))
    som_grid_cols = int(config.get("som_grid_cols", 8))
    som_sigma = float(config.get("som_sigma", 4.0))
    som_lr = float(config.get("som_learning_rate", 0.5))
    som_iters = int(config.get("som_num_iterations", 150000))
    pca_n_components = config.get("pca_n_components", None)

    # --- Load data ---
    print("Loading scene pairs...")
    pairs = load_scene_pairs(gt_dir, pred_dir)
    print(f"  Found {len(pairs)} scene pairs")

    # --- Residuals ---
    print("\nComputing residuals...")
    scene_names, R, min_frames = build_residual_matrix(pairs)
    print(f"  Residual matrix shape: {R.shape}  (N × {min_frames}×6×2 flattened)")

    # --- Normalize ---
    print("\nNormalizing (StandardScaler)...")
    scaler = StandardScaler()
    R_std = scaler.fit_transform(R)

    # --- PCA ---
    print("\nRunning PCA (full, for scree)...")
    max_components = min(R_std.shape[0] - 1, R_std.shape[1])
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(R_std)

    scree_path = output_dir / "scree_plot.png"
    plot_scree(pca_full.explained_variance_ratio_, scree_path)
    print_cumulative_variance_table(pca_full.explained_variance_ratio_)

    if pca_n_components is None:
        print(
            f"\n  [Phase 1 complete] pca_n_components is null.\n"
            f"  Review: {scree_path.resolve()}\n"
            f"  Then set 'pca_n_components' in configs/som_config.json and re-run."
        )
        return 0

    # --- PCA with chosen n_components ---
    n_components = int(pca_n_components)
    print(f"\n  Projecting to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(R_std)
    cum_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"  Z shape: {Z.shape}, cumulative variance: {cum_var*100:.2f}%")

    # --- SOM ---
    print(f"\nTraining SOM ({som_grid_rows}×{som_grid_cols}, "
          f"sigma={som_sigma}, lr={som_lr}, iters={som_iters})...")
    som = train_som(
        Z=Z.astype(np.float64),
        grid_rows=som_grid_rows,
        grid_cols=som_grid_cols,
        sigma=som_sigma,
        learning_rate=som_lr,
        num_iterations=som_iters,
        random_state=random_state,
    )

    bmu_coords = [som.winner(z) for z in Z]
    q_error = som.quantization_error(Z)
    t_error = som.topographic_error(Z)
    print(f"  Quantization error:  {q_error:.6f}")
    print(f"  Topographic error:   {t_error:.6f}")

    # --- Save outputs ---
    full_config = {**config, "pca_n_components": n_components, "random_state": random_state,
                   "som_grid_rows": som_grid_rows, "som_grid_cols": som_grid_cols,
                   "som_sigma": som_sigma, "som_learning_rate": som_lr,
                   "som_num_iterations": som_iters}
    print("\nSaving outputs...")
    save_projection(
        scene_names=scene_names,
        bmu_coords=bmu_coords,
        quantization_error=q_error,
        topographic_error=t_error,
        config=full_config,
        output_dir=output_dir,
    )
    save_node_prototypes(
        som=som,
        pca=pca,
        scaler=scaler,
        min_frames=min_frames,
        config=full_config,
        output_dir=output_dir,
    )
    save_density_plot(
        scene_names=scene_names,
        bmu_coords=bmu_coords,
        grid_rows=som_grid_rows,
        grid_cols=som_grid_cols,
        output_dir=output_dir,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
