"""
project_ego_som.py — Ego trajectory kinematic SOM projection.

Goal: cluster scenes by *driving behaviour pattern*, not trajectory geometry.

Pipeline:
  1. Load GT ego_poses for each scene
  2. Coordinate alignment: translate origin to pos[0], rotate by yaw[0]
  3. Compute kinematic sequence [v, a, yaw_rate] over first 34 frames
       - v[t]        = |pos[t+1] - pos[t]| / dt          (speed, m/s)
       - yaw_rate[t] = wrap(yaw[t+1] - yaw[t]) / dt      (rad/s)
       - a[t]        = (v[t+1] - v[t]) / dt               (m/s²)
       requires ego_poses[0..35] (36 positions) → tensor 34×3
  4. Flatten 34×3 → 102-dim vector per scene
  5. StandardScaler (per-column) → PCA (scree mode or fixed k) → SOM 6×6
  6. Output: ego_som_scene_projection.json + ego_som_node_prototypes.json
             + scree_plot.png + som_grid_density.png

Two-phase run:
  Phase 1: pca_n_components=null → scree_plot.png + cumulative table → exit
  Phase 2: pca_n_components=<k> → full pipeline
"""

from __future__ import annotations

import json
import sys
from math import atan2, cos, sin, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG_PATH = Path("configs/ego_som_config.json")

# Number of ego_poses entries needed to produce n_kinematic_frames of [v, a, yaw_rate].
# v needs pos[0..n], yaw_rate needs yaw[0..n]  → n+1 values each (n frames)
# a needs v[0..n]                               → n values (need one extra v)
# To get 34 a values: need v[0..34] (35 v values) → pos[0..35] (36 positions)
_EXTRA_POSES_NEEDED = 2  # beyond n_kinematic_frames


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {DEFAULT_CONFIG_PATH}")
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Kinematic feature extraction
# ---------------------------------------------------------------------------


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-π, π]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def compute_kinematics(
    ego_poses: list[dict],
    dt: float,
    n_frames: int,
) -> np.ndarray | None:
    """
    Compute aligned kinematic tensor for one scene.

    Steps:
      1. Extract positions and yaw values from ego_poses[0..n_frames+1]
      2. Translate pos[0] to origin, rotate by -yaw[0] (ego alignment)
      3. Compute v[0..n_frames] (n_frames+1 values), yaw_rate[0..n_frames-1],
         a[0..n_frames-1]; truncate all to n_frames values

    Returns float32 array of shape (n_frames, 3) = [v, a, yaw_rate], or None
    if the scene has fewer than n_frames + _EXTRA_POSES_NEEDED poses.
    """
    required = n_frames + _EXTRA_POSES_NEEDED  # 34 + 2 = 36
    if len(ego_poses) < required:
        return None

    poses = ego_poses[:required]

    # Raw global positions and yaw values
    raw_pos = np.array([[p["translation"][0], p["translation"][1]] for p in poses],
                       dtype=np.float64)
    raw_yaw = np.array([float(p["yaw"]) for p in poses], dtype=np.float64)

    # --- Coordinate alignment ---
    origin = raw_pos[0]
    yaw_0 = raw_yaw[0]
    c, s = cos(yaw_0), sin(yaw_0)
    # Rotation matrix R(-yaw_0): global → ego at t=0
    dx = raw_pos[:, 0] - origin[0]
    dy = raw_pos[:, 1] - origin[1]
    pos_x = dx * c + dy * s
    pos_y = -dx * s + dy * c
    pos_aligned = np.stack([pos_x, pos_y], axis=1)  # (required, 2)

    # Aligned yaw (subtract yaw_0, wrap)
    yaw_aligned = _wrap_angle(raw_yaw - yaw_0)

    # --- Kinematic channels ---
    # Speed: |Δpos| / dt,  length = required - 1 = n_frames + 1
    diff_pos = np.diff(pos_aligned, axis=0)
    v_full = np.sqrt((diff_pos ** 2).sum(axis=1)) / dt  # (n_frames+1,)

    # Acceleration: Δv / dt,  length = n_frames
    a = np.diff(v_full) / dt  # (n_frames,)

    # Yaw rate: wrap(Δyaw) / dt,  length = n_frames (use first n_frames+1 yaw)
    delta_yaw = _wrap_angle(np.diff(yaw_aligned[:n_frames + 1]))
    yaw_rate = delta_yaw / dt  # (n_frames,)

    # Truncate v to n_frames
    v = v_full[:n_frames]  # (n_frames,)

    # Stack: (n_frames, 3) = [v, a, yaw_rate]
    return np.stack([v, a, yaw_rate], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gt_scenes(gt_dir: Path) -> list[tuple[str, dict]]:
    """Load all GT scene JSON files. Returns list of (scene_name, gt_data)."""
    files = sorted(gt_dir.glob("scene-*.json"))
    scenes = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        scene_name = data.get("scene_name", f.stem)
        scenes.append((scene_name, data))
    return scenes


def build_feature_matrix(
    scenes: list[tuple[str, dict]],
    dt: float,
    n_frames: int,
) -> tuple[list[str], np.ndarray]:
    """
    Compute kinematic feature vectors for all scenes and stack into matrix.

    Returns: (scene_names, X) where X ∈ R^{N × (n_frames × 3)}
    """
    scene_names: list[str] = []
    features: list[np.ndarray] = []

    for scene_name, gt_data in scenes:
        ego_poses = gt_data.get("ego_poses", [])
        kin = compute_kinematics(ego_poses, dt=dt, n_frames=n_frames)
        if kin is None:
            print(f"  [skip] {scene_name}: insufficient ego_poses "
                  f"(need {n_frames + _EXTRA_POSES_NEEDED}, got {len(ego_poses)})")
            continue
        scene_names.append(scene_name)
        features.append(kin.flatten())

    if not features:
        raise RuntimeError("No valid scenes found.")

    X = np.stack(features, axis=0)  # (N, n_frames * 3)
    return scene_names, X


# ---------------------------------------------------------------------------
# PCA scree
# ---------------------------------------------------------------------------


def plot_scree(evr: np.ndarray, output_path: Path) -> None:
    cumulative = np.cumsum(evr)
    n = len(evr)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(1, n + 1), evr * 100, alpha=0.55, color="#2ca02c", label="Per-component")
    ax.plot(range(1, n + 1), cumulative * 100, "r-o", markersize=3,
            linewidth=1.2, label="Cumulative")
    ax.axhline(90, color="#555", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot — Ego Kinematic Features [v, a, yaw_rate]")
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
# SOM
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
            "feature": "ego_kinematic",
            "channels": ["v_m_s", "a_m_s2", "yaw_rate_rad_s"],
            "n_kinematic_frames": config["n_kinematic_frames"],
            "dt_s": config["dt"],
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
    path = output_dir / "ego_som_scene_projection.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path}")


def save_node_prototypes(
    som: MiniSom,
    pca: PCA,
    scaler: StandardScaler,
    n_frames: int,
    config: dict,
    output_dir: Path,
) -> None:
    """
    Inverse-transform each node's prototype through PCA + StandardScaler to
    recover a (n_frames, 3) kinematic curve: [v, a, yaw_rate] over time.
    """
    grid_rows = config["som_grid_rows"]
    grid_cols = config["som_grid_cols"]
    weights = som.get_weights()  # (grid_rows, grid_cols, n_pca_components)

    nodes = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            proto_pca = weights[i, j].reshape(1, -1)
            proto_scaled = pca.inverse_transform(proto_pca)
            proto_raw = scaler.inverse_transform(proto_scaled)
            # Reshape to (n_frames, 3) = [v, a, yaw_rate]
            kinematic_curve = proto_raw[0].reshape(n_frames, 3).tolist()
            nodes.append({
                "row": i,
                "col": j,
                "node_id": f"{i}_{j}",
                "kinematic_curve": kinematic_curve,
            })

    out = {
        "meta": {
            "n_frames": n_frames,
            "channels": ["v_m_s", "a_m_s2", "yaw_rate_rad_s"],
            "dt_s": config["dt"],
            "coordinate_system": "ego_aligned",
            "description": (
                "kinematic_curve[t] = [v(m/s), a(m/s²), yaw_rate(rad/s)] "
                "at time step t in ego-aligned frame (origin at pos[0], "
                "heading aligned to yaw[0])"
            ),
        },
        "nodes": nodes,
    }
    path = output_dir / "ego_som_node_prototypes.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path}")


def save_density_plot(
    scene_names: list[str],
    bmu_coords: list[tuple[int, int]],
    grid_rows: int,
    grid_cols: int,
    output_dir: Path,
) -> None:
    density = np.zeros((grid_rows, grid_cols), dtype=int)
    cell_scenes: dict[tuple[int, int], list[str]] = {}
    for name, (row, col) in zip(scene_names, bmu_coords):
        density[row, col] += 1
        cell_scenes.setdefault((row, col), []).append(name)

    max_count = int(density.max()) or 1

    fig, ax = plt.subplots(figsize=(grid_cols * 1.6, grid_rows * 1.6), dpi=130)
    im = ax.imshow(density, cmap="YlGn", vmin=0, vmax=max_count, aspect="equal")

    for r in range(grid_rows):
        for c in range(grid_cols):
            count = density[r, c]
            text_color = "white" if count > max_count * 0.55 else "black"

            if count == 0:
                ax.text(c, r, "—", ha="center", va="center", fontsize=8, color="#aaa")
                continue

            ids = cell_scenes[(r, c)]
            short_ids = [s.replace("scene-", "") for s in ids]

            ax.text(c, r - 0.18, str(count),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

            label = ", ".join(short_ids) if count <= 3 else ", ".join(short_ids[:3]) + "…"
            ax.text(c, r + 0.28, label,
                    ha="center", va="center",
                    fontsize=5, color=text_color)

    for x in np.arange(-0.5, grid_cols, 1):
        ax.axvline(x, color="white", linewidth=0.8)
    for y in np.arange(-0.5, grid_rows, 1):
        ax.axhline(y, color="white", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Scene count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.set_xticklabels([str(c) for c in range(grid_cols)], fontsize=9)
    ax.set_yticklabels([str(r) for r in range(grid_rows)], fontsize=9)
    ax.set_xlabel("SOM column", fontsize=9)
    ax.set_ylabel("SOM row", fontsize=9)
    ax.set_title(
        f"Ego Kinematic SOM — {len(scene_names)} scenes on {grid_rows}×{grid_cols} nodes\n"
        f"features: [v, a, yaw_rate]  |  "
        f"empty: {int((density == 0).sum())}  |  max: {max_count}",
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
    output_dir = Path(config.get("output_dir", "outputs/ego_som"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = float(config.get("dt", 0.5))
    n_frames = int(config.get("n_kinematic_frames", 34))
    random_state = int(config.get("random_state", 42))
    som_grid_rows = int(config.get("som_grid_rows", 6))
    som_grid_cols = int(config.get("som_grid_cols", 6))
    som_sigma = float(config.get("som_sigma", 3.0))
    som_lr = float(config.get("som_learning_rate", 0.5))
    som_iters = int(config.get("som_num_iterations", 150000))
    pca_n_components = config.get("pca_n_components", None)

    # --- Load ---
    print("Loading GT scenes...")
    scenes = load_gt_scenes(gt_dir)
    print(f"  Found {len(scenes)} scene files")

    # --- Features ---
    print(f"\nComputing kinematics (dt={dt}s, n_frames={n_frames})...")
    scene_names, X = build_feature_matrix(scenes, dt=dt, n_frames=n_frames)
    print(f"  Feature matrix: {X.shape}  ({n_frames}×3 flattened per scene)")

    # --- Normalize ---
    print("\nNormalizing (StandardScaler, per-column)...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # --- PCA scree ---
    print("\nRunning PCA (full, for scree)...")
    max_components = min(X_std.shape[0] - 1, X_std.shape[1])
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(X_std)

    scree_path = output_dir / "scree_plot.png"
    plot_scree(pca_full.explained_variance_ratio_, scree_path)
    print_cumulative_variance_table(pca_full.explained_variance_ratio_)

    if pca_n_components is None:
        print(
            f"\n  [Phase 1 complete] pca_n_components is null.\n"
            f"  Review: {scree_path.resolve()}\n"
            f"  Then set 'pca_n_components' in configs/ego_som_config.json and re-run."
        )
        return 0

    # --- PCA with chosen k ---
    n_components = int(pca_n_components)
    print(f"\n  Projecting to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X_std)
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

    # --- Save ---
    full_config = {
        **config,
        "pca_n_components": n_components,
        "random_state": random_state,
        "som_grid_rows": som_grid_rows,
        "som_grid_cols": som_grid_cols,
        "som_sigma": som_sigma,
        "som_learning_rate": som_lr,
        "som_num_iterations": som_iters,
        "n_kinematic_frames": n_frames,
        "dt": dt,
    }

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
        n_frames=n_frames,
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
