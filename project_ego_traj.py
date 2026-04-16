"""
project_ego_traj.py — Ego trajectory dimensionality reduction experiment.

Pipeline:
  1. Load nuScenes (val split, 150 scenes)
  2. Extract raw ego (x, y) trajectories: 34 frames → (N, 34, 2)
  3. Normalize: translate t=0 to origin; rotate initial heading to +x axis
     using cumulative displacement of first `rotation_cum_frames` frames;
     skip rotation if displacement < rotation_threshold_m (scene logged).
  4. Flatten + StandardScaler → X_std (N, 68)
  5. Full PCA → auto k via 90% cumulative variance → X_pca (N, k)
     Outputs: scree_plot.png, pc_directions.png
  6. k-means (k=5) on X_pca → shared cluster color labels for all methods
  7. Run 5 DR methods (single run each, fixed hyperparameters):
       E: PCA-2D  on X_std       (baseline, deterministic)
       A: X_pca → t-SNE          (perplexity=30, PCA init)
       B: X_pca → UMAP           (n_neighbors=15, min_dist=0.1)
       C: X_std → t-SNE          (same params as A)
       D: X_std → UMAP           (same params as B)
  8. For each method: save scatter plot (PNG) + CSV to output_dir

Config: configs/ego_traj_config.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG_PATH = Path("configs/ego_traj_config.json")

# Five visually distinct colors for k=5 clusters
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {DEFAULT_CONFIG_PATH}")
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_settings(cfg: dict) -> dict:
    dataroot = Path(cfg.get("nuscenes_dataroot", "~/Data/nuscenes")).expanduser()
    return {
        "version": cfg.get("nuscenes_version", "v1.0-trainval"),
        "dataroot": dataroot,
        "split": cfg.get("nuscenes_split", "val"),
        "output_dir": Path(cfg.get("output_dir", "outputs/ego_traj")),
        "n_frames": int(cfg.get("n_frames", 34)),
        "rotation_cum_frames": int(cfg.get("rotation_cum_frames", 3)),
        "rotation_threshold_m": float(cfg.get("rotation_threshold_m", 0.1)),
        "pca_variance_threshold": float(cfg.get("pca_variance_threshold", 0.90)),
        "kmeans_k": int(cfg.get("kmeans_k", 5)),
        "tsne_perplexity": float(cfg.get("tsne_perplexity", 30.0)),
        "umap_n_neighbors": int(cfg.get("umap_n_neighbors", 15)),
        "umap_min_dist": float(cfg.get("umap_min_dist", 0.1)),
        "random_state": int(cfg.get("random_state", 42)),
    }


# ---------------------------------------------------------------------------
# nuScenes helpers
# ---------------------------------------------------------------------------


def load_nuscenes(version: str, dataroot: str):
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'nuscenes-devkit'.") from exc
    return NuScenes(version=version, dataroot=dataroot, verbose=True)


def get_split_scenes(nusc, split: str) -> list[dict]:
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'nuscenes-devkit'.") from exc

    splits = create_splits_scenes()
    if split not in splits:
        raise RuntimeError(f"Unknown split '{split}'. Available: {sorted(splits)}")
    split_names = set(splits[split])
    return sorted(
        (s for s in nusc.scene if s["name"] in split_names),
        key=lambda s: s["name"],
    )


def collect_sample_tokens(nusc, scene: dict, n_frames: int) -> list[str]:
    tokens: list[str] = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    if len(tokens) < n_frames:
        raise RuntimeError(
            f"Scene '{scene['name']}' has only {len(tokens)} keyframes, need {n_frames}."
        )
    return tokens[:n_frames]


def get_ego_xy(nusc, sample: dict) -> np.ndarray:
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(pose["translation"][:2], dtype=np.float32)


def extract_trajectory(nusc, scene: dict, n_frames: int) -> np.ndarray:
    """Return raw global-frame (x, y) positions, shape (n_frames, 2)."""
    tokens = collect_sample_tokens(nusc, scene, n_frames)
    return np.stack([get_ego_xy(nusc, nusc.get("sample", t)) for t in tokens])


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------


def normalize_trajectory(
    xy: np.ndarray,
    cum_frames: int,
    threshold_m: float,
) -> tuple[np.ndarray, bool]:
    """
    Translate so that frame 0 is at the origin.
    Rotate so that the cumulative displacement over the first `cum_frames`
    frames aligns to the +x axis.
    Returns (normalized_xy, rotation_applied).
    """
    xy = xy - xy[0]  # translate; now xy[0] == [0, 0]

    cum_disp = xy[cum_frames]  # vector from origin to frame `cum_frames`
    dist = float(np.linalg.norm(cum_disp))

    if dist < threshold_m:
        return xy, False  # near-stationary start: skip rotation

    # Rotation matrix that maps cum_disp direction → +x axis
    cos_a = cum_disp[0] / dist
    sin_a = cum_disp[1] / dist
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)
    return (R @ xy.T).T, True


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------


def run_pca_analysis(
    X_std: np.ndarray,
    variance_threshold: float,
    n_frames: int,
    output_dir: Path,
    random_state: int,
) -> tuple[np.ndarray, int]:
    """
    Fit full PCA. Save scree_plot.png and pc_directions.png.
    Auto-select k via cumulative variance threshold.
    Return (X_pca, k).
    """
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_std)
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # Auto k: smallest number of components that meet the threshold
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k = min(k, X_std.shape[1])

    # --- Scree plot ---
    n_show = min(30, len(evr))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    axes[0].bar(range(1, n_show + 1), evr[:n_show], color="#1f77b4", alpha=0.8)
    axes[0].axvline(k, color="red", linestyle="--", linewidth=1.2, label=f"k={k}")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")
    axes[0].legend(frameon=False)

    axes[1].plot(range(1, n_show + 1), cumvar[:n_show], "-o", markersize=3, color="#ff7f0e")
    axes[1].axhline(variance_threshold, color="gray", linestyle="--", linewidth=1,
                    label=f"{variance_threshold:.0%} threshold")
    axes[1].axvline(k, color="red", linestyle="--", linewidth=1.2, label=f"k={k}")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend(frameon=False)

    fig.suptitle(f"PCA Analysis  —  auto k={k}  ({variance_threshold:.0%} threshold)")
    fig.tight_layout()
    fig.savefig(output_dir / "scree_plot.png")
    plt.close(fig)

    # --- PC direction shapes (PC1, PC2, PC3) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
    for i, ax in enumerate(axes):
        traj = pca_full.components_[i].reshape(n_frames, 2)
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, linewidth=1.5)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="green", zorder=5, s=40, label="start")
        ax.set_title(f"PC{i + 1}  ({evr[i]:.1%} var)")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("x (forward)")
        ax.set_ylabel("y (lateral)")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Principal Component Direction Shapes (reshaped to trajectory)")
    fig.tight_layout()
    fig.savefig(output_dir / "pc_directions.png")
    plt.close(fig)

    # Fit k-component PCA for downstream use
    pca_k = PCA(n_components=k, random_state=random_state)
    X_pca = pca_k.fit_transform(X_std)

    print(f"PCA: k={k} components, cumulative variance = {cumvar[k - 1]:.3f}")
    return X_pca, k


# ---------------------------------------------------------------------------
# DR methods
# ---------------------------------------------------------------------------


def method_e(X_std: np.ndarray, random_state: int) -> np.ndarray:
    return PCA(n_components=2, random_state=random_state).fit_transform(X_std)


def method_tsne(X: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    n = X.shape[0]
    safe_perplexity = max(1.0, min(perplexity, float(n - 1)))
    return TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        random_state=random_state,
    ).fit_transform(X)


def method_umap(X: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'umap-learn'.") from exc
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        ).fit_transform(X)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_scatter(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    kmeans_k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for c in range(kmeans_k):
        mask = cluster_labels == c
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=22,
            alpha=0.85,
            c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            label=f"Cluster {c + 1}",
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_csv(
    embedding: np.ndarray,
    scene_tokens: list[str],
    scene_names: list[str],
    cluster_labels: np.ndarray,
    output_path: Path,
) -> None:
    pd.DataFrame(
        {
            "scene_token": scene_tokens,
            "scene_name": scene_names,
            "cluster": cluster_labels + 1,
            "dim1": embedding[:, 0],
            "dim2": embedding[:, 1],
        }
    ).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = load_config()
    s = resolve_settings(cfg)

    output_dir = s["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load nuScenes
    nusc = load_nuscenes(s["version"], str(s["dataroot"]))
    scenes = get_split_scenes(nusc, s["split"])
    print(f"Split '{s['split']}': {len(scenes)} scenes")

    # Extract and normalize trajectories
    skipped_rotation: list[str] = []
    trajectories: list[np.ndarray] = []
    scene_tokens: list[str] = []
    scene_names: list[str] = []

    for scene in scenes:
        try:
            xy = extract_trajectory(nusc, scene, s["n_frames"])
            xy_norm, rotated = normalize_trajectory(
                xy,
                cum_frames=s["rotation_cum_frames"],
                threshold_m=s["rotation_threshold_m"],
            )
        except RuntimeError as exc:
            print(f"WARNING: Skipping scene '{scene['name']}': {exc}")
            continue

        trajectories.append(xy_norm)
        scene_tokens.append(scene["token"])
        scene_names.append(scene["name"])
        if not rotated:
            skipped_rotation.append(scene["name"])

    N = len(trajectories)
    print(f"Valid scenes: {N}")
    if skipped_rotation:
        print(f"Rotation skipped (near-stationary start) for: {skipped_rotation}")

    # Normalized trajectory overview
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    for xy in trajectories:
        ax.plot(xy[:, 0], xy[:, 1], alpha=0.15, linewidth=0.8, color="#1f77b4")
    ax.set_title(f"Normalized Ego Trajectories  (N={N})")
    ax.set_xlabel("x (forward)")
    ax.set_ylabel("y (lateral)")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories_normalized.png")
    plt.close(fig)

    # Feature matrix
    X_raw = np.stack([t.flatten() for t in trajectories])  # (N, n_frames*2)
    X_std = StandardScaler().fit_transform(X_raw)

    # PCA analysis → X_pca
    X_pca, k = run_pca_analysis(
        X_std,
        variance_threshold=s["pca_variance_threshold"],
        n_frames=s["n_frames"],
        output_dir=output_dir,
        random_state=s["random_state"],
    )

    # k-means on X_pca (shared labels for all methods)
    cluster_labels = KMeans(
        n_clusters=s["kmeans_k"],
        random_state=s["random_state"],
        n_init=10,
    ).fit_predict(X_pca)

    # Run 5 DR methods
    print("Running DR methods...")
    rs = s["random_state"]
    methods = [
        ("E_pca",      "PCA-2D (baseline)",  "PC-1",    "PC-2",    method_e(X_std, rs)),
        ("A_pca_tsne", "PCA → t-SNE",        "t-SNE-1", "t-SNE-2", method_tsne(X_pca, s["tsne_perplexity"], rs)),
        ("B_pca_umap", "PCA → UMAP",         "UMAP-1",  "UMAP-2",  method_umap(X_pca, s["umap_n_neighbors"], s["umap_min_dist"], rs)),
        ("C_raw_tsne", "Raw → t-SNE",        "t-SNE-1", "t-SNE-2", method_tsne(X_std, s["tsne_perplexity"], rs)),
        ("D_raw_umap", "Raw → UMAP",         "UMAP-1",  "UMAP-2",  method_umap(X_std, s["umap_n_neighbors"], s["umap_min_dist"], rs)),
    ]

    for method_id, title, xlabel, ylabel, embedding in methods:
        save_scatter(
            embedding=embedding,
            cluster_labels=cluster_labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            output_path=output_dir / f"{method_id}_scatter.png",
            kmeans_k=s["kmeans_k"],
        )
        save_csv(
            embedding=embedding,
            scene_tokens=scene_tokens,
            scene_names=scene_names,
            cluster_labels=cluster_labels,
            output_path=output_dir / f"{method_id}.csv",
        )
        print(f"  Saved: {method_id}")

    print(f"\nDone. Outputs written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
