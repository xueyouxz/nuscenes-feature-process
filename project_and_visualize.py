from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

MODALITY_SPECS = {
    "agent": ("features_agent_raw.npy", 17),
    "map": ("features_map_raw.npy", 8),
    "ego": ("features_ego_raw.npy", 9),
}
DEFAULT_CONFIG_PATH = Path("configs/projection_config.json")
CONFIG_ENV_VAR = "NUSCENES_PROJECT_CONFIG"


def resolve_config_path() -> Path:
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CONFIG_PATH


def load_project_config() -> dict:
    config_path = resolve_config_path()
    if not config_path.exists():
        raise RuntimeError(f"Projection config file not found: {config_path}")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse projection config JSON: {config_path}") from exc


def resolve_runtime_settings(cfg: dict) -> dict:
    input_dir = Path(cfg.get("input_dir", "outputs"))
    plots_dir_cfg = cfg.get("plots_dir")
    plots_dir = Path(plots_dir_cfg) if plots_dir_cfg else input_dir / "plots"

    modalities = cfg.get("modalities", ["agent", "map", "ego"])
    if not isinstance(modalities, list) or not modalities:
        raise RuntimeError("Projection config 'modalities' must be a non-empty list.")
    invalid = [m for m in modalities if m not in MODALITY_SPECS]
    if invalid:
        raise RuntimeError(f"Unsupported modalities in config: {invalid}")

    projection = str(cfg.get("projection", "umap")).lower()
    if projection == "tsne":
        projection = "tsn"
    if projection not in {"umap", "tsn", "pca"}:
        raise RuntimeError(
            "Projection config 'projection' must be one of: ['umap', 'tsn', 'pca']."
        )

    return {
        "input_dir": input_dir,
        "plots_dir": plots_dir,
        "modalities": modalities,
        "projection": projection,
        "random_state": int(cfg.get("random_state", 42)),
        "disable_review_gate": bool(cfg.get("disable_review_gate", False)),
        "disable_numba_jit": bool(cfg.get("disable_numba_jit", False)),
        "fallback_on_umap_error": bool(cfg.get("fallback_on_umap_error", False)),
        "umap_n_neighbors": int(cfg.get("umap_n_neighbors", 20)),
        "umap_min_dist": float(cfg.get("umap_min_dist", 0.1)),
        "tsn_perplexity": float(cfg.get("tsn_perplexity", 10.0)),
        "tsn_learning_rate": float(cfg.get("tsn_learning_rate", 200.0)),
    }


def load_metadata(input_dir: Path) -> pd.DataFrame:
    try:
        metadata = pd.read_parquet(input_dir / "metadata.parquet")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to read metadata.parquet. Install 'pyarrow' or 'fastparquet' for parquet support."
        ) from exc

    required_cols = {"scene_token", "scene_name", "location", "log_name"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {sorted(missing)}")
    return metadata


def load_modality(input_dir: Path, modality: str, rows: int) -> np.ndarray:
    filename, dim = MODALITY_SPECS[modality]
    arr = np.load(input_dir / filename)
    if arr.shape != (rows, dim):
        raise RuntimeError(
            f"{modality} shape mismatch: {arr.shape}, expected ({rows}, {dim})"
        )
    return arr.astype(np.float32)


def normalize_location_label(value: str) -> str:
    text = str(value).lower()
    if "boston" in text:
        return "Boston"
    if "singapore" in text:
        return "Singapore"
    return "Other"


def run_umap(
    features_scaled: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    fallback_on_error: bool,
):
    try:
        import umap

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*n_jobs value .* overridden .* setting random_state.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*n_neighbors is larger than the dataset size.*",
                category=UserWarning,
            )

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            return reducer.fit_transform(features_scaled)
    except ImportError as exc:
        if not fallback_on_error:
            raise RuntimeError(
                "Missing dependency 'umap-learn'. Install requirements first."
            ) from exc
    except Exception as exc:  # noqa: BLE001
        if not fallback_on_error:
            raise RuntimeError(
                "UMAP projection failed. You can retry with --disable-numba-jit "
                "or --fallback-on-umap-error."
            ) from exc

    return PCA(n_components=2, random_state=random_state).fit_transform(features_scaled)


def run_tsn(
    features_scaled: np.ndarray,
    perplexity: float,
    learning_rate: float,
    random_state: int,
) -> np.ndarray:
    n_samples = features_scaled.shape[0]
    if n_samples < 2:
        raise RuntimeError("t-SNE projection needs at least 2 samples.")
    # sklearn TSNE requires perplexity < n_samples.
    safe_perplexity = max(1.0, min(float(perplexity), float(n_samples - 1)))
    reducer = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    )
    return reducer.fit_transform(features_scaled)


def separation_signal(embedding: np.ndarray, labels: np.ndarray) -> float:
    boston = embedding[labels == "Boston"]
    singapore = embedding[labels == "Singapore"]
    if len(boston) < 2 or len(singapore) < 2:
        return 0.0

    c1 = np.mean(boston, axis=0)
    c2 = np.mean(singapore, axis=0)
    inter = float(np.linalg.norm(c1 - c2))

    s1 = float(np.mean(np.linalg.norm(boston - c1, axis=1)))
    s2 = float(np.mean(np.linalg.norm(singapore - c2, axis=1)))
    intra = max((s1 + s2) / 2.0, 1e-6)
    return inter / intra


def save_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    modality: str,
    projection_title: str,
    params_title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    colors = {"Boston": "#1f77b4", "Singapore": "#ff7f0e", "Other": "#9e9e9e"}
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)

    for label in ("Boston", "Singapore", "Other"):
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=20,
                alpha=0.85,
                c=colors[label],
                label=label,
                edgecolors="none",
            )

    ax.set_title(f"{modality} {projection_title} | {params_title}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_projection_csv(
    scene_names: np.ndarray,
    tsn_embedding: np.ndarray,
    umap_embedding: np.ndarray,
    output_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "scene-name": scene_names,
            "tsne-comp1": tsn_embedding[:, 0],
            "tsne-comp2": tsn_embedding[:, 1],
            "umap-comp1": umap_embedding[:, 0],
            "umap-comp2": umap_embedding[:, 1],
        }
    )
    df.to_csv(output_path, index=False)


def main() -> int:
    cfg = load_project_config()
    settings = resolve_runtime_settings(cfg)

    if settings["disable_numba_jit"]:
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    input_dir = settings["input_dir"]
    plots_dir = settings["plots_dir"]
    images_dir = plots_dir / "images"
    tables_dir = plots_dir / "tables"

    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(input_dir)
    n_rows = len(metadata)
    scene_names = metadata["scene_name"].to_numpy()
    location_labels = metadata["location"].map(normalize_location_label).to_numpy()

    total_plots = 0
    best_signal = 0.0
    for modality in settings["modalities"]:
        raw_features = load_modality(input_dir, modality, n_rows)
        scaled = RobustScaler().fit_transform(raw_features)
        umap_embedding = run_umap(
            features_scaled=scaled,
            n_neighbors=settings["umap_n_neighbors"],
            min_dist=settings["umap_min_dist"],
            random_state=settings["random_state"],
            fallback_on_error=settings["fallback_on_umap_error"],
        )
        tsn_embedding = run_tsn(
            features_scaled=scaled,
            perplexity=settings["tsn_perplexity"],
            learning_rate=settings["tsn_learning_rate"],
            random_state=settings["random_state"],
        )

        save_projection_csv(
            scene_names=scene_names,
            tsn_embedding=tsn_embedding,
            umap_embedding=umap_embedding,
            output_path=tables_dir / f"{modality}_projection.csv",
        )

        save_scatter(
            embedding=umap_embedding,
            labels=location_labels,
            modality=modality,
            projection_title="UMAP",
            params_title=(
                f"nn={settings['umap_n_neighbors']}, "
                f"min_dist={settings['umap_min_dist']:.2f}"
            ),
            x_label="UMAP-1",
            y_label="UMAP-2",
            output_path=images_dir / f"{modality}_umap.png",
        )
        total_plots += 1

        safe_perplexity = max(1.0, min(settings["tsn_perplexity"], float(n_rows - 1)))
        save_scatter(
            embedding=tsn_embedding,
            labels=location_labels,
            modality=modality,
            projection_title="TSN",
            params_title=(
                f"perplexity={safe_perplexity:.2f}, "
                f"lr={settings['tsn_learning_rate']:.1f}"
            ),
            x_label="TSN-1",
            y_label="TSN-2",
            output_path=images_dir / f"{modality}_tsn.png",
        )
        total_plots += 1

        if settings["projection"] == "umap":
            embedding = umap_embedding
            projection_title = "UMAP"
            params_title = (
                f"nn={settings['umap_n_neighbors']}, "
                f"min_dist={settings['umap_min_dist']:.2f}"
            )
            x_label = "UMAP-1"
            y_label = "UMAP-2"
            out_name = f"{modality}_umap.png"
        elif settings["projection"] == "tsn":
            embedding = tsn_embedding
            projection_title = "TSN"
            params_title = (
                f"perplexity={safe_perplexity:.2f}, "
                f"lr={settings['tsn_learning_rate']:.1f}"
            )
            x_label = "TSN-1"
            y_label = "TSN-2"
            out_name = f"{modality}_tsn.png"
        else:
            embedding = PCA(n_components=2, random_state=settings["random_state"]).fit_transform(
                scaled
            )
            projection_title = "PCA"
            params_title = "deterministic"
            x_label = "PCA-1"
            y_label = "PCA-2"
            out_name = f"{modality}_pca.png"

        score = separation_signal(embedding, location_labels)
        best_signal = max(best_signal, score)

        if settings["projection"] == "pca":
            save_scatter(
                embedding=embedding,
                labels=location_labels,
                modality=modality,
                projection_title=projection_title,
                params_title=params_title,
                x_label=x_label,
                y_label=y_label,
                output_path=images_dir / out_name,
            )
            total_plots += 1

    print(f"Rows: {n_rows}")
    print(f"Modalities: {settings['modalities']}")
    print(f"Projection: {settings['projection']}")
    print(f"Plots written: {total_plots}")
    print(f"Images dir: {images_dir.resolve()}")
    print(f"Tables dir: {tables_dir.resolve()}")
    print(f"Best location-separation signal: {best_signal:.4f}")

    if (not settings["disable_review_gate"]) and best_signal < 0.25:
        marker = tables_dir / "REVIEW_REQUIRED.txt"
        marker.write_text(
            "Configured projection showed weak Boston/Singapore separation.\n"
            "Review feature construction before continuing downstream tasks.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Review gate failed (best signal={best_signal:.4f} < 0.25). "
            f"See {marker}."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
