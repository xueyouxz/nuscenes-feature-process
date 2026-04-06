from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extractors import AgentFeatureExtractor, EgoFeatureExtractor, MapFeatureExtractor

DEFAULT_CONFIG_PATH = Path("configs/extraction_config.json")
CONFIG_ENV_VAR = "NUSCENES_EXTRACT_CONFIG"


def resolve_config_path() -> Path:
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CONFIG_PATH


def load_extract_config() -> dict:
    config_path = resolve_config_path()
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse config JSON: {config_path}") from exc


def resolve_runtime_settings(cfg: dict) -> dict:
    version = cfg.get("default_version")
    if not version:
        raise RuntimeError("No version provided and config missing 'default_version'.")

    split_default = cfg.get("default_split_by_version", {}).get(version)
    split = split_default
    if not split:
        raise RuntimeError(
            f"No split provided and config missing default split for version '{version}'."
        )

    dataroot_str = cfg.get("version_to_dataroot", {}).get(version)
    if not dataroot_str:
        raise RuntimeError(
            f"No dataroot provided and config missing version_to_dataroot for '{version}'."
        )
    dataroot = Path(dataroot_str).expanduser()
    if not dataroot.exists():
        raise RuntimeError(f"Resolved dataroot does not exist: {dataroot}")

    output_dir = cfg.get("output_dir", "outputs")
    keyframes = int(cfg.get("keyframes", 34))
    if keyframes <= 0:
        raise RuntimeError(f"Invalid keyframes value: {keyframes}. Must be > 0.")
    expected_scenes = cfg.get("split_to_expected_scenes", {}).get(split)
    if expected_scenes is None:
        raise RuntimeError(
            f"No expected scene count provided and config missing split_to_expected_scenes for '{split}'."
        )

    return {
        "version": version,
        "split": split,
        "dataroot": dataroot,
        "expected_scenes": int(expected_scenes),
        "output_dir": Path(output_dir),
        "keyframes": keyframes,
    }


def load_nuscenes(version: str, dataroot: str):
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'nuscenes-devkit'. Install it before running extraction."
        ) from exc

    return NuScenes(version=version, dataroot=dataroot, verbose=True)


def build_map_cache(dataroot: str, locations: list[str]) -> dict[str, object]:
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'nuscenes-devkit' map API. Install it before running extraction."
        ) from exc

    cache: dict[str, object] = {}
    for location in sorted(set(locations)):
        cache[location] = NuScenesMap(dataroot=dataroot, map_name=location)
    return cache


def get_split_scene_names(split: str) -> set[str]:
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'nuscenes-devkit'. Install it before running extraction."
        ) from exc

    splits = create_splits_scenes()
    if split not in splits:
        raise RuntimeError(f"Unknown split '{split}'. Available splits: {sorted(splits)}")
    return set(splits[split])


def get_scene_order(nusc, split_scene_names: set[str], split: str, expected_scenes: int) -> list[dict]:

    ordered = sorted(
        (scene for scene in nusc.scene if scene["name"] in split_scene_names),
        key=lambda item: item["name"],
    )

    if len(ordered) != expected_scenes:
        raise RuntimeError(
            f"Split '{split}' expected {expected_scenes} scenes but found {len(ordered)}"
        )

    return ordered


def collect_metadata_row(nusc, scene: dict) -> dict:
    log = nusc.get("log", scene["log_token"])
    return {
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "location": log.get("location"),
        "log_name": log.get("logfile"),
        # Reserved metric fields for future model overlays.
        "metric_map": np.nan,
        "metric_nde": np.nan,
        "metric_planner": np.nan,
    }


def write_metadata_parquet(metadata: pd.DataFrame, output_path: Path) -> None:
    try:
        metadata.to_parquet(output_path, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to write metadata.parquet. Install 'pyarrow' or 'fastparquet' for parquet support."
        ) from exc


def main() -> int:
    cfg = load_extract_config()
    settings = resolve_runtime_settings(cfg)

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = load_nuscenes(settings["version"], str(settings["dataroot"]))
    split_scene_names = get_split_scene_names(settings["split"])
    ordered_scenes = get_scene_order(
        nusc,
        split_scene_names=split_scene_names,
        split=settings["split"],
        expected_scenes=settings["expected_scenes"],
    )
    scene_locations = [nusc.get("log", scene["log_token"])["location"] for scene in ordered_scenes]
    map_cache = build_map_cache(str(settings["dataroot"]), scene_locations)

    agent_extractor = AgentFeatureExtractor(keyframes=settings["keyframes"])
    map_extractor = MapFeatureExtractor(keyframes=settings["keyframes"])
    ego_extractor = EgoFeatureExtractor(keyframes=settings["keyframes"])

    metadata_rows: list[dict] = []
    features_agent: list[np.ndarray] = []
    features_map: list[np.ndarray] = []
    features_ego: list[np.ndarray] = []

    for idx, scene in enumerate(ordered_scenes, start=1):
        scene_name = scene["name"]
        scene_token = scene["token"]
        print(f"[{idx:03d}/{len(ordered_scenes)}] extracting {scene_name} ({scene_token})")

        try:
            metadata = collect_metadata_row(nusc, scene)
            metadata_rows.append(metadata)
            features_agent.append(agent_extractor.extract(nusc, scene_token))
            nusc_map = map_cache[metadata["location"]]
            features_map.append(map_extractor.extract(nusc, nusc_map, scene_token))
            features_ego.append(ego_extractor.extract(nusc, scene_token))
        except Exception as exc:  # noqa: BLE001 - explicit fail-fast behavior.
            raise RuntimeError(
                f"Feature extraction failed for scene {scene_name} ({scene_token}): {exc}"
            ) from exc

    features_agent_arr = np.stack(features_agent).astype(np.float32)
    features_map_arr = np.stack(features_map).astype(np.float32)
    features_ego_arr = np.stack(features_ego).astype(np.float32)

    if features_agent_arr.shape != (settings["expected_scenes"], 17):
        raise RuntimeError(f"Unexpected agent feature shape: {features_agent_arr.shape}")
    if features_map_arr.shape != (settings["expected_scenes"], 8):
        raise RuntimeError(f"Unexpected map feature shape: {features_map_arr.shape}")
    if features_ego_arr.shape != (settings["expected_scenes"], 9):
        raise RuntimeError(f"Unexpected ego feature shape: {features_ego_arr.shape}")

    metadata = pd.DataFrame(metadata_rows)
    if len(metadata) != settings["expected_scenes"]:
        raise RuntimeError(f"Unexpected metadata rows: {len(metadata)}")

    np.save(output_dir / "features_agent_raw.npy", features_agent_arr)
    np.save(output_dir / "features_map_raw.npy", features_map_arr)
    np.save(output_dir / "features_ego_raw.npy", features_ego_arr)

    write_metadata_parquet(metadata, output_dir / "metadata.parquet")

    (output_dir / "scene_order.json").write_text(
        json.dumps([row["scene_name"] for row in metadata_rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
