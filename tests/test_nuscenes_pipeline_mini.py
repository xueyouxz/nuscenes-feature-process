from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from extractors.agent import AgentFeatureExtractor
from extractors.ego import EgoFeatureExtractor
from extractors.map import MapFeatureExtractor

MINI_ROOT = Path("~/Data/nuscenes-mini/").expanduser()
EXTRACT_CONFIG_ENV = "NUSCENES_EXTRACT_CONFIG"
PROJECT_CONFIG_ENV = "NUSCENES_PROJECT_CONFIG"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_extract_config(
    path: Path, mini_dataroot: Path, output_dir: Path, keyframes: int = 34
) -> None:
    path.write_text(
        json.dumps(
            {
                "default_version": "v1.0-mini",
                "version_to_dataroot": {"v1.0-mini": str(mini_dataroot)},
                "split_to_expected_scenes": {"mini_val": 2},
                "default_split_by_version": {"v1.0-mini": "mini_val"},
                "output_dir": str(output_dir),
                "keyframes": keyframes,
            }
        ),
        encoding="utf-8",
    )


def run_extract_with_config(cfg_path: Path) -> None:
    env = os.environ.copy()
    env[EXTRACT_CONFIG_ENV] = str(cfg_path)
    cmd = [sys.executable, "extract_features.py"]
    subprocess.run(cmd, check=True, env=env)


def write_project_config(
    path: Path,
    input_dir: Path,
    modalities: list[str],
    disable_review_gate: bool = False,
    disable_numba_jit: bool = False,
    fallback_on_umap_error: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "plots_dir": None,
                "modalities": modalities,
                "random_state": 42,
                "disable_review_gate": disable_review_gate,
                "disable_numba_jit": disable_numba_jit,
                "fallback_on_umap_error": fallback_on_umap_error,
            }
        ),
        encoding="utf-8",
    )


def run_project_with_config(cfg_path: Path) -> None:
    env = os.environ.copy()
    env[PROJECT_CONFIG_ENV] = str(cfg_path)
    cmd = [sys.executable, "project_and_visualize.py"]
    subprocess.run(cmd, check=True, env=env)


@pytest.fixture(scope="session")
def mini_dataroot() -> Path:
    if not MINI_ROOT.exists():
        pytest.skip(f"nuscenes-mini not found at {MINI_ROOT}")
    return MINI_ROOT


@pytest.fixture(scope="session")
def nusc_mini(mini_dataroot: Path) -> NuScenes:
    os.environ.setdefault("MPLCONFIGDIR", str((mini_dataroot / ".mplconfig").resolve()))
    return NuScenes(version="v1.0-mini", dataroot=str(mini_dataroot), verbose=False)


@pytest.fixture(scope="session")
def mini_val_scenes(nusc_mini: NuScenes) -> list[dict]:
    val_names = set(create_splits_scenes()["mini_val"])
    scenes = sorted(
        [scene for scene in nusc_mini.scene if scene["name"] in val_names],
        key=lambda x: x["name"],
    )
    assert len(scenes) == 2
    return scenes


@pytest.fixture(scope="session")
def map_cache(nusc_mini: NuScenes, mini_dataroot: Path) -> dict[str, NuScenesMap]:
    locations = {
        nusc_mini.get("log", scene["log_token"])["location"]
        for scene in nusc_mini.scene
    }
    return {
        location: NuScenesMap(dataroot=str(mini_dataroot), map_name=location)
        for location in locations
    }


def test_agent_extractor_contract_on_mini(nusc_mini: NuScenes, mini_val_scenes: list[dict]) -> None:
    extractor = AgentFeatureExtractor()
    feat = extractor.extract(nusc_mini, mini_val_scenes[0]["token"])

    assert feat.shape == (17,)
    assert np.isfinite(feat).all()

    dynamic_props = feat[2:10]
    assert np.all(dynamic_props >= 0.0)
    assert np.all(dynamic_props <= 1.0)
    assert float(dynamic_props.sum()) <= 1.0 + 1e-6

    band_densities = feat[10:13]
    assert np.all(band_densities >= 0.0)

    assert 0.0 <= float(feat[15]) <= 1.0
    assert float(feat[16]) >= 0.0


def test_map_extractor_contract_on_mini(
    nusc_mini: NuScenes,
    mini_val_scenes: list[dict],
    map_cache: dict[str, NuScenesMap],
) -> None:
    extractor = MapFeatureExtractor()
    scene = mini_val_scenes[0]
    location = nusc_mini.get("log", scene["log_token"])["location"]

    feat = extractor.extract(nusc_mini, map_cache[location], scene["token"])
    assert feat.shape == (8,)
    assert np.isfinite(feat).all()

    assert float(feat[0]) >= 0.0
    assert float(feat[1]) >= 0.0
    assert float(feat[2]) >= 0.0
    assert 0.0 <= float(feat[3]) <= 1.0
    assert 0.0 <= float(feat[4]) <= 1.0
    assert 0.0 <= float(feat[5]) <= 1.0
    assert float(feat[6]) >= 0.0
    assert float(feat[7]) in (0.0, 1.0)


def test_ego_extractor_contract_on_mini(nusc_mini: NuScenes, mini_val_scenes: list[dict]) -> None:
    extractor = EgoFeatureExtractor()
    feats = [extractor.extract(nusc_mini, scene["token"]) for scene in mini_val_scenes]

    for feat in feats:
        assert feat.shape == (9,)
        assert np.isfinite(feat).all()
        assert float(feat[0]) >= 0.0
        assert float(feat[1]) >= 0.0
        assert float(feat[2]) >= 0.0
        assert float(feat[3]) >= 0.0
        assert float(feat[4]) >= 0.0
        assert float(feat[5]) >= 0.0
        assert float(feat[6]) >= 0.0
        assert float(feat[7]) >= 0.0
        assert 0.0 <= float(feat[8]) <= 1.0

    assert max(float(x[7]) for x in feats) > 0.0


def test_extract_features_script_on_mini_val(mini_dataroot: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    cfg_path = tmp_path / "extract_config.json"
    write_extract_config(cfg_path, mini_dataroot, out_dir)
    run_extract_with_config(cfg_path)

    agent = np.load(out_dir / "features_agent_raw.npy")
    map_feat = np.load(out_dir / "features_map_raw.npy")
    ego = np.load(out_dir / "features_ego_raw.npy")
    metadata = pd.read_parquet(out_dir / "metadata.parquet")

    assert agent.shape == (2, 17)
    assert map_feat.shape == (2, 8)
    assert ego.shape == (2, 9)
    assert len(metadata) == 2


def test_project_script_generates_plots_without_overwriting_raw(
    mini_dataroot: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "outputs"
    cfg_path = tmp_path / "extract_config.json"
    write_extract_config(cfg_path, mini_dataroot, out_dir)
    run_extract_with_config(cfg_path)

    agent_path = out_dir / "features_agent_raw.npy"
    before_hash = file_sha256(agent_path)

    project_cfg = tmp_path / "project_config.json"
    write_project_config(
        project_cfg,
        input_dir=out_dir,
        modalities=["agent"],
        disable_review_gate=True,
        disable_numba_jit=True,
        fallback_on_umap_error=True,
    )
    run_project_with_config(project_cfg)

    after_hash = file_sha256(agent_path)
    assert before_hash == after_hash

    plots_dir = out_dir / "plots"
    pngs = sorted(plots_dir.glob("agent_nn*_md*.png"))
    assert len(pngs) == 12


def test_extract_features_uses_config_version_mapping_and_split_list(
    mini_dataroot: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "outputs_cfg"
    cfg_path = tmp_path / "extract_config.json"
    write_extract_config(cfg_path, mini_dataroot, out_dir)
    run_extract_with_config(cfg_path)

    metadata = pd.read_parquet(out_dir / "metadata.parquet")
    scene_order = json.loads((out_dir / "scene_order.json").read_text(encoding="utf-8"))
    expected_names = sorted(create_splits_scenes()["mini_val"])

    assert len(metadata) == 2
    assert sorted(metadata["scene_name"].tolist()) == expected_names
    assert scene_order == sorted(scene_order)


def test_extract_features_respects_keyframes_config(
    mini_dataroot: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "outputs_kf5"
    cfg_path = tmp_path / "extract_config_kf5.json"
    write_extract_config(cfg_path, mini_dataroot, out_dir, keyframes=5)
    run_extract_with_config(cfg_path)

    agent = np.load(out_dir / "features_agent_raw.npy")
    map_feat = np.load(out_dir / "features_map_raw.npy")
    ego = np.load(out_dir / "features_ego_raw.npy")

    assert agent.shape == (2, 17)
    assert map_feat.shape == (2, 8)
    assert ego.shape == (2, 9)
