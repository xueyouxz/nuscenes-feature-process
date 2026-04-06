from __future__ import annotations

import numpy as np


class AgentFeatureExtractor:
    """Extract 17-D agent modality features for one scene."""

    OUTPUT_DIM = 17
    MAX_RADIUS_METERS = 50.0
    DYNAMIC_CLASS_KEYS = (
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
    )

    CLASS_PREFIX_MAP = {
        "car": ("vehicle.car",),
        "truck": ("vehicle.truck",),
        "bus": ("vehicle.bus",),
        "trailer": ("vehicle.trailer",),
        "construction_vehicle": ("vehicle.construction",),
        "pedestrian": ("human.pedestrian",),
        "motorcycle": ("vehicle.motorcycle",),
        "bicycle": ("vehicle.bicycle",),
    }
    STATIC_PREFIXES = ("movable_object.trafficcone", "movable_object.barrier")

    BAND_RADII = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0))
    BAND_AREAS = tuple(np.pi * (r2**2 - r1**2) for r1, r2 in BAND_RADII)
    ROI_AREA = np.pi * (MAX_RADIUS_METERS**2)
    FPS = 2.0
    DEFAULT_KEYFRAMES = 34

    def __init__(self, keyframes: int = DEFAULT_KEYFRAMES) -> None:
        if keyframes <= 0:
            raise ValueError("keyframes must be > 0")
        self.keyframes = keyframes

    def extract(self, nusc, scene_token: str) -> np.ndarray:
        scene = nusc.get("scene", scene_token)
        sample_tokens = self._collect_sample_tokens(scene["first_sample_token"], nusc)
        frame_features = [self._extract_one_frame(nusc, token) for token in sample_tokens]

        counts = np.array([x["total_count"] for x in frame_features], dtype=np.float32)
        dynamic_props = np.array([x["dynamic_props"] for x in frame_features], dtype=np.float32)
        band_densities = np.array([x["band_densities"] for x in frame_features], dtype=np.float32)
        speed_means = np.array([x["speed_mean"] for x in frame_features], dtype=np.float32)
        speed_stds = np.array([x["speed_std"] for x in frame_features], dtype=np.float32)
        v1_ratios = np.array([x["v1_ratio"] for x in frame_features], dtype=np.float32)
        static_densities = np.array(
            [x["static_density"] for x in frame_features], dtype=np.float32
        )

        # 2 + 8 + 3 + 2 + 1 + 1 = 17
        features = np.concatenate(
            [
                np.array([float(np.mean(counts)), float(np.var(counts))], dtype=np.float32),
                np.mean(dynamic_props, axis=0).astype(np.float32),
                np.mean(band_densities, axis=0).astype(np.float32),
                np.array(
                    [float(np.mean(speed_means)), float(np.mean(speed_stds))], dtype=np.float32
                ),
                np.array([float(np.mean(v1_ratios))], dtype=np.float32),
                np.array([float(np.mean(static_densities))], dtype=np.float32),
            ]
        ).astype(np.float32)

        if features.shape != (self.OUTPUT_DIM,):
            raise RuntimeError(
                f"Agent extractor produced wrong shape {features.shape}, expected {(self.OUTPUT_DIM,)}"
            )

        return features

    def _collect_sample_tokens(self, first_sample_token: str, nusc) -> list[str]:
        tokens: list[str] = []
        token = first_sample_token
        while token:
            tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]

        if len(tokens) < self.keyframes:
            raise RuntimeError(
                f"Scene has {len(tokens)} keyframes, expected at least {self.keyframes}"
            )

        return tokens[: self.keyframes]

    def _extract_one_frame(self, nusc, sample_token: str) -> dict:
        sample = nusc.get("sample", sample_token)
        ego_xy = self._get_ego_xy(nusc, sample)

        total_count = 0
        dynamic_count = 0
        dynamic_hist = {k: 0 for k in self.DYNAMIC_CLASS_KEYS}
        static_count = 0
        distance_bands = np.zeros(3, dtype=np.float32)
        speed_values: list[float] = []
        severe_occlusion = 0

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            dist = self._distance_xy(ann["translation"][:2], ego_xy)
            if dist > self.MAX_RADIUS_METERS:
                continue

            class_key = self._classify_category(ann["category_name"])
            if class_key is None:
                continue

            total_count += 1
            if ann.get("visibility_token") == "1":
                severe_occlusion += 1

            band_idx = self._distance_band_index(dist)
            if band_idx is not None:
                distance_bands[band_idx] += 1.0

            if class_key in dynamic_hist:
                dynamic_count += 1
                dynamic_hist[class_key] += 1
                speed = nusc.box_velocity(ann_token)
                if speed is not None:
                    speed_xy = np.asarray(speed[:2], dtype=np.float32)
                    speed_values.append(float(np.linalg.norm(speed_xy)))
            else:
                static_count += 1

        dynamic_props = np.zeros(len(self.DYNAMIC_CLASS_KEYS), dtype=np.float32)
        if dynamic_count > 0:
            for i, key in enumerate(self.DYNAMIC_CLASS_KEYS):
                dynamic_props[i] = dynamic_hist[key] / dynamic_count

        band_densities = distance_bands / np.array(self.BAND_AREAS, dtype=np.float32)
        static_density = static_count / self.ROI_AREA
        v1_ratio = severe_occlusion / total_count if total_count > 0 else 0.0

        speed_arr = np.asarray(speed_values, dtype=np.float32)
        if speed_arr.size == 0:
            speed_mean = 0.0
            speed_std = 0.0
        else:
            speed_mean = float(np.nanmean(speed_arr))
            speed_std = float(np.nanstd(speed_arr))
            if np.isnan(speed_mean):
                speed_mean = 0.0
            if np.isnan(speed_std):
                speed_std = 0.0

        return {
            "total_count": float(total_count),
            "dynamic_props": dynamic_props,
            "band_densities": band_densities.astype(np.float32),
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "v1_ratio": float(v1_ratio),
            "static_density": float(static_density),
        }

    @staticmethod
    def _distance_xy(a_xy, b_xy) -> float:
        ax, ay = float(a_xy[0]), float(a_xy[1])
        bx, by = float(b_xy[0]), float(b_xy[1])
        return float(np.hypot(ax - bx, ay - by))

    @classmethod
    def _classify_category(cls, category_name: str) -> str | None:
        for key, prefixes in cls.CLASS_PREFIX_MAP.items():
            if category_name.startswith(prefixes):
                return key
        if category_name.startswith(cls.STATIC_PREFIXES):
            return "static"
        return None

    @classmethod
    def _distance_band_index(cls, dist: float) -> int | None:
        for i, (left, right) in enumerate(cls.BAND_RADII):
            if left <= dist < right:
                return i
        if np.isclose(dist, cls.MAX_RADIUS_METERS):
            return len(cls.BAND_RADII) - 1
        return None

    @staticmethod
    def _get_ego_xy(nusc, sample: dict) -> tuple[float, float]:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        tx, ty = ego_pose["translation"][:2]
        return float(tx), float(ty)
