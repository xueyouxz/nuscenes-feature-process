from __future__ import annotations

import math

import numpy as np


class MapFeatureExtractor:
    """Extract 8-D map modality features for one scene."""

    OUTPUT_DIM = 8
    ROI_RADIUS_METERS = 30.0
    CANONICAL_SAMPLE_IDXS = (0, 9, 19, 29, 39)
    DEFAULT_KEYFRAMES = 34
    ROI_AREA = math.pi * (ROI_RADIUS_METERS**2)

    def __init__(self, keyframes: int = DEFAULT_KEYFRAMES) -> None:
        if keyframes <= 0:
            raise ValueError("keyframes must be > 0")
        self.keyframes = keyframes

    def extract(self, nusc, nusc_map, scene_token: str) -> np.ndarray:
        if nusc_map is None:
            raise RuntimeError("Map extractor requires a valid nusc_map instance.")

        scene = nusc.get("scene", scene_token)
        sample_tokens = self._collect_sample_tokens(scene["first_sample_token"], nusc)

        per_frame = []
        for idx in self._sample_indices():
            sample = nusc.get("sample", sample_tokens[idx])
            ego_x, ego_y = self._get_ego_xy(nusc, sample)
            per_frame.append(self._extract_at_pose(nusc_map, ego_x, ego_y))

        features = np.mean(np.asarray(per_frame, dtype=np.float32), axis=0).astype(np.float32)
        if features.shape != (self.OUTPUT_DIM,):
            raise RuntimeError(
                f"Map extractor produced wrong shape {features.shape}, expected {(self.OUTPUT_DIM,)}"
            )
        return features

    def _extract_at_pose(self, nusc_map, ego_x: float, ego_y: float) -> np.ndarray:
        layer_names = [
            "lane",
            "lane_connector",
            "ped_crossing",
            "carpark_area",
            "drivable_area",
            "road_segment",
            "road_block",
        ]
        records = nusc_map.get_records_in_radius(
            ego_x, ego_y, self.ROI_RADIUS_METERS, layer_names=layer_names
        )

        lane_tokens = records["lane"]
        lane_connector_tokens = records["lane_connector"]
        road_segment_tokens = records["road_segment"]
        road_block_tokens = records["road_block"]

        lane_count = float(len(lane_tokens))
        lane_length = float(self._lane_length_sum(nusc_map, lane_tokens))
        lane_connector_count = float(len(lane_connector_tokens))

        ped_crossing_ratio = self._layer_area_ratio(nusc_map, "ped_crossing", records["ped_crossing"], ego_x, ego_y)
        carpark_ratio = self._layer_area_ratio(nusc_map, "carpark_area", records["carpark_area"], ego_x, ego_y)
        drivable_ratio = self._layer_area_ratio(nusc_map, "drivable_area", records["drivable_area"], ego_x, ego_y)

        road_segment_count = float(len(road_segment_tokens))
        intersection_exists = 1.0 if len(road_block_tokens) > 0 else 0.0

        return np.asarray(
            [
                lane_count,
                lane_length,
                lane_connector_count,
                ped_crossing_ratio,
                carpark_ratio,
                drivable_ratio,
                road_segment_count,
                intersection_exists,
            ],
            dtype=np.float32,
        )

    def _layer_area_ratio(
        self, nusc_map, layer: str, tokens: list[str], ego_x: float, ego_y: float
    ) -> float:
        if not tokens:
            return 0.0

        try:
            from shapely.geometry import Point
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'shapely'. Install requirements before map extraction."
            ) from exc

        circle = Point(ego_x, ego_y).buffer(self.ROI_RADIUS_METERS, resolution=48)
        total_area = 0.0

        for token in tokens:
            for polygon in self._polygons_for_record(nusc_map, layer, token):
                try:
                    total_area += float(polygon.intersection(circle).area)
                except Exception:
                    continue

        ratio = total_area / self.ROI_AREA
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def _polygons_for_record(nusc_map, layer: str, token: str):
        record = nusc_map.get(layer, token)
        polygons = []

        if layer == "drivable_area":
            poly_tokens = record.get("polygon_tokens", []) or []
            for poly_token in poly_tokens:
                polygons.append(nusc_map.extract_polygon(poly_token))
            return polygons

        poly_token = record.get("polygon_token")
        if poly_token:
            polygons.append(nusc_map.extract_polygon(poly_token))
        return polygons

    @staticmethod
    def _lane_length_sum(nusc_map, lane_tokens: list[str]) -> float:
        if not lane_tokens:
            return 0.0

        try:
            discretized = nusc_map.discretize_lanes(lane_tokens, 1.0)
        except Exception:
            return 0.0

        total = 0.0
        for pts in discretized.values():
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            deltas = arr[1:, :2] - arr[:-1, :2]
            total += float(np.sum(np.linalg.norm(deltas, axis=1)))
        return total

    @staticmethod
    def _collect_all_sample_tokens(first_sample_token: str, nusc) -> list[str]:
        tokens: list[str] = []
        token = first_sample_token
        while token:
            tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]
        return tokens

    def _collect_sample_tokens(self, first_sample_token: str, nusc) -> list[str]:
        tokens = self._collect_all_sample_tokens(first_sample_token, nusc)
        if len(tokens) < self.keyframes:
            raise RuntimeError(
                f"Scene has {len(tokens)} keyframes, expected at least {self.keyframes}"
            )
        return tokens[: self.keyframes]

    def _sample_indices(self) -> list[int]:
        # Keep the 5-point temporal pattern from the 40-keyframe design and clamp to configured length.
        max_idx = self.keyframes - 1
        indices = [min(i, max_idx) for i in self.CANONICAL_SAMPLE_IDXS]
        # Remove duplicates while preserving order (e.g., very small keyframes).
        deduped = list(dict.fromkeys(indices))
        return deduped

    @staticmethod
    def _get_ego_xy(nusc, sample: dict) -> tuple[float, float]:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        tx, ty = ego_pose["translation"][:2]
        return float(tx), float(ty)
