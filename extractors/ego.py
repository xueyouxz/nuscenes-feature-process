from __future__ import annotations

import math

import numpy as np


class EgoFeatureExtractor:
    """Extract 9-D ego modality features for one scene."""

    OUTPUT_DIM = 9
    DT_SECONDS = 0.5
    LOW_SPEED_THRESHOLD = 2.0
    CURVATURE_SPEED_THRESHOLD = 1.0
    DEFAULT_KEYFRAMES = 34

    def __init__(self, keyframes: int = DEFAULT_KEYFRAMES) -> None:
        if keyframes <= 0:
            raise ValueError("keyframes must be > 0")
        self.keyframes = keyframes

    def extract(self, nusc, scene_token: str) -> np.ndarray:
        scene = nusc.get("scene", scene_token)
        sample_tokens = self._collect_sample_tokens(scene["first_sample_token"], nusc)

        xy = []
        yaw = []
        for token in sample_tokens:
            sample = nusc.get("sample", token)
            ego_pose = self._get_ego_pose(nusc, sample)
            xy.append(np.asarray(ego_pose["translation"][:2], dtype=np.float32))
            yaw.append(self._yaw_from_quaternion(ego_pose["rotation"]))

        xy_arr = np.asarray(xy, dtype=np.float32)
        yaw_arr = np.asarray(yaw, dtype=np.float32)

        delta_xy = xy_arr[1:] - xy_arr[:-1]
        speeds = np.linalg.norm(delta_xy, axis=1) / self.DT_SECONDS
        if speeds.size == 0:
            raise RuntimeError("Insufficient keyframes to compute ego speed.")

        accel = np.diff(speeds) / self.DT_SECONDS
        accel_abs = np.abs(accel)

        yaw_delta = self._angle_diff(yaw_arr[1:], yaw_arr[:-1])
        steer_rate = yaw_delta / self.DT_SECONDS

        curvature_mask = speeds >= self.CURVATURE_SPEED_THRESHOLD
        total_curvature = float(np.sum(np.abs(yaw_delta[curvature_mask])))
        low_speed_ratio = float(np.mean(speeds < self.LOW_SPEED_THRESHOLD))

        features = np.asarray(
            [
                float(np.mean(speeds)),
                float(np.var(speeds)),
                float(np.max(speeds)),
                float(np.mean(accel_abs)) if accel_abs.size > 0 else 0.0,
                float(np.max(accel_abs)) if accel_abs.size > 0 else 0.0,
                float(np.mean(np.abs(steer_rate))),
                float(np.var(steer_rate)),
                total_curvature,
                low_speed_ratio,
            ],
            dtype=np.float32,
        )

        if features.shape != (self.OUTPUT_DIM,):
            raise RuntimeError(
                f"Ego extractor produced wrong shape {features.shape}, expected {(self.OUTPUT_DIM,)}"
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

    @staticmethod
    def _get_ego_pose(nusc, sample: dict) -> dict:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        return nusc.get("ego_pose", sample_data["ego_pose_token"])

    @staticmethod
    def _yaw_from_quaternion(quat_wxyz) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a - b
        return (diff + np.pi) % (2.0 * np.pi) - np.pi
