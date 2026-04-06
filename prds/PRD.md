# PRD: nuScenes三模态场景特征提取与投影

## Problem Statement

自动驾驶场景理解研究中，需要对nuScenes验证集（150个scene）进行场景多样性分析。当前缺乏一种方法，能从三种不同的语义视角（交通参与者密度/构成、道路地图拓扑、自车行驶状态）独立量化场景间的相似性，从而揭示不同视角下的场景分组规律。直接处理原始传感器数据或模型输出代价过高，需要一条从nuScenes ground truth标注到可视化散点图的轻量化流水线。

## Solution

构建一条两阶段线性流水线：第一阶段从nuScenes验证集原始标注中为每个scene提取三个独立的统计特征向量（交通参与者模态17维、地图模态8维、自车状态模态9维），持久化到磁盘；第二阶段加载特征，经RobustScaler标准化后独立进行UMAP参数扫描，输出每模态12张散点图（共36张），用已知场景属性（Boston/Singapore地点、直行/转弯工况）对结果做肉眼有效性验证。

## User Stories

1. As a researcher, I want to extract agent features from all 150 validation scenes in a single script run, so that I don't have to manually process each scene.
2. As a researcher, I want agent features to capture per-frame statistics averaged over 40 keyframes, so that temporal dynamics are represented rather than a single snapshot.
3. As a researcher, I want to restrict agent feature computation to objects within 50m of the ego vehicle, so that distant low-quality annotations don't pollute the feature distribution.
4. As a researcher, I want dynamic objects (car/truck/bus/trailer/construction_vehicle/pedestrian/motorcycle/bicycle) and static infrastructure (traffic_cone/barrier) computed with separate denominators, so that construction-zone scenes don't inflate the dynamic class proportions.
5. As a researcher, I want distance-band densities at 0-15m / 15-30m / 30-50m, so that spatial crowding patterns at different interaction distances are captured.
6. As a researcher, I want object velocity statistics computed with `np.nanmean`/`np.nanstd` (skipping NaN), so that objects tracked only in a single frame don't break the velocity computation.
7. As a researcher, I want severe occlusion (visibility v1, 0-40% visible) ratio as a single feature dimension, so that scene visibility quality is represented without introducing correlated histogram bins.
8. As a researcher, I want map features sampled at frames 1/10/20/30/40 and averaged, so that the full 20-second traversal path is represented rather than a single ego position.
9. As a researcher, I want map features extracted within a 30m circular radius around the ego vehicle, so that the extraction is invariant to ego heading without needing coordinate rotation.
10. As a researcher, I want map features to include lane count, lane length, lane connector count, crosswalk area ratio, parking area ratio, drivable area ratio, road segment count, and intersection existence, so that both topological complexity and geometric coverage are captured in 8 dimensions.
11. As a researcher, I want ego state features derived purely from pose differentiation without CAN bus dependency, so that all 150 validation scenes are covered without missing-data gaps.
12. As a researcher, I want trajectory curvature computed as cumulative heading-angle change (skipping frames with speed < 1 m/s), so that the computation is robust to stop-and-go segments without requiring curve fitting.
13. As a researcher, I want a "low-speed frame ratio" (fraction of frames with speed < 2 m/s) instead of stop-event count, so that congestion and parking scenarios are captured without noisy zero-crossing detection.
14. As a researcher, I want raw (un-normalized) feature matrices saved to disk alongside normalized versions, so that I can diagnose why specific scenes appear in unexpected projection positions.
15. As a researcher, I want a `metadata.parquet` file aligned row-by-row with the feature matrices, containing `scene_token`, `scene_name`, `location`, and `log_name`, so that scatter plot coloring by city or scene type requires no re-initialization of the nuScenes SDK.
16. As a researcher, I want the `metadata.parquet` schema to have reserved columns for future prediction metrics, so that model performance can be overlaid onto the projections without schema migration.
17. As a researcher, I want UMAP projections swept over `n_neighbors` ∈ [5, 10, 20, 30] and `min_dist` ∈ [0.0, 0.1, 0.5] with `random_state=42`, so that parameter sensitivity is decoupled from feature validity.
18. As a researcher, I want each of the 36 scatter plots colored by `location` (Boston/Singapore), so that the Boston/Singapore separation criterion can be assessed visually without additional processing.
19. As a researcher, I want the projection script to be independent of the extraction script, so that I can re-run parameter sweeps without re-extracting features.
20. As a researcher, I want RobustScaler (median + IQR) applied independently per modality, so that outlier scenes with extreme feature values don't dominate the normalization of the full dataset.

## Implementation Decisions

### Module Architecture

**`extract_features.py`** — Orchestrator script. Initializes nuScenes SDK once, iterates over all 150 validation scenes in a deterministic order, calls the three extractor classes, accumulates results, and writes output files. Exits with a clear error if any scene fails extraction.

**`extractors/agent.py` — `AgentFeatureExtractor`**
- Interface: `extract(nusc, scene_token) -> np.ndarray` (shape: 17,)
- Iterates over all 40 keyframes, computes per-frame statistics, returns 40-frame mean
- Filters to 50m radius; separates dynamic classes (8) from static infrastructure (cone+barrier)
- Uses `np.nanmean`/`np.nanstd` for velocity; uses v1 visibility flag for occlusion ratio

**`extractors/map.py` — `MapFeatureExtractor`**
- Interface: `extract(nusc, nusc_map, scene_token) -> np.ndarray` (shape: 8,)
- Samples at keyframe indices [0, 9, 19, 29, 39], queries circular 30m ROI at each ego position
- Queries: `lane`, `lane_connector`, `ped_crossing`, `carpark_area`, `drivable_area`, `road_segment`, `road_block` layer types via `get_records_in_radius`
- Returns mean across 5 frames

**`extractors/ego.py` — `EgoFeatureExtractor`**
- Interface: `extract(nusc, scene_token) -> np.ndarray` (shape: 9,)
- Derives velocity and acceleration from consecutive ego pose differences (finite difference, dt=0.5s)
- Derives heading angle from pose quaternion; steering approximated as heading-angle change rate
- Trajectory curvature: cumulative absolute heading-angle change, skipping frames with speed < 1 m/s
- Low-speed ratio: fraction of frames with speed < 2 m/s

**`project_and_visualize.py`** — Loads `features_*_raw.npy` and `metadata.parquet`, applies `RobustScaler`, runs UMAP over 4×3 parameter grid with `random_state=42`, saves 36 PNG scatter plots colored by `location`.

### Data Contracts

- All feature `.npy` files are shape `(150, D)` where D ∈ {17, 8, 9}
- Row order is identical across all three feature files and `metadata.parquet`
- Scene order is determined once in `extract_features.py` and treated as canonical

### Feature Dimensions

| 模态 | 维度 | 特征列表 |
|------|------|----------|
| Agent | 17 | 目标数量均值、目标数量方差、8类别占比（car/truck/bus/trailer/construction_vehicle/pedestrian/motorcycle/bicycle）、0-15m密度、15-30m密度、30-50m密度、速度均值、速度方差、v1遮挡占比、静态设施密度 |
| Map | 8 | 车道线数量、车道线总长度、lane_connector数量、人行横道面积占比、停车区面积占比、可行驶区域面积占比、road_segment数量、intersection存在性 |
| Ego | 9 | 速度均值、速度方差、速度最大值、加速度均值绝对值、加速度最大绝对值、转向角均值绝对值、转向角方差、轨迹总曲率、低速帧占比 |

### Normalization

- `RobustScaler` (sklearn) applied per modality independently
- Raw features saved before scaling; normalized features used only inside `project_and_visualize.py`

### Output Files

```
outputs/
  features_agent_raw.npy     # shape (150, 17)
  features_map_raw.npy       # shape (150, 8)
  features_ego_raw.npy       # shape (150, 9)
  metadata.parquet           # 150 rows: scene_token, scene_name, location, log_name, [reserved metric columns]
  plots/
    agent_nn{5,10,20,30}_md{0.0,0.1,0.5}.png   # 12 plots
    map_nn{5,10,20,30}_md{0.0,0.1,0.5}.png     # 12 plots
    ego_nn{5,10,20,30}_md{0.0,0.1,0.5}.png     # 12 plots
```

## Testing Decisions

Good tests verify observable outputs for given inputs — they do not inspect internal state or call private methods.

**`AgentFeatureExtractor`**
- Input: mock nuScenes scene with known annotation counts, classes, positions, velocities, and visibility flags
- Assert: output vector has shape (17,); class proportion dimensions sum to 1.0; distance bands are non-negative; NaN velocities are handled gracefully (no exception, result is finite)

**`MapFeatureExtractor`**
- Input: mock map response with known lane/crosswalk/drivable-area geometries within 30m
- Assert: output vector has shape (8,); area ratios are in [0, 1]; 5-frame averaging produces expected means

**`EgoFeatureExtractor`**
- Input: synthetic pose sequence (pure straight line, pure circle, stop-and-go)
- Assert: straight-line curvature ≈ 0; circular path curvature > 0; low-speed ratio correct for known velocity profile; no CAN bus file dependency

## Out of Scope

- Processing the nuScenes training set (444 scenes)
- Model prediction outputs or error metrics
- Joint multi-modal projection (e.g., concatenated features or learned fusion)
- Online or streaming feature extraction
- Any data augmentation
- 3D or interactive visualizations
- Quantitative clustering metrics (Silhouette score, ARI, etc.) — validation is visual only for MVP
- nuScenes mini or test splits

## Further Notes

- 两脚本架构（`extract_features.py` + `project_and_visualize.py`）是硬性要求，不是优化选项。36次UMAP参数扫描只有在特征提取不重复运行的前提下才可行。
- Boston/Singapore在地图投影中的分离以及自车投影中的直行大簇，是内置的合理性验证标准。如果任何参数组合下两者均不出现，应在继续开发前审查特征构建逻辑。
- 后续扩展：`metadata.parquet`预留列可填入逐场景预测指标（如mAP、速度误差），将模型性能叠加到投影空间，实现基于难度的场景分层分析。
