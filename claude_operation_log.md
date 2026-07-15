### 2026-07-14 — 阶段一：任务 1.1 + 1.5 开发

* **当前操作动作**：创建新文件 — `SkeletonDataLoader` (1.1) + `SlidingWindow` (1.5) 核心模块及单元测试
* **核心变更说明**：
  1. 实现 `SkeletonDataLoader` 类 (`src/video_analysis/data_loader.py`)：支持从 .zip 或目录读取 Toyota Smarthome V1.2 骨骼 JSON，暴露 `load()` / `list_files()` / `iter_frames()` / `iter_sliding_windows()` 等 API，返回带类型的 `SkeletonSequence → SkeletonFrame → PersonPose` 数据容器，pose2d/pose3d 以 `np.float32` 数组返回
  2. 实现 `SlidingWindow[T]` 泛型类 (`src/video_analysis/sliding_window.py`)：基于 `collections.deque(maxlen)` 的高性能时序滑动窗口，O(1) push，O(stride) advance，内置性能计时（实测 push 0.55 µs，advance 2.74 µs），附带 `stream_windows()` 流式便捷函数
  3. 创建 `config.py`：13 关节名称常量、窗口默认参数、夜间时段定义、数据集路径常量
  4. 编写 36 个单元测试（DataLoader 15 个 + SlidingWindow 21 个）
  5. 端到端验证：成功解析 zip 内全部 16115 条真实 skeleton 序列，多人序列 (K=5) 正常
* **涉及/修改的文件清单**：
  - `src/video_analysis/__init__.py` (Created)
  - `src/video_analysis/config.py` (Created)
  - `src/video_analysis/data_loader.py` (Created)
  - `src/video_analysis/sliding_window.py` (Created)
  - `tests/__init__.py` (Created)
  - `tests/test_data_loader.py` (Created)
  - `tests/test_sliding_window.py` (Created)
* **执行结果与验证状态**：36/36 测试通过，真实数据集解析及滑动窗口性能压测通过
* **置信度或遗留待办（TODO）**：13 关节名称映射基于标准 Toyota Smarthome V1.2 约定，如有差异需调整 `config.py` 中的 `JOINT_NAMES`
---

### 2026-07-14 — 阶段一：任务 1.2 开发

* **当前操作动作**：创建新文件 — `VideoStreamReader` 视频流接入模块 (1.2)
* **核心变更说明**：
  1. 实现抽象基类 `VideoStreamReader`：定义统一的帧迭代接口（`__iter__` / `__next__`）、上下文管理器、`fps` / `total_frames` / `resolution` / `duration_s` / `is_live` 属性
  2. 实现 `LocalVideoReader`：基于 OpenCV `VideoCapture` 读取本地 .mp4 文件，支持 `start_frame` / `max_frames` / `frame_skip` 参数
  3. 实现 `RTSPStreamReader`：连接 RTSP/HTTP 网络摄像头流，断线自动重连（指数退避，上限 30s），支持 `reconnect` / `timeout_s` / `max_frames` / `frame_skip`
  4. 实现 `MockVideoReader`：无摄像头/无文件环境下生成合成帧（弹跳球 + 网格 + 帧计数器），支持 `frame_skip`，用于 CI / 单元测试 / 管线冒烟测试
  5. 实现 `VideoFrame` dataclass：封装帧图像 (BGR ndarray) + frame_index + timestamp_s + 宽高，提供 `.rgb` 属性转换通道
  6. 实现 `create_reader()` 工厂函数：根据 URL scheme 自动创建对应 Reader（mock:// / rtsp:// / http:// / 本地路径）
  7. 编写 32 个单元测试（MockVideoReader 5 + LocalVideoReader 11 + RTSPStreamReader 3 + VideoFrame 3 + create_reader 5 + EdgeCases 5）
* **涉及/修改的文件清单**：
  - `src/video_analysis/video_stream.py` (Created)
  - `tests/test_video_stream.py` (Created)
* **执行结果与验证状态**：全量 68/68 测试通过（含 1.1/1.5 回归），Mock/Local 均可正常读取帧，RTSP 类接口正确（实际 RTSP 连接需真实摄像头）
* **置信度或遗留待办（TODO）**：RTSP 重连逻辑未经真实摄像头长期压测；可后续添加 `ImageStreamReader` 支持连续图片序列输入
---

### 2026-07-14 — 阶段一：任务 1.3 开发

* **当前操作动作**：创建新文件 — `VideoFeatureExtractor` 特征提取基类及骨架实现 (1.3)
* **核心变更说明**：
  1. 定义输出数据类：`BasicFeatures`（A1 6 项指标 + multi_person_duration）+ `FeatureWindow`（窗口级结果 + monitoring_quality），均支持 `to_dict()` JSON 序列化
  2. 实现抽象基类 `VideoFeatureExtractor`：`_compute_velocity()` / `_is_night_hour()` / `_make_window()` 共享方法，窗口 ID 自动递增
  3. 实现 `SkeletonFeatureExtractor`（纯 CPU）：从 SkeletonDataLoader 读取数据 → 逐帧计算 pelvis 速度/活跃状态/房间切换/夜间活动/多人共现 → 滑动窗口聚合 → 产出 `FeatureWindow`。关键参数：`velocity_threshold`（默认 0.02 m/frame）、`room_transition_threshold`（0.5 m）、`video_start_hour`（8.0）
  4. 实现 `YOLOPoseFeatureExtractor` 桩：真实 GPU 模式抛出 `NotImplementedError`；`mock=True` 模式生成合成特征窗口用于测试
  5. 实现 `_MetricsAccumulator`：每窗口内部状态累积器，支持 reset，用 `__slots__` 优化内存
  6. 实现 `process_all()`：批量处理全部 skeleton 序列
  7. 编写 27 个单元测试（BasicFeatures 3 + FeatureWindow 2 + 基类 3 + MetricsAccumulator 2 + SkeletonFeatureExtractor 13 + YOLOPose 2 + EdgeCases 2）
  8. 真实数据端到端验证：3 条 Toyota Smarthome 序列成功产出 123 个特征窗口，全部通过 JSON Schema 校验
* **涉及/修改的文件清单**：
  - `src/video_analysis/feature_extractor.py` (Created)
  - `tests/test_feature_extractor.py` (Created)
* **执行结果与验证状态**：全量 95/95 测试通过（含 1.1/1.2/1.5 回归），真实数据管道运行正常
* **置信度或遗留待办（TODO）**：`velocity_threshold` 默认值（0.02 m/frame @ 15fps = 0.3 m/s）可能需要根据实际场景校准；房间切换目前基于空间位移阈值替代，接入多摄像头后可改为 camera_id 切换检测
---

### 2026-07-14 — 阶段一：任务 1.4 开发

* **当前操作动作**：创建新文件 — `FeatureAggregator` 基础行为特征聚合模块 (1.4)
* **核心变更说明**：
  1. 实现 `HourlyAggregation` dataclass：按小时 (0–23) 聚合 A1 指标，加权平均 sedentary_ratio / average_velocity，直接求和 activity_minutes / room_transitions / night_activity / multi_person_duration
  2. 实现 `DailyAggregation` dataclass：24 小时日报，包含 `basic_features`（日总计）+ `hourly_breakdown`（仅含非零小时）+ `monitoring_quality`（跨窗口平均）
  3. 实现 `SequenceReport` dataclass：完全匹配项目 JSON Schema §6.1 的输出格式（user_id / device_id / time_window / basic_features / hourly_breakdown / monitoring_quality）
  4. 实现 `FeatureAggregator` 类：流式摄入 `FeatureWindow` → 按 `start_frame → clock_hour` 自动分 bin → `flush_daily()` / `flush_sequence_report()` 产出结构化结果。支持 `reset()` 复用
  5. 实现 `batch_process_sequences()`：一键串联 Extractor → Aggregator，遍历全部 skeleton 文件输出 SequenceReport 流
  6. 编写 24 个单元测试（Hourly 2 + Daily 2 + SequenceReport 1 + Aggregator 12 + EdgeCases 5 + 集成 2）；真实数据验证通过
* **涉及/修改的文件清单**：
  - `src/video_analysis/aggregator.py` (Created)
  - `tests/test_aggregator.py` (Created)
* **执行结果与验证状态**：全量 119/119 测试通过（含 1.1–1.3 回归），真实 skeleton 数据产出完整 SequenceReport，项目 JSON Schema 字段齐全
* **置信度或遗留待办（TODO）**：日级聚合目前按 per-sequence 定义"一天"；多摄像头/多序列跨天聚合需在接入真实时间戳后扩展
---

### 2026-07-14 — 阶段一：任务 1.6 开发 + 文档更新

* **当前操作动作**：创建全链路集成测试 + 编写项目 README 文档
* **核心变更说明**：
  1. 创建 `tests/conftest.py`：共享 session-scoped fixtures（skeleton zip 构建器、真实数据集路径、window 工厂、GPU 检测）
  2. 创建 `tests/test_pipeline.py`（21 tests）：覆盖五大类别——真实数据全链路（3 tests）、合成数据端到端（4 tests）、CPU-only 模式验证（6 tests）、性能约束（3 tests）、边界情况（5 tests）
  3. 补全 `batch_process_sequences()` 参数：添加 `velocity_threshold` 和 `room_transition_threshold` 透传
  4. 编写 `README.md`（436 行）：严格遵循 `video_tasks.md` §9.1 规范，包含项目概览、Mermaid 系统架构图、快速开始（环境/安装/数据集/测试/示例代码）、工程目录结构、JSON API Schema（§6.1 + §6.2 Qwen 复核）、Python API 速查、维护说明（操作日志/测试命令/进度/更新时机）、无卡模式约束
* **涉及/修改的文件清单**：
  - `tests/conftest.py` (Created)
  - `tests/test_pipeline.py` (Created)
  - `src/video_analysis/aggregator.py` (Modified — batch 函数参数扩展)
  - `README.md` (Updated — 全面重写)
* **执行结果与验证状态**：全量 140/140 测试通过（62s），真实数据管道吞吐正常，全部 Schema 校验通过；README 覆盖 video_tasks.md §9.1 全部要求
* **置信度或遗留待办（TODO）**：阶段一全部完成；可直接进入阶段二（专项行为判定逻辑）
---

