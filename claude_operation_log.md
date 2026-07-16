# Claude Code 自动化操作日志

---

### [2026-07-15] - Batch 1：项目结构 + 配置 + 基础工具

* **当前操作动作**：创建项目目录结构，编写 5 个核心基础模块
* **核心变更说明**：
  1. 创建 `configs/default.yaml`：集中定义数据路径、模型路径、滑动窗口参数、行为检测阈值、夜间时段、MLLM 采样参数等
  2. 创建 `src/video_analysis/config.py`：YAML 配置加载器，支持环境变量覆盖（`PSY_VIDEO_*` 前缀），提供模块级单例 `get_config()`
  3. 创建 `src/utils/skeleton_parser.py`：Toyota Smarthome Skeleton V1.2 解析器，支持多种 JSON 格式变体，输出标准化 `(T, K, 3)` 关键点张量，附带质心计算和速度序列方法
  4. 创建 `src/video_analysis/sliding_window.py`：固定时间窗 + 最大容量的双约束滑动窗口，O(1) 插入/淘汰，线程安全；同时提供 `TimedSlidingWindow` 支持定时回调
  5. 创建 `src/utils/frame_sampler.py`：视频帧均匀采样器，支持 8/16/24 帧采样和事件窗口采样
  6. 创建 `src/utils/schema_validator.py`：JSON Schema 校验器，内置 §6.1 `DAILY_METRICS_SCHEMA` 和 §6.2 `QWEN_VL_EVENT_SCHEMA`，支持 Markdown 剥离重试和缺失字段自动填充
* **涉及/修改的文件清单**：
  - `configs/default.yaml` (Created)
  - `src/__init__.py` (Created)
  - `src/video_analysis/__init__.py` (Created)
  - `src/utils/__init__.py` (Created)
  - `src/video_analysis/config.py` (Created)
  - `src/video_analysis/sliding_window.py` (Created)
  - `src/utils/skeleton_parser.py` (Created)
  - `src/utils/frame_sampler.py` (Created)
  - `src/utils/schema_validator.py` (Created)
  - `tests/__init__.py` (Created)
* **执行结果与验证状态**：所有模块通过 Python 导入和基础功能验证；Config、SlidingWindow、SkeletonParser、SchemaValidator 均正常运行
* **置信度或遗留待办（TODO）**：
  - Skeleton V1.2 实际文件格式需待 `/dataset/` 挂载后做兼容性验证
  - `frame_sampler.py` 的 cv2 依赖已确认可用（5.0.0）

---

### [2026-07-15] - Batch 2：数据加载层（video_stream + data_loader）

* **当前操作动作**：创建视频流抽象层和双模式数据加载器
* **核心变更说明**：
  1. 创建 `src/video_analysis/video_stream.py`：`VideoStream` 抽象基类 + 三个具体实现
     - `FileVideoStream`：封装 cv2.VideoCapture，支持目标帧率降采样、BGR→RGB 转换、逐帧迭代
     - `CameraStream`：实时摄像头流，带帧率限速、自动 resize
     - `RTSPStream`：网络 RTSP 流，内置 5 次自动重连、TCP 传输优化、最小缓冲区
  2. 创建 `src/video_analysis/data_loader.py`：
     - `PerFrameData` dataclass：标准化帧数据（image / keypoints / bboxes / track_ids / metadata）
     - `RGBVideoLoader`：视频/摄像头/RTSP → 逐帧 RGB 图像的迭代器（仅读帧，不推理）
     - `SkeletonLoader`：Skeleton JSON → 直接输出标准化关键点序列，支持 `get_slice()` 时间区间切片
     - `DataLoaderFactory`：策略工厂，根据 source_type 自动创建对应加载器
  3. 双模式切换通过 `DataLoaderFactory.create(source, source_type, ...)` 一行代码完成
* **涉及/修改的文件清单**：
  - `src/video_analysis/video_stream.py` (Created)
  - `src/video_analysis/data_loader.py` (Created)
* **执行结果与验证状态**：SkeletonLoader 合成数据测试通过（10 帧解析、切片查询正确）；FileVideoStream 合成视频测试通过（30fps→15fps 降采样正确，RGB 输出正确）；所有导入和工厂模式验证通过
* **置信度或遗留待办（TODO）**：
  - CameraStream 和 RTSPStream 需在实际设备上验证
  - SkeletonLoader 的实际 V1.2 格式兼容性需待 `/dataset/` 挂载后验证

---

### [2026-07-15] - Batch 3：姿态推理与多目标跟踪封装

* **当前操作动作**：创建 YOLOv8-Pose 推理封装和 ByteTrack 风格多目标跟踪器
* **核心变更说明**：
  1. 创建 `pose_estimator.py`：
     - `PoseEstimator` 类支持 mock/real 双模式
     - Mock 模式：基于 COCO 17 点人体模板 + 正弦运动生成模拟关键点，模拟 1-3 人
     - Real 模式：封装 ultralytics YOLO 模型，需 `approve_gpu=True` 审批后才加载
     - 输出标准化 `{keypoints: (N,17,3), bboxes: (N,4), confidences: (N,)}`
  2. 创建 `tracker.py`：
     - `TrackState`：匀速运动预测 + 指数平滑更新的单个目标状态
     - `MultiObjectTracker`：ByteTrack 风格的 IOU 匹配 + 二次匹配（高分/低分检测分离）
     - 纯 numpy 实现，支持 scipy 匈牙利算法或贪心回退
     - 支持 track 确认/丢失/删除生命周期管理
* **涉及/修改的文件清单**：
  - `src/video_analysis/pose_estimator.py` (Created)
  - `src/video_analysis/tracker.py` (Created)
* **执行结果与验证状态**：
  - Mock 推理：正确生成 (1,17,3) 关键点 + 检测框
  - 单目标跟踪：10 帧后 track 确认，坐标平滑正确
  - 多目标跟踪：2 人 15 帧，track ID 稳定保持
  - IOU 计算：与预期值 0.143 一致
* **置信度或遗留待办（TODO）**：
  - Real 模式的 YOLO 推理待 GPU 审批后验证（需安装 ultralytics）
  - numpy 2.2.6 与 torch 2.1.2 兼容性问题已通过降级 numpy 到 1.26.4 解决
  - ultralytics 8.4.95 + scipy 已安装就绪

---

### [2026-07-15] - Batch 4：特征提取器 + 日级聚合器

* **当前操作动作**：创建 VideoFeatureExtractor 和 DailyAggregator，实现 A1 核心 6 项指标
* **核心变更说明**：
  1. 创建 `feature_extractor.py`：
     - `VideoFeatureExtractor` 消费标准化关键点帧数据，通过滑动窗口计算 6 项指标
     - 支持 activity_minutes / sedentary_ratio / room_transitions / movement_velocity / night_activity / multi_person_duration
     - 网格化空间建图检测房间切换，髋部中点计算质心位移和速度
     - 窗口历史自动保留用于日级聚合
     - `_CumulativeMetrics` 作为窗口不足时的回退累计器
  2. 创建 `aggregator.py`：
     - `DailyAggregator` 收集窗口指标，聚合输出严格符合 §6.1 JSON Schema
     - 内置 Schema 自动校验和修复（缺失字段填充、范围约束）
     - 支持多日范围聚合 `aggregate_range()`
* **涉及/修改的文件清单**：
  - `src/video_analysis/feature_extractor.py` (Created)
  - `src/video_analysis/aggregator.py` (Created)
* **执行结果与验证状态**：
  - 300 帧单人行走模拟：输出 2 个窗口，Schema 校验通过
  - 150 帧多人场景：多人共现正确检测
  - 空数据边界：Schema 校验通过（全零输出）
  - 日级输出所有 9 个字段齐全且类型正确
* **置信度或遗留待办（TODO）**：
  - sedentary 判定阈值（per-frame 50px）需用真实数据校准，当前单帧位移太小导致过度判定为静止
  - 夜间时段基于虚拟时钟（elapsed_sec），实际部署需接入真实时钟或视频元数据时间戳

---

### [2026-07-15] - Batch 5：A1 单元测试 + 全链路集成测试

* **当前操作动作**：编写 10 个测试文件，覆盖 A1 所有模块 + 全链路集成
* **核心变更说明**：
  1. 创建 `tests/conftest.py`：共享 fixtures（合成骨骼数据、合成视频、Mock PoseEstimator、PerFrameData 生成器）
  2. 创建 9 个测试文件，覆盖 a1.1–a1.10 全部子任务
  3. 全链路集成测试 `test_pipeline.py`：3 条独立管道路径
     - 单人模拟视频 → Mock 推理 → 跟踪 → 特征提取 → 聚合 → Schema 校验
     - 多人模拟场景全链路
     - Skeleton JSON 验证模式全链路
* **测试文件清单**：
  - `tests/conftest.py` (Created)
  - `tests/test_sliding_window.py` (Created) — 14 tests
  - `tests/test_skeleton_parser.py` (Created) — 12 tests
  - `tests/test_schema_validator.py` (Created) — 12 tests
  - `tests/test_video_stream.py` (Created) — 7 tests
  - `tests/test_data_loader.py` (Created) — 12 tests
  - `tests/test_pose_estimator.py` (Created) — 11 tests
  - `tests/test_tracker.py` (Created) — 18 tests
  - `tests/test_feature_extractor.py` (Created) — 8 tests
  - `tests/test_aggregator.py` (Created) — 6 tests
  - `tests/test_pipeline.py` (Created) — 3 tests
* **执行结果与验证状态**：**104 passed, 0 failed**，耗时 57s，全部在 CPU/Mock 模式下运行
* **A1 总结**：
  - 全部 A1.1–A1.10 子任务完成，产出 15 个源文件 + 11 个测试文件
  - Skeleton 验证模式 + RGB 视频生产模式均已跑通
  - 日级输出严格符合 §6.1 JSON Schema
  - 全链路 Pipeline 可在 CPU 模式完整运行
* **置信度或遗留待办（TODO）**：
  - GPU 模式待用户开启后验证 YOLO 真实推理
  - 静止判定阈值需在实际数据上校准（Skeleton V1.2 Ground Truth）
  - `/dataset/` 未挂载，Skeleton V1.2 真实格式兼容性待验证

---

### [2026-07-16] - GPU 全流程验证 + 姿态检测修复

* **当前操作动作**：RTX 4090 GPU 开启，运行 YOLOv8-Pose 真实推理，修复 sedentary 检测算法
* **核心变更说明**：
  1. GPU 验证：yolov8n-pose.pt 在 RTX 4090 上推理速度 109 fps，显存仅 45 MB
  2. Sedentary 检测重构：从单帧质心位移 → 多组关键点姿态高度估算
     - 肩-踝 / 髋-踝 / 髋-膝 / 鼻-髋 四级回退，置信度阈值 0.3→0.1
     - 站姿判定：姿态高度 > 图像高度 15%（480p 下 72px）
  3. coverage_minutes 修复：从窗口叠加求和 → 实际经过时间封顶
  4. room_transitions 修复：网格分辨率 50px → 200px
  5. 创建 `results/` 目录，输出文件带视频名和时间戳，永不覆盖
  6. 创建 `scripts/run_gpu_pipeline.py`：GPU 全流程脚本
* **测试视频结果**：
  - P12T05C05 (22.6min): active=17.88min, sedentary=0.61, coverage=22.6min ✅
  - P14T14C06 (9.6min): active=9.57min, sedentary=0.28, coverage=9.57min ✅
* **涉及/修改的文件清单**：
  - `src/video_analysis/feature_extractor.py` (Modified) — 姿态高度检测 + 多级回退
  - `src/video_analysis/pose_estimator.py` (No change, GPU verified)
  - `scripts/run_gpu_pipeline.py` (Created)
  - `scripts/run_cpu_pipeline.py` (Created)
  - `results/` (Created, 8 files)
* **执行结果与验证状态**：104/104 单元测试通过，Schema 校验通过，GPU 推理正常
* **置信度或遗留待办（TODO）**：
  - sedentary_ratio 在边界帧（人物刚进入/离开画面）仍有少量误报，需进一步调优
  - 早期 `output_*.json` 文件为中间调试结果，后续以 `results/` 中时间戳文件为准

---

### [2026-07-16] - A2 Batch 1：轨迹建图 + 徘徊检测 + 重复动作检测

* **当前操作动作**：创建 `special_behavior.py`，实现空间轨迹映射和两项专项检测器
* **核心变更说明**：
  1. `SpatialTrajectoryMap`：200px 网格空间建图，记录路径序列、网格访问计数、网格间转移计数
  2. `RepetitivePathDetector`：10 分钟滑动窗口内统计重复边（同一段路径出现 ≥3 次），计算路径重合度，超过 40% 阈值标记徘徊
  3. `RepeatedActionDetector`：空间聚类识别热点区域，统计短时间窗内的进出次数，超过阈值标记重复行为
  4. 所有输出均包含 `time_window`、`valid_duration`、`confidence_score`（§A2 质量要求）
* **涉及/修改的文件清单**：
  - `src/video_analysis/special_behavior.py` (Created)
* **执行结果与验证状态**：SpatialTrajectoryMap 网格映射测试通过；RepetitivePathDetector 徘徊模拟测试通过；RepeatedActionDetector 热点检测测试通过
* **置信度或遗留待办（TODO）**：
  - 徘徊检测需在真实视频上验证（当前模拟数据测试）
  - 重复动作检测的热点聚类半径需根据实际场景标定
  - A2 Batch 2 待实现：异常久坐、昼夜节律、社交互动
---
