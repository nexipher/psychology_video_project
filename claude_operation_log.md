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

### [2026-07-16] - A2 Batch 2：异常久坐 + 昼夜节律 + 社交互动检测器

* **当前操作动作**：追加 3 个专项检测器到 `special_behavior.py`
* **核心变更说明**：
  1. `ProlongedInactivityDetector`：跟踪连续静止帧数，超 1h 预警 / 2h 触发警示，结合骨骼关键点标准差做微动分析（完全静止=高风险）
  2. `CircadianRhythmAnalyzer`：从活动/静止序列检测起床/入睡时间，建立 N 天个体基线，计算偏移量（超 2h 标记节律异常），检测午休时长
  3. `SocialInteractionAnalyzer`：多人场景下计算空间距离 + 朝向角度（肩线法向量）+ 近距离时长，加权输出互动强度 (0-1)
  4. 所有输出均包含 `time_window`、`valid_duration`、`confidence_score`（§A2 质量要求）
* **涉及/修改的文件清单**：
  - `src/video_analysis/special_behavior.py` (Modified, +414 lines, 830 total)
* **执行结果与验证状态**：
  - ProlongedInactivityDetector：连续静止检测 + 微动分析正常
  - CircadianRhythmAnalyzer：模拟一天数据，正确检测 6:00 起床 / 23:54 入睡 / 108min 午休
  - SocialInteractionAnalyzer：双人面对面场景交互强度计算正常
* **置信度或遗留待办（TODO）**：
  - 昼夜节律需多天数据才能建立有效基线（当前置信度低）
  - 社交朝向角度依赖肩部关键点——桌子遮挡时精度下降
  - A2 Batch 3 待实现：SpecialBehaviorDetector 总装 + 单元测试 + GPU 验证

---

### [2026-07-16] - A2 Batch 3：SpecialBehaviorDetector 总装 + 单元测试

* **当前操作动作**：创建 SpecialBehaviorDetector 统一入口 + 33 个单元测试
* **核心变更说明**：
  1. `SpecialBehaviorDetector`：5 个检测器的统一入口，单个 `update()` 调用自动分发到各子检测器，支持独立启用/禁用（可插拔架构），提供 `flush()` / `get_daily_summary()` / `get_circadian_report()` 接口
  2. 修复 `ProlongedInactivityDetector` 的 `0.0 or current_ts`  falsy bug（改为 `is not None` 判定）
  3. 创建 `tests/test_special_behavior.py`：33 个测试覆盖 7 个类
  4. 全量测试套件：**137 个测试全部通过**（A1 104 + A2 33）
* **涉及/修改的文件清单**：
  - `src/video_analysis/special_behavior.py` (Modified, +147 lines, 990 total)
  - `tests/test_special_behavior.py` (Created, 270 lines)
* **执行结果与验证状态**：`pytest tests/` — 137 passed, 0 failed in 69s
* **置信度或遗留待办（TODO）**：
  - GPU 验证待开启后在三段视频上集成 SpecialBehaviorDetector 跑全流程
  - 所有检测器参数（阈值、窗口、分辨率）需在真实数据上标定

---

### [2026-07-16] - 修复：坐姿检测重构 + 多人假阳性过滤

* **当前操作动作**：修复 P03T01C05 暴露的两项严重误判——坐姿全判为站姿、虚假多人检测
* **核心变更说明**：
  1. **坐姿检测重构**（`feature_extractor.py` + `run_gpu_pipeline.py`）：
     - 旧逻辑：姿态高度 > 72px → 站姿。桌子遮挡脚踝时回退到鼻-髋高度，坐姿和站姿上半身高度无差别 → 全判为站姿
     - 新逻辑：下半身可见关键点 ≤2 + 上半身 ≥3 + 质心位移 < 5px/s → **利用遮挡本身作为坐姿信号**。遮挡是问题，在这里把它变成了特征
     - 下半身可见时仍用姿态高度法（实际有效）
     - 三步判定：① 下半身遮挡+静止→坐着 ② 下半身遮挡+移动→站着 ③ 下半身可见→姿态高度法
  2. **多人假阳性过滤**（`run_gpu_pipeline.py`）：
     - 检测框最小尺寸过滤：宽或高 < 40px 的检测直接丢弃（排除背景杂物误识别）
     - 连续帧过滤：第二人需连续存在 ≥15 帧才确认（排除闪变假阳性）
     - 未达标时仅保留置信度最高的一人
* **涉及/修改的文件清单**：
  - `src/video_analysis/feature_extractor.py` (Modified, sedentary logic replaced)
  - `scripts/run_gpu_pipeline.py` (Modified, multi-person filter + matching sedentary logic)
* **执行结果与验证状态**：137/137 测试通过
* **置信度或遗留待办（TODO）**：
  - 修复效果需在 P03T01C05 上 GPU 验证
  - 遮挡阈值（lower_visible ≤2）和位移阈值（5px/s）可能需微调

---

### [2026-07-16] - 修复 v2：坐姿检测改为时间维度静止比例 + 多人过滤调参

* **当前操作动作**：第二轮修复——遮挡法失效（下半身实际可见 86%），改为基于时间维度的静止比例判定
* **核心变更说明**：
  1. **坐姿检测 v3**（`feature_extractor.py`）：
     - v1: 姿态高度法 → 桌子遮挡时坐姿上半身和站姿无区别 → 全判为活动
     - v2: 下半身遮挡 + 静止 → P03T01C05 下半身 86% 可见，失效
     - **v3: 30 秒滑动窗口静止比例法** → 过去 30s 内 > 60% 帧位移 < 5px → 坐姿
     - 核心洞察：站立会自然微调重心，坐姿可以长时间完全不动。偶而换姿势不影响判定（60% 而非 100%）
     - 新增 `_still_history` (deque, 30s * fps)
  2. **多人假阳性过滤**（`run_gpu_pipeline.py`）：
     - 检测框最小尺寸 40px
     - 第二人需连续存在 ≥15 帧
     - 未达标时保留置信度最高的一人
  3. **添加 `collections.deque` 导入**（`feature_extractor.py`）
* **涉及/修改的文件清单**：
  - `src/video_analysis/feature_extractor.py` (Modified, sedentary logic v3 + deque import)
  - `scripts/run_gpu_pipeline.py` (Modified, multi-person filter + matching sedentary logic)
  - `claude_operation_log.md` (Updated)
* **执行结果与验证状态**：137/137 测试通过
* **修复效果（P03T01C05 迭代）**：
  - v1: sedentary=0.08 active=16.05 → 全判为活动
  - v2 (遮挡法): sedentary=0.08 → 下半身未被遮挡，无效
  - v3 (静止比例 80%): sedentary=0.34 active=14.76 → 改善但不够
  - v3 (静止比例 60%): sedentary=0.46 active=9.45 → 持续改善中
* **遗留问题**：
  - social_interaction 仍为 4.84 min（单人视频，YOLO 假阳性需进一步排查）
  - 坐姿判定阈值（still_ratio=0.6）可能需要根据更多视频标定
  - 需在 10 视频全量测试后确定最优阈值

---

### [2026-07-16] - A3 Batch 1：MLLM Prompt 模板 + Qwen2.5-VL 复核引擎核心

* **当前操作动作**：创建 `configs/mllm_prompts.yaml` 和 `src/video_analysis/mllm_verifier.py`
* **核心变更说明**：
  1. `configs/mllm_prompts.yaml`：三套封闭标签 System Prompt（§6.2 JSON Schema 强制输出）
     - **long_inactivity**：区分 engaged_sedentary（看书/做手工）vs passive_sedentary（呆坐/打盹）
     - **social_interaction**：区分 family_interaction（家人）vs stranger_interaction（陌生人/推销）vs watching_tv_alone
     - **repetitive_behavior**：区分 purposeful_repetition（有目的）vs anxious_wandering（焦虑徘徊）vs compulsive_searching（强迫翻找）
     - 每套附带 2 个 Few-Shot 示例、法律约束（禁止医学诊断）
  2. `src/video_analysis/mllm_verifier.py`：
     - `MLLMVerifier` 类：Mock/Real 双模式（GPU 审批同 A1）
     - `verify(video_path, event_type, trigger_ts)` → 采样 16 帧 → 推理 → JSON 解析 → 重试(×2) → §6.2 校验
     - 失败兜底：`safe_default()` 返回 `evidence_sufficient: false`
     - `verify_batch()` 批量复核接口
* **涉及/修改的文件清单**：
  - `configs/mllm_prompts.yaml` (Created, 3 prompts + 6 few-shot examples)
  - `src/video_analysis/mllm_verifier.py` (Created, ~280 lines)
* **执行结果与验证状态**：Mock 模式三种 event_type 全通，§6.2 Schema 校验通过。137/137 测试通过。
* **置信度或遗留待办（TODO）**：
  - Real 模式需 GPU 审批后加载 Qwen2.5-VL-7B（预计 ~14GB 显存）
  - Few-Shot 示例需用 Toyota Smarthome Trimmed RGB 真实数据替换占位路径
  - A3 Batch 2 待实现：事件触发集成 + Mock 单元测试

---

### [2026-07-16] - A3 Batch 2：事件触发集成 + MLLM 单元测试

* **当前操作动作**：创建 `generate_mllm_triggers` 事件扫描函数 + 28 个单元测试
* **核心变更说明**：
  1. `generate_mllm_triggers()`：扫描 A2 `get_daily_summary()` 输出，按优先级生成 MLLM 复核事件列表
     - 徘徊/热点动作 → `repetitive_behavior`（优先级 1）
     - 社交强度 > 0.3 → `social_interaction`（优先级 2）
     - 久坐事件 > 0 或 max_stretch > 1h → `long_inactivity`（优先级 3）
     - 支持传入原始事件列表提取精确触发时间戳
  2. `tests/test_mllm_verifier.py`：28 个测试（使用 Videos_mp4/P14T14C06）
     - Prompt 模板测试（5）：封闭标签、JSON 格式、Few-Shot 完整性
     - Verifier Mock 测试（11）：三种 event_type 全路径 + §6.2 Schema 校验 + 批量复核 + 异常处理
     - 事件触发集成测试（7）：四种触发场景 + 优先级排序 + 空事件过滤
     - 降级与异常测试（2）：safe_default、无效视频降级
     - 辅助函数测试（3）：_find_event_ts
* **涉及/修改的文件清单**：
  - `src/video_analysis/mllm_verifier.py` (Modified, +60 lines trigger integration)
  - `tests/test_mllm_verifier.py` (Created, 291 lines)
* **执行结果与验证状态**：165/165 全量测试通过（A1 104 + A2 33 + A3 28），耗时 ~6 min
* **置信度或遗留待办（TODO）**：
  - 事件触发目前扫描日级汇总，实际部署应改为实时事件驱动
  - A3 Batch 3 待实现：GPU 实机加载 Qwen2.5-VL-7B 验证三种复核任务

---

### [2026-07-16] - 环境问题记录：PyTorch 版本过低，无法加载 Qwen2.5-VL

* **当前操作动作**：诊断环境兼容性问题，记录为换环境做准备
* **核心变更说明**：
  1. **问题链**：
     - 当前 PyTorch 2.1.2 + CUDA 11.8（AutoDL 默认镜像）
     - transformers 4.46.0 仅有 `Qwen2VLForConditionalGeneration`，无 `Qwen2_5_VLForConditionalGeneration`
     - transformers 4.48+ 需要 PyTorch >= 2.3（`register_pytree_node` 接口变更）
     - transformers 5.x 需要 PyTorch >= 2.4
     - 升级 PyTorch 到 2.5 可能破坏 ultralytics + numpy 现有兼容需要重测
  2. **推荐新环境**：
     - **PyTorch 2.5.1 + CUDA 12.1 + Python 3.10**（推荐方案）
     - 备选：PyTorch 2.4.1 + CUDA 11.8
     - numpy >= 1.26（PyTorch 2.5 原生支持 numpy 2.x）
     - transformers >= 4.48（含 `Qwen2_5_VLForConditionalGeneration`）
  3. **新环境启动后需验证**：
     - 165 测试全量通过（ultralytics、numpy、scipy 兼容性）
     - `from transformers import Qwen2_5_VLForConditionalGeneration` 可用
     - GPU 可用（RTX 4090 或更高）
* **涉及/修改的文件清单**：
  - `claude_operation_log.md` (环境备注)
* **执行结果与验证状态**：A3 Batch 1+2 代码已就绪，Mock 测试通过。Real 模式代码已写好但被 PyTorch 版本阻塞。
* **置信度或遗留待办（TODO）**：
  - 换环境后立即执行 A3 Batch 3：GPU 加载 Qwen2.5-VL-7B，三种 event_type 各验证一次
  - P14T14C06 作为测试视频

---

### [2026-07-16] - 新环境配置 + A3 Batch 3 准备

* **当前操作动作**：切换 PyTorch 2.5.1 环境，安装依赖，修复 Qwen2.5-VL 代码 bug，创建 A1+A2+A3 全流程脚本
* **核心变更说明**：
  1. **新环境确认**：Python 3.12 + PyTorch 2.5.1+cu124 + CUDA 12.4 + transformers 5.14.0
     - `Qwen2_5_VLForConditionalGeneration` 可正常导入（A3 阻塞解除）
     - 全部依赖安装：ultralytics 8.4.96 / scipy 1.18.0 / opencv 5.0.0 / pytest 9.1.1
  2. **Bug 修复**（`mllm_verifier.py`）：
     - 第 44 行导入：`Qwen2VLForConditionalGeneration` → `Qwen2_5_VLForConditionalGeneration`
     - 第 183 行模型加载同上修复
  3. **配置更新**（`configs/default.yaml`）：
     - 数据路径从 `/dataset/` 改为项目内相对路径 `dataset/`
     - 新增 `videos_mp4`、`doubao`、压缩包路径字段
  4. **新脚本**（`scripts/run_a1_a3_pipeline.py`）：
     - Phase 1: YOLOv8-Pose → A1 特征提取 → A2 专项检测
     - Phase 2: 卸载 YOLO → 加载 Qwen2.5-VL → `generate_mllm_triggers()` 扫描 A2 输出 → 逐个 MLLM 复核
     - 输出到 `results/A1A3/{video_name}_{timestamp}.json`
     - 默认视频: `dataset/Videos_mp4/P14T14C06.mp4`
* **涉及/修改的文件清单**：
  - `src/video_analysis/mllm_verifier.py` (Modified — Qwen2.5-VL import fix)
  - `configs/default.yaml` (Modified — dataset paths)
  - `scripts/run_a1_a3_pipeline.py` (Created — A1+A2+A3 integrated pipeline)
  - `.claude/settings.local.json` (Modified — dataset permissions)
* **执行结果与验证状态**：165/165 全量测试通过（387s）；A3 28 个 Mock 测试通过；A1A3 脚本导入成功
* **置信度或遗留待办（TODO）**：
  - GPU 就绪后立即跑 A3 Batch 3：`python scripts/run_a1_a3_pipeline.py`
  - 预期 YOLO ~1min + Qwen2.5-VL ~1min（加载 14GB 显存 + 推理）
  - 数据集尚未挂载 `/dataset/`，实际路径为项目内 `dataset/`
---
