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
  - numpy 2.2.6 与 torch 2.1.2 存在兼容性警告（_ARRAY_API），在无 GPU mock 模式下不影响使用
---
