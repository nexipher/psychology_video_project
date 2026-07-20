# 同学A — 视频识别与大模型辅助判断 实施计划

> 基于 `video_tasks.md` v1.0，制定可执行的工程落地计划。
> 所有编码、测试、架构设计均严格遵循文档中的「工程化原则」和「数据接口规范」。

---

## 一、项目当前状态

| 项目 | 状态 |
|:---|:---|
| 环境 | Python 3.12 + PyTorch 2.5.1+cu124 + CUDA 12.4 |
| GPU | NVIDIA RTX 4090 (23.5 GB VRAM) |
| 数据集 | `dataset/Videos_mp4/` 10 个视频；Toyota Smarthome 压缩包（暂不解压） |
| A1 | ✅ 全部完成，104 tests |
| A2 | ✅ 全部完成，33 tests |
| A3 | ✅ Mock 完成 + GPU 验证通过（P14T14C06, P10T07C04），28 tests |
| A4 | ❌ 未开始 |
| README | ✅ 已完成 v4.0 |
| 操作日志 | ✅ 持续更新 |

---

## 二、工程目录结构设计

```
psychology_video_project/
├── README.md                          # 项目入口文档
├── video_tasks.md                     # 核心任务指令书（只读参考）
├── plan.md                            # 本文件 — 实施计划
├── claude_operation_log.md            # 自动化操作审计日志
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── video_analysis/
│   │   ├── __init__.py
│   │   ├── config.py                  # 全局配置（路径、阈值、时间窗参数）
│   │   ├── data_loader.py             # 双模式数据加载器（RGB视频 / Skeleton JSON）
│   │   ├── video_stream.py            # 视频流抽象层（文件/摄像头/RTSP）
│   │   ├── feature_extractor.py       # A1: VideoFeatureExtractor 基类 + 滑动窗口
│   │   ├── tracker.py                 # ByteTrack 多目标跟踪封装
│   │   ├── pose_estimator.py          # YOLOv8-Pose 推理封装（GPU/Mock 双模式）
│   │   ├── sliding_window.py          # 通用滑动窗口数据结构
│   │   ├── aggregator.py              # A1: 日级/周期级指标聚合器
│   │   ├── special_behavior.py        # A2: SpecialBehaviorDetector（徘徊/重复/久坐/节律/社交）
│   │   ├── mllm_verifier.py           # A3: MLLMVerifier — Qwen2.5-VL 事件复核引擎
│   │   ├── cross_validator.py         # A4: 多模型一致性校验与拒判机制
│   │   └── pipeline.py                # 顶层 Pipeline 编排器
│   │
│   └── utils/
│       ├── __init__.py
│       ├── schema_validator.py         # JSON Schema 校验工具
│       ├── frame_sampler.py            # 视频帧均匀采样器（给 MLLM 用）
│       └── skeleton_parser.py          # Toyota Smarthome Skeleton V1.2 解析器
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures（Mock 模型、模拟数据）
│   ├── test_data_loader.py
│   ├── test_video_stream.py
│   ├── test_feature_extractor.py
│   ├── test_tracker.py
│   ├── test_sliding_window.py
│   ├── test_aggregator.py
│   ├── test_special_behavior.py
│   ├── test_mllm_verifier.py
│   ├── test_cross_validator.py
│   ├── test_pipeline.py
│   └── test_schema_validator.py
│
└── configs/
    ├── default.yaml                    # 默认配置
    └── mllm_prompts.yaml              # Qwen2.5-VL System Prompt 模板
```

---

## 三、任务分解与排期（7 天）

### 第 1–2 天：任务 A1 — 视频感知基座与数据加载架构

**目标**：构建 `VideoStream → YOLOv8-Pose → ByteTrack → 关键点平滑 → 滑动窗口特征计算` 的实时 Pipeline。

| # | 子任务 | 产出 | 验收标准 |
|:--|:---|:---|:---|
| A1.1 | `config.py` + `default.yaml` | 全局配置（模型路径、阈值、窗口参数） | 所有模块可通过 `config` 统一获取参数 |
| A1.2 | `skeleton_parser.py` | Toyota Smarthome Skeleton V1.2 格式解析器 | 能正确解析骨骼 JSON，输出标准化的 `(T, K, 3)` 关键点张量 |
| A1.3 | `data_loader.py` | 双模式 DataLoader（RGB视频 / Skeleton JSON） | 生产模式读视频 → 帧迭代器；测试模式读骨骼 → 关键点迭代器 |
| A1.4 | `video_stream.py` | 视频流抽象层（文件/摄像头/RTSP） | 统一 `read()` 接口，支持 `cv2.VideoCapture` 封装 |
| A1.5 | `pose_estimator.py` | YOLOv8-Pose 推理封装 | GPU 模式正常推理；CPU 模式优雅降级（返回 Mock 数据或跳过） |
| A1.6 | `tracker.py` | ByteTrack 多目标跟踪封装 | 输入检测框+关键点 → 输出带 track_id 的目标序列 |
| A1.7 | `sliding_window.py` | 通用滑动窗口数据结构 | O(1) 插入/淘汰，支持固定时间窗长度，线程安全 |
| A1.8 | `feature_extractor.py` | `VideoFeatureExtractor` — 6 项基础指标计算 | 正确输出 activity_minutes / sedentary_ratio / room_transitions / movement_velocity / night_activity_stats / multi_person_duration |
| A1.9 | `aggregator.py` | 日级/周期级聚合器 | 输出严格符合 §6.1 `daily_metrics` JSON Schema |
| A1.10 | 单元测试 | 覆盖 A1 所有模块（CPU 模式，Mock GPU） | `pytest tests/` 全绿，测试覆盖率 ≥ 85% |

**工程化要求**：
- 所有核心特征计算逻辑（滑动窗口、指标聚合）**纯 CPU 运行**，无需 GPU
- `pose_estimator.py` 在无 GPU 时自动降级，不抛出异常
- 双输入模式通过 `data_loader.py` 的策略模式切换

---

### 第 3–4 天：任务 A2 — 专项高危与异常行为统计模块

**目标**：实现 5 项专项行为判定算法，利用 Untrimmed Annotation 验证长周期检出率。

| # | 子任务 | 产出 | 验收标准 |
|:--|:---|:---|:---|
| A2.1 | 轨迹空间建图算法 | 基于网格/栅格的运动轨迹记录器 | 支持自定义网格分辨率，O(1) 位置查询 |
| A2.2 | 徘徊检测 `RepetitivePathDetector` | 重复路线/无目的徘徊判定 | 输出 `repetitive_path_count`，通过 Untrimmed Annotation 验证 |
| A2.3 | 重复动作检测 `RepeatedActionDetector` | 反复开关/翻找行为统计 | 基于特定区域（门、抽屉）的动作频次统计 |
| A2.4 | 久坐/久卧检测 `ProlongedInactivityDetector` | 长时间静止异常检测 | 超过预设阈值（如 2h）触发标记，附带 effective_duration |
| A2.5 | 昼夜节律偏移 `CircadianRhythmAnalyzer` | 起床/入睡/午休时间偏移分析 | 对比个体基线，输出偏移量（小时） |
| A2.6 | 社交互动强度 `SocialInteractionAnalyzer` | 多人共现时的空间距离/朝向/肢体交互量化 | 输出 social_interaction_minutes 和 interaction_intensity |
| A2.7 | `special_behavior.py` 总装 | `SpecialBehaviorDetector` 统一入口 | 所有子检测器可插拔，输出带 time_window / valid_duration / confidence_score |
| A2.8 | 单元测试 | 覆盖所有检测器 + Untrimmed Annotation 模拟数据 | `pytest tests/test_special_behavior.py` 全绿 |

**工程化要求**：
- 所有检测算法**纯 CPU 运行**（几何/统计算法，不依赖深度学习）
- 每个检测器输出必须包含 `time_window`、`valid_duration`、`confidence_score` 三要素
- 与 A1 基础指标模块完全解耦，通过接口通信

---

### 第 5 天：任务 A3 — Qwen2.5-VL-7B 事件驱动复核引擎

**目标**：A2 检测器触发异常后，实时调用 Qwen2.5-VL 进行语义复核。支持冷却期机制，同一类型事件冷却期内仅计数不调用 MLLM。

| # | 子任务 | 产出 | 状态 |
|:--|:---|:---|:---|
| A3.1 | `frame_sampler.py` | 视频均匀帧采样器（16 帧） | ✅ |
| A3.2 | `mllm_prompts.yaml` | System Prompt ×3（long_inactivity / social_interaction / repetitive_behavior），各含 Few-Shot | ✅ |
| A3.3 | `mllm_verifier.py` | `MLLMVerifier` 类（Mock/Real 双模式） + `generate_mllm_triggers()` 事件扫描 | ✅ |
| A3.4 | JSON 异常兜底 | 非标准 JSON 重试 ×2 → `_safe_default()` → `evidence_sufficient: false` | ✅ |
| A3.5 | `schema_validator.py` | §6.1 + §6.2 JSON Schema 校验，自动修复缺失字段 | ✅ |
| A3.6 | 单元测试 | 28 tests，覆盖 Prompt 模板 / Mock 推理 / 事件触发集成 / 异常处理 | ✅ |

#### A3 实时事件驱动架构

```
每帧 → A2 检测器内部实时判定
         │
         ├─ 触发条件满足 → 检查冷却期
         │                    │
         │          冷却期内 ──┴── 冷却期外
         │           │                  │
         │    num_of_occurrences++   记录时间戳 trigger_ts
         │    不调用 MLLM            截取 16 帧 → Qwen2.5-VL 推理
         │                          标记冷却期开始
         │
         └─ 未触发 → 继续监控
```

#### 冷却期设计

| event_type | 冷却期 | 冷却期内行为 |
|:---|:---|:---|
| `repetitive_behavior` | **60s** | 仅累加 `num_of_occurrences`，不调用 MLLM |
| `social_interaction` | **120s** | 仅累加 `num_of_occurrences`，不调用 MLLM |
| `long_inactivity` | **120s** | 仅累加 `num_of_occurrences`，不调用 MLLM |

冷却期从每次 MLLM 调用完成时开始计时，期间 A2 若再次检测到同一 event_type，仅增加 `num_of_occurrences` 计数，不发起新的 MLLM 请求。冷却期结束后首帧再次触发时，启动新一轮 MLLM 复核。

#### YOLO + Qwen 共驻显存

- YOLOv8n-pose：~45 MB VRAM
- Qwen2.5-VL-7B：~15.5 GB VRAM
- 合计：~15.5 GB / 23.5 GB，完全可同时驻留
- **不需要加载/卸载切换**，A2 触发后直接调 A3，零模型加载延迟

#### GPU 验证结果

| 视频 | 时长 | A3 事件 | 结果 |
|:---|:---|:---|:---|
| P14T14C06 | 9.6min | 1 (repetitive_behavior) | same_route, evidence_sufficient ✅ |
| P10T07C04 | 19.5min | 2 (repetitive + social) | both evidence_sufficient ✅ |

#### 已修复的 Bug

| Bug | 根因 | 修复 |
|:---|:---|:---|
| `Qwen2VLForConditionalGeneration` 不可用 | transformers 5.x 中类名为 `Qwen2_5_VLForConditionalGeneration` | `mllm_verifier.py:44,183` 更新导入 |
| `analytical_summary` 缺失 | video_tasks.md §6.2 定义但代码三处遗漏 | `schema_validator.py` + `mllm_prompts.yaml` + `mllm_verifier.py` 同步补全 |
| social_interaction 输出 `event_type: "family_interaction"` | Prompt Task 子类型标签与 Output Format 固定值冲突，模型选了前者 | 三个 Prompt 统一加固：去掉 Task 中的子类型标签，Output Format 和字段说明中两次强调 event_type 固定值 |
| `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | AutoDL 设 `OMP_NUM_THREADS=0` | Pipeline 脚本顶部在 C 库导入前 `del os.environ["OMP_NUM_THREADS"]` |

#### §6.2 Schema 新增字段

用户已更新 `video_tasks.md`，新增两个 required 字段：

```json
"cooling_period": {
  "type": "integer",
  "enum": [60, 120],
  "description": "冷却周期（秒）"
},
"num_of_occurrences": {
  "type": "integer",
  "minimum": 0,
  "description": "事件在观察周期内的发生次数"
}
```

- `cooling_period`：该事件类型的冷却期设置（60s or 120s）
- `num_of_occurrences`：冷却期内 A2 检测到同类型异常的次数（含触发 MLLM 的那一次）

---

### 第 6–7 天：任务 A4 — 一致性校验与拒判闭环集成

**目标**：构建 CV 模型与 MLLM 的交叉校验逻辑，实现完整闭环。

| # | 子任务 | 产出 | 验收标准 |
|:--|:---|:---|:---|
| A4.1 | `cross_validator.py` | `CrossValidator` 双重一致性确认 | CV 结果 + MLLM 结果 → 一致则提升置信度，冲突则降级 |
| A4.2 | 拒判机制 | 基于 `evidence_sufficient` 的真值过滤 | 光线不足/遮挡/证据不足 → `status: "uncertain"`，不触发强报警 |
| A4.3 | `pipeline.py` 顶层编排 | 完整 Pipeline：视频输入 → A1特征 → A2异常检测 → A3复核 → A4校验 → 最终输出 | 端到端可运行 |
| A4.4 | 集成测试 | 端到端测试（使用 Skeleton 数据 + Mock MLLM） | 全链路跑通，无断点 |
| A4.5 | README 同步更新 | 填写 README.md 全部章节 | 符合 §9.1 规范 |

---

## 四、核心架构决策

### 4.1 双输入模式设计（策略模式）

```
                    ┌─────────────────────┐
                    │   DataLoaderFactory  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
   ┌──────────────────────┐        ┌──────────────────────┐
   │  RGBVideoLoader      │        │  SkeletonLoader      │
   │  (生产模式)           │        │  (测试/验证模式)      │
   │  - 读取视频帧         │        │  - 解析骨骼 JSON      │
   │  - 调用 YOLO-Pose    │        │  - 直接输出关键点      │
   │  - 输出关键点          │        │  - 绕过神经网络       │
   └──────────┬───────────┘        └──────────┬───────────┘
              └────────────────┬──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  FeatureExtractor    │
                    │  (消费标准关键点流)   │
                    └─────────────────────┘
```

### 4.2 GPU 审批流程

```
代码需要 GPU → Claude Code 暂停
             → 向用户说明：预计运行时间 / 显存占用 / 任务必要性
             → 用户确认「确认使用 GPU」
             → 执行 GPU 代码
             → 完成后切回 CPU 模式
```

### 4.3 MLLM 事件驱动唤醒流

```
A1 特征流 ──→ A2 异常检测 ──→ 触发事件?
                                    │
                           No ──────┴────── Yes
                            │                  │
                            ▼                  ▼
                       继续监控        截取 10-30s 关键片段
                                            │
                                            ▼
                                     A3 Qwen2.5-VL 复核
                                            │
                                            ▼
                                     A4 交叉校验
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                         一致 (evidence_sufficient)     冲突/不足
                              │                           │
                              ▼                           ▼
                         提升置信度                   status: "uncertain"
                         触发预警 (可选)              延迟复核 / 不报警
```

---

## 五、数据接口实现要点

### 5.1 日级统计输出（§6.1）

```python
# aggregator.py 输出签名
def aggregate_daily(user_id: str, date: str, window_results: List[WindowMetrics]) -> DailyMetrics:
    """返回严格符合 daily_metrics Schema 的字典"""
    return {
        "user_id": str,
        "date": "YYYY-MM-DD",
        "daily_metrics": {
            "active_minutes": float,
            "sedentary_ratio": float,
            "room_transition_count": int,
            "night_activity_count": int,
            "social_interaction_minutes": float,
            "repetitive_path_count": int,
            "movement_speed": float,
            "coverage_minutes": float,
            "feature_confidence": float,
        }
    }
```

### 5.2 MLLM 复核输出（§6.2）

- 使用 `jsonschema` 库对 Qwen 返回做自动校验
- 缺失 `required` 字段时自动填充默认值并标记 `evidence_sufficient: false`
- 重试逻辑：首次失败 → 补充 prompt 要求 → 二次失败 → 返回 safe default

---

## 六、测试策略

| 层级 | 策略 | 运行模式 |
|:---|:---|:---|
| 单元测试 | 每个模块独立测试，Mock 所有 GPU 依赖 | CPU（无卡模式） |
| 集成测试 | A1→A2→A3→A4 链路测试，Mock MLLM | CPU（无卡模式） |
| 精度验证 | 使用 Skeleton V1.2 黄金标准对比特征计算结果 | CPU（无卡模式） |
| E2E 测试 | 全链路 + 真实模型（需 GPU 审批） | GPU（有卡模式） |

### Mock 策略

- `pose_estimator.py`：Mock 返回预生成的 17 点 COCO 格式关键点
- `mllm_verifier.py`：Mock 返回符合 §6.2 Schema 的标准 JSON
- `tracker.py`：Mock 返回稳定的 track_id 序列

---

## 七、风险与应对

| 风险 | 影响 | 应对措施 |
|:---|:---|:---|
| `/dataset/` 未挂载 | 无法使用 Toyota Smarthome 数据 | 编写合成数据生成器，优先完成代码逻辑 |
| RTX 4090 不可用 | 无法测试真实模型推理 | Mock 机制保证全链路可用，GPU 可用后再验证 |
| Qwen2.5-VL JSON 输出不稳定 | 复核结果不可解析 | 重试 + schema 校验 + 安全默认值兜底 |
| Untrimmed 视频仅 10 个 | 长周期测试样本不足 | 循环播放模拟长时间流，验证算法稳定性 |

---

## 八、A1 判定算法详解（当前实现 v3）

### 8.1 静止/活动判定（帧级）

**输入**：YOLOv8-Pose 输出的 17 点关键点 → 多目标跟踪(ByteTrack) → 髋部质心追踪

**每帧计算**：

```
centroid = (left_hip + right_hip) / 2          # 髋部中点（质心）
max_disp = ||centroid_t - centroid_{t-15}||    # 1 秒累计位移（15fps × 1s）
is_still_frame = max_disp < 5.0                # 1 秒内移动 < 5px → 静止
```

### 8.2 坐姿判定（时间维度）

**核心假设**：站立会自然微调重心，坐姿可以长时间完全不动。

```
_still_history = deque(maxlen=450)             # 30 秒 × 15fps，存储 bool
still_ratio = sum(True) / 450                  # 过去 30 秒静止帧占比

is_truly_sedentary = (
    len(_still_history) >= 450                 # 积累够 30 秒数据
    AND still_ratio >= 0.6                     # ≥60% 帧静止 → 判定坐姿
)

is_standing = NOT is_truly_sedentary
```

**阈值设计**：60%（而非 100%）——允许坐姿中偶而换姿势而不打破整体判定。

### 8.3 窗口级聚合（每 60s 输出一次）

| 指标 | 计算方式 |
|:---|:---|
| `active_frames` | `person_count > 0 AND is_sedentary == False` |
| `sedentary_frames` | `is_sedentary == True AND person_count > 0` |
| `active_ratio` | `active_frames / window_total_frames` |
| `sedentary_ratio` | `sedentary_frames / window_total_frames` |

### 8.4 A2 久坐检测（独立运行）

`ProlongedInactivityDetector` 独立追踪 `is_sedentary` 信号：

| 阶段 | 阈值 | 动作 |
|:---|:---|:---|
| 正常 | 连续静止 < 1h | 无 |
| 预警告 | 连续静止 ≥ 1h | `warning_triggered = True` |
| 异常久坐 | 连续静止 ≥ 2h | `prolonged_triggered = True` |

附加骨骼微动分析：静止期间关键点标准差越小 → 置信度越高。

### 8.5 多人假阳性过滤

```
1. 检测框过滤：width < 40px OR height < 40px → 丢弃（背景杂物）
2. 连续帧过滤：第二人需连续存在 ≥ 15 帧才确认
3. 未达标 → 只保留置信度最高的 1 人
```

### 8.6 关键参数速查

| 参数 | 值 | 位置 | 用途 |
|:---|:---|:---|:---|
| `max_disp` 阈值 | 5 px | `feature_extractor.py` | 单帧静止判定 |
| `_still_history` | 450 帧 (30s) | `feature_extractor.py` | 静止回溯窗口 |
| `still_ratio` 阈值 | 60% | `feature_extractor.py` | 坐姿判定 |
| 窗口大小 | 300s (5min) | `run_gpu_pipeline.py` | 聚合窗口 |
| 窗口步长 | 60s (1min) | `run_gpu_pipeline.py` | 输出频率 |
| 多人最小框 | 40px | `run_gpu_pipeline.py` | 假阳性过滤 |
| 多人连续帧 | 15 帧 | `run_gpu_pipeline.py` | 假阳性过滤 |
| A2 久坐预警 | 3600s | `special_behavior.py` | 连续静止 1h |
| A2 久坐异常 | 7200s | `special_behavior.py` | 连续静止 2h |

### 8.7 已知局限

| 问题 | 影响 | 状态 |
|:---|:---|:---|
| 坐姿换姿势破坏静止比例 | 短暂移动使 still_ratio 跌破 60% | 60% 阈值已缓解 |
| YOLO nano 远端关键点不稳定 | 髋/踝置信度低，质心抖动 | 见下节 |
| 单人视频产生虚假多人检测 | social_interaction 偏高 | 40px+15帧过滤已缓解 |
| 昼夜节律仅单日数据 | 基线不可靠，全部默认值 | 需多日数据 |

---

## 九、YOLO 模型限制与已知问题

### 9.1 yolov8n-pose 在居家场景的表现

| 维度 | 表现 | 影响 |
|:---|:---|:---|
| 人物检测 | ✅ 良好，检出率 >95% | 基本可靠 |
| 上半身关键点 | ✅ 正常（鼻/眼/肩稳定） | 社交朝向可用 |
| 下半身关键点 | ⚠️ YOLO nano 对远端关节检测不稳定 | 质心计算波动大 |
| 多人检测 | ⚠️ 偶发假阳性（杂物误识为第二人） | social_interaction 需过滤 |
| 桌子遮挡 | ⚠️ 无法穿透，但遮挡本身成为坐姿信号 | 已利用 |

### 9.2 无法通过纯 CV 解决的问题（需 A3 MLLM 复核）

| 场景 | CV 局限 | 需要 MLLM 做什么 |
|:---|:---|:---|
| 坐着看书 vs 呆坐打盹 | 姿态都是坐姿，无法区分 | 判断是否有书/手机，是否闭眼 |
| 与家人交谈 vs 陌生人到访 | 仅知两人共处，不知身份 | 判断面部特征、衣着、交互氛围 |
| 反复进出房间 vs 正常走动 | 路径重合度高但目的不明 | 判断是否焦虑、是否在找东西 |

---

## 十、当前进度与下一步

### 已完成

| 阶段 | 内容 | 测试 |
|:---|:---|:---|
| A1 全部 | 视频感知基座 + 特征提取 + 日级聚合 | 104 passed ✅ |
| A2 全部 | 5 项专项行为检测器 + 总装 | 33 passed ✅ |
| A3 Mock | Prompt 模板 + Verifier + Schema 校验 + 异常兜底 | 28 passed ✅ |
| A3 GPU | Qwen2.5-VL-7B 实机验证（P14T14C06, P10T07C04 各两次） | ✅ |
| 管线集成 | A1+A2+A3 全流程脚本 `run_a1_a3_pipeline.py` | ✅ |
| 文档 | README.md v4.0 + plan.md + claude_operation_log.md | ✅ |

### 下一步

1. **A3 实时化改造**：实现冷却期机制、YOLO+Qwen 共驻显存、逐帧 A2 信号监听 → 实时触发 MLLM
2. **同步 §6.2 Schema**：`cooling_period` / `num_of_occurrences` 字段落实到 `schema_validator.py` + `mllm_prompts.yaml` + `mllm_verifier.py`
3. **批量 GPU 验证**：10 个视频全量跑 A1+A2+A3
4. **A4 开发**：多模型一致性校验与拒判机制

---

## 十一、A3 实时化改造计划

### 11.1 改造目标

将 A3 从「视频结束后批量扫描日级汇总」改为「逐帧监听 A2 信号，实时触发 MLLM 复核」。

```
改造前 (batch):
  视频播完 → get_daily_summary() → generate_mllm_triggers() → 逐个 MLLM

改造后 (streaming):
  每帧 → A2 检测器内部判定 → 触发信号 → A3EventDispatcher → 检查冷却期 → 实时 MLLM
```

### 11.2 改造范围

| 模块 | 改造内容 | 优先级 |
|:---|:---|:---|
| `schema_validator.py` | §6.2 Schema 新增 `cooling_period` + `num_of_occurrences` | P0 |
| `mllm_prompts.yaml` | 三个 Prompt 的 Output Format 同步新增两个字段 | P0 |
| `mllm_verifier.py` | Mock 响应 + safe_default 补全新字段；`_inference_real` 输出包含新字段 | P0 |
| `special_behavior.py` | 每个子检测器新增冷却期状态 + 触发回调钩子 | P1 |
| **新文件** `event_dispatcher.py` | `A3EventDispatcher`：冷却期检查、MLLM 调用调度、事件队列管理 | P1 |
| **新脚本** `run_streaming_pipeline.py` | 流式管线：YOLO+Qwen 共驻、逐帧 A1→A2→A3 | P1 |
| `run_a1_a3_pipeline.py` | 旧的 batch 模式保留不变，新脚本独立 | — |
| `tests/` | 新增冷却期逻辑单测 + A3EventDispatcher 单测 + 流式集成测试 | P2 |

### 11.3 冷却期状态机

```
A2 触发 ──→ 检查 _cooldown_until[event_type]
                │
    未在冷却期 ──┴── 正在冷却期
         │                 │
  1. 记录 trigger_ts    累加 _pending_count[event_type]++
  2. 标记冷却期开始      不调用 MLLM（跳过）
  3. _pending_count = 1
  4. 调用 MLLM.verify()
  5. MLLM 返回后，用 _pending_count 覆盖 result["num_of_occurrences"]
  6. 重置 _pending_count = 0
```

**`num_of_occurrences` 语义**：

- 首次 A2 触发 → 调 MLLM，`num_of_occurrences` 填的是本次触发 + 之前累积未上报的次数
- 冷却期内 → 不调 MLLM，仅累加 `_pending_count`
- 冷却期结束后再次触发 → 新一轮 MLLM 调用，`num_of_occurrences` 包含本轮冷却期内累积的所有次数
- Qwen2.5-VL 只看 16 帧画面，无法感知冷却期次数 → MLLM 返回后由 `A3EventDispatcher` 覆盖该字段

### 11.4 A3EventDispatcher 设计

```python
class A3EventDispatcher:
    """A2→A3 实时事件调度器。

    职责：
    1. 持有 MLLMVerifier 引用（Qwen 已加载，驻留显存）
    2. 接收 A2 触发信号 (event_type, trigger_ts)
    3. 检查冷却期，决定是否调用 MLLM
    4. 管理事件队列，收集所有 MLLM 复核结果
    """

    COOLDOWN = {
        "repetitive_behavior": 60,   # 秒
        "social_interaction": 120,
        "long_inactivity": 120,
    }

    def __init__(self, verifier: MLLMVerifier, video_path: str):
        self._verifier = verifier           # Qwen 已加载
        self._video_path = video_path
        self._cooldown_until: dict[str, float] = {}   # event_type → 冷却结束时间戳
        self._pending_count: dict[str, int] = {}       # event_type → 冷却期内累计触发次数
        self._results: list[dict] = []                # 收集所有 MLLM 结果

    def on_trigger(self, event_type: str, trigger_ts: float) -> Optional[dict]:
        """A2 检测器触发回调。

        冷却期内：仅递增 _pending_count，返回 None
        冷却期外：调用 MLLM，用 _pending_count 覆盖 result["num_of_occurrences"]
        """
        now = trigger_ts

        # 累加触发计数（冷却期内或首次触发都计入）
        self._pending_count[event_type] = self._pending_count.get(event_type, 0) + 1

        if event_type in self._cooldown_until and now < self._cooldown_until[event_type]:
            return None  # 冷却期内，仅计数

        # 冷却期外：设置冷却期，调用 MLLM
        cooldown = self.COOLDOWN.get(event_type, 60)
        self._cooldown_until[event_type] = now + cooldown

        result = self._verifier.verify(
            video_path=self._video_path,
            event_type=event_type,
            trigger_ts=trigger_ts,
        )
        result["cooling_period"] = cooldown
        result["num_of_occurrences"] = self._pending_count[event_type]  # 覆盖 MLLM 返回值
        self._pending_count[event_type] = 0  # 重置计数器

        self._results.append(result)
        return result

    def flush(self) -> list[dict]:
        """返回所有已收集的 MLLM 复核结果。"""
        return self._results
```

### 11.5 A2 检测器改造点

每个需实时触发的子检测器增加：

```python
self._trigger_callback: Optional[Callable] = None   # 对外回调钩子
```

`_trigger_callback` 签名统一为 `(event_type: str, trigger_ts: float) → None`，调用方只负责通知「发生了异常」，计次和冷却期判断由 `A3EventDispatcher.on_trigger()` 内部完成。

**以 `RepeatedActionDetector` 为例**：

```python
def update(self, centroid_x, centroid_y, timestamp):
    # ... 原有逻辑 ...
    if hotspot_detected:
        if self._trigger_callback:
            self._trigger_callback("repetitive_behavior", timestamp)
```

#### 11.5.1 `repetitive_behavior` 共享冷却期

`RepetitivePathDetector`（徘徊）和 `RepeatedActionDetector`（热点动作）映射到同一个 MLLM `event_type = "repetitive_behavior"`，二者**共享**一个 60s 冷却期。

```
RepetitivePathDetector  ──→ callback("repetitive_behavior", ts)
RepeatedActionDetector   ──→ callback("repetitive_behavior", ts)
                                    │
                          A3EventDispatcher.on_trigger("repetitive_behavior", ts)
                                    │
                          检查 _cooldown_until["repetitive_behavior"]
```

任一个检测器触发 → 两个检测器同时进入 60s 冷却。MLLM 的 Prompt 本身会区分 `same_route` vs `repeated_search`，无需 A2 侧做区分。

#### 11.5.2 人物离开画面处理

当 `person_count == 0`（画面中无人）时，所有检测器：

- **暂停累积**：不产生新触发信号
- **不重置已有计数**：`A3EventDispatcher._pending_count` 和 `_cooldown_until` 保持不变
- **人物回来时**：从零重新累积检测器内部状态（`_still_start` 归零、`recent_positions` 清空等），但冷却期状态不受影响

```python
# SpecialBehaviorDetector.update() 中
if centroid_x is None:  # 无人
    self._pause_all_detectors()
    return  # 不调用任何 _trigger_callback
```

#### 11.5.3 A2 检测器 → A3 event_type 映射

| A2 检测器 | A3 event_type | 冷却期 | 触发条件 |
|:---|:---|:---|:---|
| `RepetitivePathDetector` | `repetitive_behavior` | 60s | 10min 窗口内同一条边出现 ≥3 次 + 重合度 >40% |
| `RepeatedActionDetector` | `repetitive_behavior` | 60s（共享）| 同一网格区域 10min 内进出 ≥4 次 |
| `SocialInteractionAnalyzer` | `social_interaction` | 120s | 加权社交强度 > 0.3 |
| `ProlongedInactivityDetector` | `long_inactivity` | 120s | 连续静止 ≥2h（1h 预警告不触发 MLLM）|
| `CircadianRhythmAnalyzer` | — | — | 不参与实时触发（需要多日基线，仅在日终汇总输出）|

### 11.6 YOLO + Qwen 共驻显存策略

```
Pipeline 启动:
  1. 加载 YOLOv8n-pose     → ~45 MB VRAM
  2. 加载 Qwen2.5-VL-7B    → ~15.5 GB VRAM
  3. 创建 A3EventDispatcher(verifier, video_path)
  4. 注册回调到 A2 检测器

主循环 (逐帧):
  1. YOLO 推理 → keypoints, bboxes
  2. ByteTrack → track_ids
  3. A1 feature_extractor.process_frame()
  4. A2 behavior.update() → 内部检测 → 触发回调 → A3EventDispatcher.on_trigger()
  5. （A3 推理异步或同步，见 11.7）

收尾:
  1. A2 behavior.flush()
  2. A3EventDispatcher.flush() → 收集所有 MLLM 结果
  3. 保存合并 JSON
```

### 11.7 同步 vs 异步推理

| 方案 | 优点 | 缺点 | 决定 |
|:---|:---|:---|:---|
| 同步（A3 推理期间阻塞 A2） | 简单，无需队列 | 推理 10s 期间丢失帧检测 | ❌ |
| 异步（A3 推理时 A2 继续检测） | 不丢检测 | 需要线程安全 + 队列 | ❌ |

**最终决定**：采用冷却期机制后，同类型事件 60-120s 才可能触发一次 MLLM，10s 推理时间相对于冷却期很短，且不同类型的事件冷却期独立。**采用同步方式**——MLLM 推理时继续逐帧处理 A1+A2，仅在 `on_trigger()` 实际调用 MLLM 时短暂阻塞主循环 ~10s，且 60-120s 才可能阻塞一次，不影响整体吞吐。

### 11.8 输出 JSON 结构变化

改造前（batch，单次汇总）：
```json
{
  "daily_metrics": { ... },
  "a2_special_behavior": { ... },
  "a3_mllm_verification": [
    { "event_type": "repetitive_behavior", ... },
    { "event_type": "social_interaction", ... }
  ]
}
```

改造后（streaming，同类型可多次出现）：
```json
{
  "daily_metrics": { ... },
  "a2_special_behavior": { ... },
  "a3_mllm_verification": [
    {
      "event_type": "repetitive_behavior",
      "cooling_period": 60,
      "num_of_occurrences": 3,
      "start_sec": 120.0,
      "end_sec": 135.0,
      ...
    },
    {
      "event_type": "repetitive_behavior",
      "cooling_period": 60,
      "num_of_occurrences": 2,
      "start_sec": 195.0,
      "end_sec": 210.0,
      ...
    },
    {
      "event_type": "social_interaction",
      "cooling_period": 120,
      "num_of_occurrences": 1,
      "start_sec": 300.0,
      "end_sec": 315.0,
      ...
    }
  ]
}
```

- `start_sec` / `end_sec` 为异常点在实际视频中的时间戳（00:00:00 起始）
- 同一 `event_type` 可在冷却期结束后再次触发，产生多条记录
- `num_of_occurrences` 反映了冷却期内 A2 检测到的异常次数

### 11.9 改造步骤

| Step | 内容 | 产出 | 验收 |
|:---|:---|:---|:---|
| 1 | 更新 §6.2 Schema + Prompts + Mocks | `schema_validator.py`, `mllm_prompts.yaml`, `mllm_verifier.py` 同步新增两字段 | 28 tests 通过 + 新字段可解析 |
| 2 | 创建 `A3EventDispatcher` | `src/video_analysis/event_dispatcher.py` | 冷却期逻辑单元测试 |
| 3 | A2 检测器加回调钩子 + 冷却期计数器 | `special_behavior.py` 5 个检测器 | 33 tests 通过 + 回调触发测试 |
| 4 | 创建流式管线脚本 | `scripts/run_streaming_pipeline.py` | 与 batch 模式对比，结果一致性 |
| 5 | 单视频 GPU 验证 | P14T14C06 流式跑通 | JSON 包含 `start_sec`/`end_sec` 实际时间戳 |
| 6 | 10 视频全量 GPU 跑批 | 全量 A1+A2+A3 输出 | 所有视频有 MLLM 复核结果 |

---

> 📋 计划版本: v5.0 | 更新日期: 2026-07-20 | 基于: `video_tasks.md` + `agent.md`
