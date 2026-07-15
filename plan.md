# 同学A — 视频识别与大模型辅助判断 实施计划

> 基于 `video_tasks.md` v1.0，制定可执行的工程落地计划。
> 所有编码、测试、架构设计均严格遵循文档中的「工程化原则」和「数据接口规范」。

---

## 一、项目当前状态

| 项目 | 状态 |
|:---|:---|
| 源代码 (`src/video_analysis/`) | 已清空，需从头搭建 |
| 测试 (`tests/`) | 已清空，需从头编写 |
| README | 空白，待填充 |
| 操作日志 | 空白，待启用 |
| 数据集 (`/dataset/`) | 尚未挂载到当前实例 |

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

**目标**：事件触发后截取 10–30s 关键片段或 8–24 帧，送入 Qwen2.5-VL 进行语义复核。

| # | 子任务 | 产出 | 验收标准 |
|:--|:---|:---|:---|
| A3.1 | `frame_sampler.py` | 视频均匀帧采样器 | 支持固定帧数（8/16/24）均匀采样，兼容 Trimmed RGB 短视频 |
| A3.2 | `mllm_prompts.yaml` | System Prompt 模板（含 Few-Shot 示例） | 封闭标签、JSON Schema 强制输出、禁止 Markdown 包装 |
| A3.3 | `mllm_verifier.py` | `MLLMVerifier` 类 | 加载 Qwen2.5-VL-7B，执行推理，返回严格符合 §6.2 Schema 的 JSON |
| A3.4 | JSON 异常兜底 | 非标准 JSON 解析失败时的重试/降级逻辑 | 最多重试 2 次，失败后返回 `{"evidence_sufficient": false, ...}` |
| A3.5 | `schema_validator.py` | 通用 JSON Schema 校验器 | 对 MLLM 输出做最终格式校验，确保字段完整 |
| A3.6 | 单元测试（Mock MLLM） | 用模拟 JSON 响应覆盖三种 event_type 分支 | `pytest tests/test_mllm_verifier.py` 全绿 |

**工程化要求**：
- Qwen2.5-VL 加载/推理**必须先获取 GPU 审批**，默认使用 Mock 模式跑测试
- Prompt 必须采用封闭标签（`enum` 约束），禁止大模型自由发挥
- 输出 JSON 必须通过 `json.loads()` 校验，且所有 `required` 字段齐全

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

## 八、下一步行动

1. **立即启动**：创建目录结构，编写 `config.py` 和 `skeleton_parser.py`
2. **按顺序推进**：严格遵循 A1 → A2 → A3 → A4 依赖链
3. **持续交付**：每完成一个子任务即追加操作日志、运行测试

---

> 📋 计划版本: v2.0 | 创建日期: 2026-07-15 | 基于: `video_tasks.md`
