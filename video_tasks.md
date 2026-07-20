# 同学A（视频识别与大模型辅助判断）核心任务指令说明书

本篇文档专为调用 **Claude Code** 进行自动化代码编写及系统实现而设计。文档完整梳理并整合了“挑战杯赛题作品初版方案”与“团队执行版心理健康模块方案”中关于**视频感知、人体姿态关键点提取、专项行为统计及多模态大模型（MLLM）辅助复核**的核心诉求。

---

## 一、 同学A 角色定位与核心技术栈

### 1.1 核心目标
将居家场景下的家庭摄像头视频流，转换为可重复统计、高鲁棒性的**结构化客观行为特征特征**。同时，利用本地部署的 **Qwen2.5-VL-7B-Instruct** 对传统视觉模型难以量化或存在多义性的关键视频片段进行**场景语义复核**，为系统整体输出“客观事实证据与置信度”，而非直接输出医学或心理学诊断结论。

### 1.2 核心技术栈要求
* **目标检测与姿态估计**：YOLOv8-Pose / YOLO-Pose（端侧轻量化运行）。
* **多目标跨帧跟踪**：ByteTrack / DeepSORT（实现个体身份连续性绑定）。
* **多模态大模型（MLLM）**：Qwen2.5-VL-7B-Instruct（基于事件驱动唤醒，输入10-30秒短视频序列或8-24帧关键帧）。
* **接口设计**：接口必须与底层具体视觉算法解耦，统一输出标准结构化的 JSON 特征流。

---

## 二、 运行环境与算力平台规范 (AutoDL)

为了在进行代码编写、环境调试与单元测试时精准控制显存与算力开销，现明确具体的运行硬件配置与常态化工作模式约束：

### 2.1 算力硬件环境
系统部署与测试运行于 **AutoDL 算力云平台**，其实例的基准硬件与软件栈配置如下：
* **GPU**：NVIDIA RTX 4090 (24GB VRAM) * 1 （支持动态升降配置）
* **CPU**：16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
* **内存**：120GB RAM
* **磁盘空间**：系统盘 30GB + 数据盘 50GB SSD（支持按需扩容/缩容）
* **网络与网络流**：同一地区实例共享带宽；自定义服务开放 6006、6008 HTTP 端口映射
* **基础镜像软件栈**：Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4

### 2.2 无卡模式（CPU 模式）常态工作约束
> ⚠️ **重要工作模式说明**：为了严格控制计费成本，该实例在非密集的跑批时段通常运行在 **“无卡模式”** 下。

* **无卡模式代码编写规范**：
  1. **逻辑解耦与独立运行**：所有核心业务逻辑（如任务 A1 的滑动窗口特征计算、任务 A2 的轨迹徘徊检测算法、基础数据解析与格式校验等）在编写时，必须确保**无需 GPU 参与即可在纯 CPU 环境下独立运行并通过单元测试**。
  2. **高可用降级机制**：算法框架必须具备环境感知能力。在没有显卡（即 `torch.cuda.is_available() == False`）的环境下，数据加载器、特征处理管线和非深度学习的统计/几何图算法需保持常态高可用。
  3. **单元测试隔离 (Mock)**：自动化测试（如 Pytest）中涉及神经网络前向传播（如 YOLO 姿态推理、Qwen 大模型复核）的部分，必须设计常规的 `mock` 机制，或者允许在检测到无 GPU 时自动跳过核心权重加载，转而使用本地预存的模拟 JSON 报文或时序特征张量完成全链路跑通。

* **有卡（RTX 4090）模式触发时机与审批流程**：
  1. **触发场景**：仅在执行大规模连续视频流密集跑批、大模型微调训练、或全链路多模态集成性能压测等必须依赖硬件加速的场景时，才考虑开启显卡实例进行作业。
  2. **GPU 启用审批机制**：**在编写或执行任何需要 GPU 资源的代码（如模型权重加载、训练任务、大规模推理）前，Claude Code 必须先暂停执行，并向用户明确说明预期的算力需求（如：预计运行时间、显存占用、任务必要性）。只有在获得用户明确授权（“确认使用 GPU”）后，方可尝试开启显卡实例进行作业。**

---

## 三、 本地数据集与基准数据资产管理（Toyota Smarthome）

为保证日常居家日常行为（久坐、徘徊、跌倒前置动作等）判定算法的准确性与基准对齐，系统引入了行业标准的居家行为数据集 **Toyota Smarthome dataset**。

### 3.1 已下载并上传的本地数据清单
当前环境已完成以下核心数据资产的下载与部署，具体位于/dataset中，Claude Code 在编写数据加载器（Data Loader）和测试用例时应直接挂载以下资源：
1. **Trimmed RGB Data** (`Toyota_Smarthome/trimmed/rgb/`)：裁剪好的居家日常行为短视频片段，用于算法早期视觉特征提取与 MLLM（Qwen2.5-VL）的感知微调/Prompt 验证。
2. **Trimmed Refined Skeleton Data (V1.2)** (`Toyota_Smarthome/trimmed/skeleton_v1.2/`)：经过精确修正的高质量 3D/2D 人体骨骼关键点时序数据。**此资产为任务 A1/A2 滑动窗口时序特征验证的黄金标准（Ground Truth）**。
3. **Untrimmed Annotation** (`Toyota_Smarthome/untrimmed/annotations/`)：未裁剪的长视频行为时间区间标注文件。用于验证时序徘徊、长周期久坐等复杂事件流检测器的切片和突发事件捕获能力。
4. **Untrimmed RGB Data** (`Toyota_Smarthome/untrimmed/rgb/`)：未裁剪的长视频。因全量数据集太大，暂时只上传了 10 个视频，用于跑通全流程以及验证模型可靠性。

### 3.2 数据集临时认证凭据（备用）
如需通过自动化脚本追加下载其余部分（如 Depth 数据），可使用以下尚在有效期内的临时凭据：
* **USERNAME**: `Smarthome`
* **PASSWORD**: `XoGHTITItYg=`

---

## 四、 开发任务详解（A1 - A4）

### 任务 A1：基础人体检测、跟踪与姿态特征提取
* **实现目标**：构建以原始视频流为输入、结构化特征为输出的感知基座。常态化运行于端侧或流媒体服务器，完成基础行为的无感感知。
* **Pipeline 与数据对接要求**：
  * **核心流水线**：系统应实现 `VideoStream -> YOLOv8-Pose (推理) -> ByteTrack (跟踪) -> 关键点平滑与特征计算` 的实时 Pipeline。
  * **测试兼容性**：在编写 `VideoFeatureExtractor` 时，必须实现双输入模式：
      1. **生产模式**：读取 RGB 视频流/摄像头流，实时推理提取关键点。
      2. **测试模式**：直接读取 `Toyota Smarthome V1.2 Skeleton` 文件作为特征计算的输入，以验证特征算法（如运动速度、活动时长）的数学正确性。
* **具体产出指标**（日级/小时级聚合）：
  1. **活动分钟数（activity_minutes）**：老人处于非静止状态的累计时长。
  2. **久坐/静止比例（sedentary_ratio）**：单次或累计保持坐姿、卧姿且无大幅度动作的时长占比。
  3. **房间切换次数（room_transitions）**：跨摄像头或跨区域检测到的空间位移频次。
  4. **平均/瞬时运动速度（movement_velocity）**：质心移动位移变化率。
  5. **夜间活动次数与时长（night_activity_stats）**：在设定的夜间时段（如22:00-06:00）内的起身及走动统计。
  6. **多人共现时长（multi_person_duration）**：画面中同时出现两个及以上人体目标的时间段。
* **Claude Code 代码编写指令**：
  > “请编写一个基于 Python 的视频特征提取基类 `VideoFeatureExtractor`。系统必须以 RGB 视频流为主要输入，内置 YOLOv8-Pose 和 ByteTrack 模块，实现人体关键点提取与多目标跟踪。同时，为保证算法验证的准确性，请增加一个针对 `Toyota Smarthome V1.2 Skeleton` 格式的数据加载适配器，使得算法能直接对预存的骨骼关键点进行特征计算。结合滑动窗口逻辑，每隔固定周期，计算并输出包含上述 6 项基础指标的结构化时序字典。确保 Pipeline 接口高度解耦，数据输入端可灵活切换为‘摄像头实时流’或‘预存骨骼 JSON 文件’。”

### 任务 A2：专项高危与异常行为统计模块
* **实现目标**：针对防跌倒、心理筛查和反诈骗三大场景，编写特定的时序行为判定算法。
* **数据资产对接要求**：
  * 利用 **Untrimmed Annotation** 提供的长视频连续行为时间轴，模拟连续不间断视频流输入。以此验证算法在面临长周期、多噪声背景下，对“徘徊”和“异常久坐”的检出率与误报率。
* **需要实现的专项算法逻辑**：
  1. **重复路线/无目的徘徊检测**：对老人的运动轨迹进行空间建图，计算路径重复率和重合度。如果老人在固定区域（如客厅、玄关）在短时间内连续呈现异常往复轨迹，标记为“徘徊”。
  2. **重复开关/反复寻找行为**：统计老人针对特定空间或物品（如反复开门、开关抽屉、在固定区域翻找物品）的动作频次。
  3. **长时间静止与异常久坐/久卧**：当检测到人体处于静卧或特定沙发椅座区域超过预设阈值，且骨骼关键点微弱变化时，触发警示。
  4. **昼夜节律偏移分析**：对比老人的“个体化行为基线”，统计其实际起床时间、入睡时间、午休时长是否较基线发生显著时间偏离。
  5. **社交互动强度检测**：当多人共现时，计算人体间的空间距离、朝向角度（是否面对面）以及肢体交互频次，量化社交互动。
* **质量控制要求**：所有输出特征必须附带 **时间窗（time_window）**、**有效监测时长（valid_duration）** 和 **检测质量置信度（confidence_score）**，以防因摄像头网络离线、画面严重遮挡、老人外出导致空数据而被模型误判为“活动量骤减”。

### 任务 A3：Qwen2.5-VL-7B 事件驱动复核引擎
* **实现目标**：由于传统视觉算法难以区分“坐着阅读与无明显活动”、“与来访者交谈与独自看电视”等复杂语义。本任务要求在异常行为触发后，截取 **10—30秒关键视频片段** 或 **8—24帧代表性关键帧画面** 送入 Qwen2.5-VL。
* **数据资产对接要求**：
  * 可抽取本地 **Trimmed RGB Data** 中包含不同动作标签（如 eating, reading, watching TV）的视频片段，对 Qwen2.5-VL 进行少样本提示（Few-Shot Prompting）设计，确保其闭环 JSON 输出的准确率。
* **Prompt 约束与策略**：
  * **必须采用封闭标签（Closed-set labels）**，禁止大模型自由发挥。
  * **必须采用固定提示词模板**，并强制开启 **JSON Schema 结构化输出**。
* **大模型核心复核任务清单**：
  * **语义区分一**：判断老人是在“坐着看书/做手工（积极认知活动）”还是“单纯呆坐/无目的打盹（消极退缩行为）”。
  * **语义区分二**：判断老人的社交场景是“与上门推销者/陌生人交谈（潜在反诈风险）”还是“与家庭成员正常互动”或“独自看电视”。
  * **语义区分三**：识别老人是否在特定区域进行“无目的反复徘徊”或“焦虑地反复查看/翻找”。
* **Claude Code 编写指令**：
  > “请使用 PyTorch 和 Transformers 库编写一个 Qwen2.5-VL-7B 辅助判断类 `MLLMVerifier`。设计专门的视频分帧采样函数，能够兼容解析 `Toyota Smarthome` 的 Trimmed RGB 短视频资产，将其均匀采样为 16 帧。编写严格的 System Prompt，要求模型仅从可见证据出发进行场景描述，必须返回符合指定 JSON 格式的判定结果（包含：环境类别、核心行为标签、可见物品清单、判断置信度、不确定项说明）。必须进行异常捕获，防止解析非标准 JSON 失败。”

### 任务 A4：多模型一致性校验与高可靠拒判机制
* **实现目标**：构建前端专用 CV 模型（YOLO-Pose）与后端大模型（Qwen2.5-VL）的交叉校验逻辑。
* **业务规则设计**：
  1. **双重一致性确认**：仅当专用视觉特征计算结果与 MLLM 的语义复核证据指向一致时，系统才提升整体风险可信度（如：时序分析提示徘徊 + Qwen2.5-VL 复核确认存在徘徊证据 $
ightarrow$ 触发中高风险预警）。
  2. **拒判机制（Refusal Mechanism）**：若两者冲突（如时序模型判定为跌倒趋势，但大模型确认为正常系鞋带动作）、画面发生严重光照不足、人体被大型家具彻底遮挡或证据不足时，系统输出 `status: "uncertain"`，不进行强等级心理报警，自动进入延迟复核或请求多视角交叉验证。

---

## 五、 跨场景功能协同设计（同学A如何赋能三大模块）

同学A所输出的视频特征与 MLLM 证据，将作为底层公共能力，同时对跌倒前置防控、心理健康和反诈骗三大模块进行支撑。在编写代码时，需确保数据字典能无缝对接以下三个业务逻辑：

### 5.1 赋能动作防跌倒模块（2.3.1 & 4.1.2 场景）
* **骨骼时序流对接**：同学A提取的人体 2D 骨骼关键点时序序列（坐标、倾斜角、加速度、支撑脚稳定性），**由 YOLO-Pose 实时推理得出**。该流数据（需支持以 15fps-30fps 的频率）必须实时向后端的 LSTM / 动态图神经网络（DGNN）输送，用于识别“稳定活动 $\rightarrow$ 失衡瞬态 $\rightarrow$ 跌倒发生”的过程。
* **环境动态异动**：通过轻量异动检测，在发生水杯打翻、液体泼洒、杂物坠落瞬间截取画面，通知大模型判断是否在老人未来行走路径上形成了“水渍溢洒”或“静态绊倒隐患”。
* **多摄像头多视角调度**：若客厅主摄像头发生家具遮挡或人体转身导致关键点丢失，需设计接口接收相邻视角摄像头的画面，执行时空交叉几何对齐与置信度加权融合。

### 5.2 赋能心理健康风险连续感知（2.3.2 & 4.1.3 场景）
* **长周期特征喂入**：同学A统计的日级/周级活动分钟数、昼夜节律偏移量、重复开关频次、社交互动时长，将作为客观特征向量（Behavioral Feature Vector）。
* **多模态编码融合接口**：编写特征导出接口，将上述视频特征向量与同学B负责的语音情绪特征向量、问卷模块的心理画像向量（PHQ-8/PSQI等）在统一低维语义空间进行对齐，再喂入 Transformer 融合网络进行跨模态计算。

### 5.3 赋能诈骗风险阻断模块（2.3.3 & 4.1.1 场景）
* **入户人脸多帧聚合比对**：从门口摄像头画面定位人脸，提取特征向量，与家庭安全人员库（包含家人、物业、亲友等）进行比对。需编写连续多帧人脸聚合算法，采用最大置信度和时序一致性判定来访身份，若非可信人员，触发“陌生人到访记录”。
* **室内涉诈交互动作与敏感物品检测**：在客厅或门口区域，专门开启针对银行卡、身份证、收据合同、保健品、宣传单页、POS机等特定敏感物品的检测目标。同时通过多目标跟踪，计算室内是否有“多人围坐劝说”、“引导老人签字”、“展示二维码”的高危交互模式。
---

## 六、 核心数据接口规范与 JSON Schema

为确保代码编写时与系统其他模块（问卷模块、语音模块、云端融合中心）实现无缝对接，同学A的算法模块必须严格遵守以下两套数据接口定义：

### 6.1 视频结构化行为特征输出接口（日级/周期级统计）
```json
{
  "user_id": "STRING",
  "date": "YYYY-MM-DD",
  "daily_metrics": {
    "active_minutes": "FLOAT (分/日)",
    "sedentary_ratio": "FLOAT (%)",
    "room_transition_count": "INT (次/日)",
    "night_activity_count": "INT (次/夜)",
    "social_interaction_minutes": "FLOAT (分/日)",
    "repetitive_path_count": "INT (次/日)",
    "movement_speed": "FLOAT (m/s 或 相对值)",
    "coverage_minutes": "FLOAT (分/日)",
    "feature_confidence": "FLOAT (0-1)"
  }
}
```

### 6.1.1 A2 专项行为检测扩展字段（`a2_special_behavior`）

以下字段由 `SpecialBehaviorDetector`（A2 模块）产出，作为 `daily_metrics` 的补充，附加在输出 JSON 的 `a2_special_behavior` 键下：

```json
{
  "a2_special_behavior": {
    "daily_repetitive_path_count": "INT (次/日)",
    "daily_hotspot_action_count": "INT (次/日)",
    "daily_prolonged_inactive_count": "INT (次/日)",
    "max_inactive_stretch_sec": "FLOAT (秒)",
    "daily_avg_social_intensity": "FLOAT (0-1)",
    "circadian": {
      "date": "YYYY-MM-DD",
      "wake_time": "FLOAT (起床时间，小时)",
      "sleep_time": "FLOAT (入睡时间，小时)",
      "nap_duration_minutes": "FLOAT (午休时长，分钟)",
      "baseline_wake_mean": "FLOAT (基线平均起床时间)",
      "baseline_sleep_mean": "FLOAT (基线平均入睡时间)",
      "wake_offset_hours": "FLOAT (起床时间偏移，小时)",
      "sleep_offset_hours": "FLOAT (入睡时间偏移，小时)",
      "is_circadian_disturbed": "BOOL (节律是否异常)",
      "baseline_days_count": "INT (基线天数)",
      "confidence_score": "FLOAT (0-1)"
    }
  }
}
```

| 字段 | 来源检测器 | 含义 |
|:---|:---|:---|
| `daily_repetitive_path_count` | `RepetitivePathDetector` | 当日检测到的重复路径/无目的徘徊事件次数。路径重合度超过 40% 阈值时触发 |
| `daily_hotspot_action_count` | `RepeatedActionDetector` | 当日检测到的反复进出同一区域的热点动作次数。进出频次超过阈值（默认 4 次）时触发 |
| `daily_prolonged_inactive_count` | `ProlongedInactivityDetector` | 当日异常久坐/久卧事件次数。连续静止超过 2 小时触发，1 小时预警告。附带骨骼关键点微动分析 |
| `max_inactive_stretch_sec` | `ProlongedInactivityDetector` | 当日最长连续静止时长（秒）。用于评估最严重的一次久坐事件 |
| `daily_avg_social_intensity` | `SocialInteractionAnalyzer` | 当日社交互动强度均值 (0-1)。综合多人共现比例、空间距离、面对面朝向角度加权计算 |
| `circadian` | `CircadianRhythmAnalyzer` | 昼夜节律分析结果。对比个体多日基线，计算起床/入睡时间偏移量。偏移超过 2 小时标记 `is_circadian_disturbed: true` |

> **注意**：`daily_metrics.repetitive_path_count` 由 A1 硬编码值改为从 `a2_special_behavior.daily_repetitive_path_count` 回填，确保 §6.1 Schema 保持兼容。

---

### 6.2 Qwen2.5-VL-7B 事件复核输出 JSON Schema（大模型返回约束）
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Qwen2.5_VL_Event_Verification",
  "type": "object",
  "properties": {
    "event_type": {
      "type": "string",
      "enum": ["long_inactivity", "social_interaction", "repetitive_behavior"]
    },
    "cooling_period": {
      "type": "integer",
      "enum": [60, 120],
      "description": "冷却周期（秒）"
    },
    "num_of_occurrences": {
      "type": "integer",
      "minimum": 0,
      "description": "事件在观察周期内的发生次数"
    },
    "observable_evidence": {
      "type": "string",
      "description": "只描述画面可见事实，如：连续坐姿、无明显肢体运动、桌面有书"
    },
    "analytical_summary": {
      "type": "string",
      "description": "一句简短的分析总结，格式为：‘老人出现[现象]，疑似[结论]，需要关注’，禁止提及诊断性结论如抑郁等，只给出建议。"
    },
    "start_sec": { "type": "number" },
    "end_sec": { "type": "number" },
    "activity_state": {
      "type": "string",
      "enum": ["active", "sedentary", "uncertain"]
    },
    "social_context": {
      "type": "string",
      "enum": ["alone", "co_present", "interacting", "uncertain"]
    },
    "repetition_type": {
      "type": "string",
      "enum": ["same_route", "repeated_search", "none", "uncertain"]
    },
    "quality_flags": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["occlusion", "low_light", "off_camera"]
      }
    },
    "evidence_sufficient": { "type": "boolean" }
  },
  "required": [
    "event_type", 
    "cooling_period",
    "num_of_occurrences",
    "observable_evidence", 
    "start_sec", 
    "end_sec", 
    "activity_state", 
    "social_context", 
    "repetition_type", 
    "evidence_sufficient"
  ]
}
```

---

## 七、 阶段性交付物与开发排期建议

同学A在调用 Claude Code 进行迭代开发时，建议遵循以下一周开发排期：

| 时间节点 | 核心任务 | Claude Code 辅助编写重点 | 最终交付产物标准 |
| :--- | :--- | :--- | :--- |
| **第 1-2 天** | 视频感知基座与数据加载架构 | 编写 `VideoFeatureExtractor`，实现 **“视频流(生产模式)”** 与 **“骨骼数据(验证模式)”** 双路加载。构建滑动窗口特征计算逻辑。 | 能够稳定从 RGB 视频解析，并使用 Skeleton 数据集进行精度对齐；输出符合 `daily_metrics` 定义的结构化特征字典。 |
| **第 3-4 天** | 专项行为算法与逻辑解耦 | 结合 `Untrimmed Annotation` 模拟真实长视频流，编写徘徊、重复行为判定算法，确保逻辑与基础指标解耦。 | 完成 `SpecialBehaviorDetector` 模块，测试用例必须覆盖 `repetitive_path_count` 等业务字段的准确计算。 |
| **第 5 天** | 集成 Qwen2.5-VL 事件复核引擎 | 配置本地 vLLM 推理管线；**强制约束 Prompt 必须返回完全符合第 6.2 节 Schema 定义的 JSON**，禁止 markdown 包装。 | 模型返回的 JSON 必须通过 `json.loads()` 校验，且字段包含 `observable_evidence`, `activity_state` 等标准枚举值。 |
| **第 6-7 天** | 一致性校验与拒判闭环集成 | 编写 A4 级校验逻辑，基于 `evidence_sufficient` 标志位实现拒判。对接问卷与语音模块的接口联调。 | 完整闭环：视频流输入 $\rightarrow$ 异常触发 $\rightarrow$ 大模型复核 $\rightarrow$ **基于 `evidence_sufficient` 的真值过滤** $\rightarrow$ 最终结构化输出。 |

---

## 八、 Claude Code 自动化开发审计规范（操作日志要求）

为了确保自动化代码编写过程可追溯、可审计，并方便团队其他成员（如同学B、同学C）或云端集成人员进行阶段性成果复核，**Claude Code 在每次执行完阶段性操作（包括但不限于：增删改代码、运行测试、解决 Bug、配置环境等）之后，必须简要记录操作内容到指定的标准日志文件中**。

### 8.1 标准操作日志文件名
统一规定日志文件名为：**`claude_operation_log.md`**（存放在项目的根目录下）。

### 8.2 日志记录格式与约束
每次操作完成后，Claude Code 需采用 **增量追加（Append）** 的方式向该文件中写入一条记录。严禁覆盖或删除历史日志。日志必须遵循以下格式：

```markdown
### [%TIMESTAMP%] - 任务阶段名称

* **当前操作动作**：[例如：创建新文件 / 修改 `VideoFeatureExtractor` 类的滑窗更新逻辑 / 运行单元测试]
* **核心变更说明**：
  1. [具体变更点 1，例如：增加了 update_window 方法，采用 collections.deque 限制窗口最大长度]
  2. [具体变更点 2，例如：优化了追踪目标丢失时的置信度衰减系数]
* **涉及/修改的文件清单**：
  - `src/video_analysis/extractor.py` (Modified)
  - `tests/test_extractor.py` (Created)
* **执行结果与验证状态**：[例如：通过本地单元测试 `pytest tests/test_extractor.py`，滑窗计算耗时 < 1.5ms，无报错]
* **置信度或遗留待办（TODO）**：[例如：YOLOv8-Pose 显存占用尚待在端侧 16GB 设备上进行长周期压力测试]
---
```

### 8.3 触发日志追加的时机
* 每次通过 `Claude Code` 成功生成一个全新模块代码后。
* 针对异常处理、拒判机制等复杂业务逻辑进行代码重构或 Bug 修复后。
* 每次本地测试成功或环境依赖库（如 `weasyprint`, `openpyxl`, `transformers` 等）发生变更后。


## 九、工程维护与协作规范（Engineering & Maintenance）

为确保项目具备良好的可维护性、可扩展性与团队协作能力，本项目遵循**“文档即资产，记录即审计”**的工程管理原则。所有开发人员及自动化开发工具（如 Claude Code）均应保持文档与代码同步更新，确保项目架构、接口定义及运行说明始终与当前代码实现保持一致。

### 9.1 README 编写规范

项目根目录下的 **`README.md`** 为整个工程的唯一入口文档（Entry Point），必须保持内容完整、结构清晰，并随着项目迭代持续维护。README 至少应包含以下内容：

* **项目概览（Project Overview）**：介绍项目背景、核心功能、设计目标及适用场景。
* **系统架构（System Architecture）**：使用 Mermaid 绘制整体架构图、算法流程图或模块调用关系图，说明系统各组成部分之间的协作关系。
* **快速开始（Getting Started）**：包含环境安装方式、依赖配置、AutoDL 镜像要求、Toyota Smarthome 数据目录挂载方式及项目启动流程。
* **工程目录（Directory Structure）**：展示项目目录结构，并对主要模块、核心文件及功能职责进行说明。
* **接口规范（API & Schemas）**：描述核心 JSON 数据格式、接口参数说明、业务触发条件及相关 Schema 定义。
* **维护说明（Maintenance Protocol）**：记录操作日志位置、版本更新规范、测试运行方式及项目维护流程。

### 9.2 文档维护规范

所有工程文档均须遵循统一的 Markdown 编写规范，并保证内容具备良好的可读性与长期维护性。

* 所有代码示例必须标注对应语言（如 `python`、`json`、`bash`、`markdown` 等），便于 IDE、GitHub 及 CI 工具进行语法高亮与静态检查。
* 涉及环境部署、依赖安装、数据集挂载等内容时，应保证文档具有**幂等性（Idempotent）**，多次按照文档执行均能够得到一致的运行结果。
* 严禁在 README 或其他文档中引用个人开发环境中的临时目录、硬编码路径或已失效的资源地址。
* 当发生模型替换、算法升级、模块重构、接口调整等架构级变更时，必须同步更新 README 中对应的架构图、接口说明及运行示例，确保文档始终反映项目的最新状态。

### 9.3 README 更新触发时机

为了保证文档与代码保持强一致性，README 必须在以下场景完成同步更新：

* 每完成**第七部分：阶段性交付物**中的一个重要开发阶段（如骨骼提取管线完成、MLLM 推理模块集成完成、系统联调测试完成等）后，应及时更新 README 中对应的架构描述、运行流程及测试说明。
* 每次更新 README 前，应首先查阅 **`claude_operation_log.md`** 最近的操作记录，确保文档中的技术实现、运行限制及功能描述与当前代码版本保持一致。
* 每次 Git Commit 若涉及接口签名修改、核心算法重构、模块职责调整等重要变更时，必须在提交代码前同步更新 README 中对应的 API 接口说明、架构设计及相关文档内容。

### 9.4 版本控制与 Git 同步规范

Git 同步操作仅在用户明确要求时执行，Claude Code 不会在每个 Batch 完成后自动提交或推送。

当用户要求 Git 同步时，遵循以下规范：

* **Git 同步流程**：
  1. `git add -A` — 暂存所有变更
  2. `git commit -m "<描述性提交信息>"` — 提交变更
  3. `git push` — 推送到远程仓库
* **提交信息格式**：`<type>(<scope>): <description>`，如 `feat(A1): Batch 1 — project structure and config`
* **每个 Batch 独立提交**：一个 Batch 对应一个 commit，不合并
* **提交前检查**：确保所有模块通过导入验证和基础功能测试后，方可提交