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

## 二、 开发任务详解（A1 - A4）

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

## 三、 跨场景功能协同设计（同学A如何赋能三大模块）

同学A所输出的视频特征与 MLLM 证据，将作为底层公共能力，同时对跌倒前置防控、心理健康和反诈骗三大模块进行支撑。在编写代码时，需确保数据字典能无缝对接以下三个业务逻辑：

### 3.1 赋能动作防跌倒模块（2.3.1 & 4.1.2 场景）
* **骨骼时序流对接**：同学A提取的人体 2D 骨骼关键点时序序列（坐标、倾斜角、加速度、支撑脚稳定性），**由 YOLO-Pose 实时推理得出**。该流数据（需支持以 15fps-30fps 的频率）必须实时向后端的 LSTM / 动态图神经网络（DGNN）输送，用于识别“稳定活动 $\rightarrow$ 失衡瞬态 $\rightarrow$ 跌倒发生”的过程。
* **环境动态异动**：通过轻量异动检测，在发生水杯打翻、液体泼洒、杂物坠落瞬间截取画面，通知大模型判断是否在老人未来行走路径上形成了“水渍溢洒”或“静态绊倒隐患”。
* **多摄像头多视角调度**：若客厅主摄像头发生家具遮挡或人体转身导致关键点丢失，需设计接口接收相邻视角摄像头的画面，执行时空交叉几何对齐与置信度加权融合。

### 3.2 赋能心理健康风险连续感知（2.3.2 & 4.1.3 场景）
* **长周期特征喂入**：同学A统计的日级/周级活动分钟数、昼夜节律偏移量、重复开关频次、社交互动时长，将作为客观特征向量（Behavioral Feature Vector）。
* **多模态编码融合接口**：编写特征导出接口，将上述视频特征向量与同学B负责的语音情绪特征向量、问卷模块的心理画像向量（PHQ-8/PSQI等）在统一低维语义空间进行对齐，再喂入 Transformer 融合网络进行跨模态计算。

### 3.3 赋能诈骗风险阻断模块（2.3.3 & 4.1.1 场景）
* **入户人脸多帧聚合比对**：从门口摄像头画面定位人脸，提取特征向量，与家庭安全人员库（包含家人、物业、亲友等）进行比对。需编写连续多帧人脸聚合算法，采用最大置信度和时序一致性判定来访身份，若非可信人员，触发“陌生人到访记录”。
* **室内涉诈交互动作与敏感物品检测**：在客厅或门口区域，专门开启针对银行卡、身份证、收据合同、保健品、宣传单页、POS机等特定敏感物品的检测目标。同时通过多目标跟踪，计算室内是否有“多人围坐劝说”、“引导老人签字”、“展示二维码”的高危交互模式。
---

## 四、 核心数据接口规范与 JSON Schema

为确保代码编写时与系统其他模块（问卷模块、语音模块、云端融合中心）实现无缝对接，同学A的算法模块必须严格遵守以下两套数据接口定义：

### 4.1 视频结构化行为特征输出接口（日级/周期级统计）
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

| 字段 | 类型 | 中文含义 | 计算方式 |
|:---|:---|:---|:---|
| `active_minutes` | float | 活跃分钟数 | 帧级 `person_count>0 AND NOT is_sedentary` 累计分钟数 |
| `sedentary_ratio` | float | 静止/久坐比例 | 30s 内 >60% 帧质心位移 <5px（基于 v3 静止比例法） |
| `room_transition_count` | int | 房间切换次数 | 髋部质心跨越 200px 网格边界的累计次数 |
| `night_activity_count` | int | 夜间活动次数 | 虚拟时钟落在 22:00-06:00 期间的活跃事件计数 |
| `social_interaction_minutes` | float | 多人共现分钟数 | YOLO 检测到 ≥2 人的累计分钟数（含假阳性过滤） |
| `repetitive_path_count` | int | 重复路径次数 | 由 A2 `daily_repetitive_path_count` 回填 |
| `movement_speed` | float | 平均运动速度 | 髋部质心帧间位移均值（相对像素值/秒） |
| `coverage_minutes` | float | 有效监测分钟数 | 至少检测到 1 人的时间长度 |
| `feature_confidence` | float | A1 特征置信度 | 综合检测稳定性、遮挡率、多人假阳性比例 (0-1) |

### 4.1.1 A2 专项行为检测扩展字段（`a2_special_behavior`）

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

### 4.2 Qwen2.5-VL-7B 事件复核输出 JSON Schema（大模型返回约束）
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

| 字段 | 类型 | 中文含义 | 枚举值 / 约束 |
|:---|:---|:---|:---|
| `event_type` | string | 事件类型 | `long_inactivity` / `social_interaction` / `repetitive_behavior` |
| `cooling_period` | int | 冷却周期（秒） | 60 (repetitive) / 120 (social, inactivity) |
| `num_of_occurrences` | int | 冷却期内同类异常发生次数 | ≥0，由 A3EventDispatcher 填入，覆盖 MLLM 返回值 |
| `observable_evidence` | string | 画面可见事实描述 | 只描述可见内容，如"连续坐姿、桌面有书"，不推断 |
| `analytical_summary` | string | 一句话分析总结 | 格式："老人出现[现象]，疑似[结论]，需要关注"，禁止医学诊断 |
| `start_sec` | number | 事件窗口起始秒数 | 视频内实际时间戳（00:00:00 起始） |
| `end_sec` | number | 事件窗口结束秒数 | 视频内实际时间戳，通常 = start_sec + 10~20s |
| `activity_state` | string | 人物活动状态 | `active` (活动) / `sedentary` (静止) / `uncertain` (不确定) |
| `social_context` | string | 社交场景上下文 | `alone` (独处) / `co_present` (共处无互动) / `interacting` (互动中) / `uncertain` |
| `repetition_type` | string | 重复行为类型 | `same_route` (固定路线重复) / `repeated_search` (反复翻找同一位置) / `none` / `uncertain` |
| `quality_flags` | array | 画面质量标记 | `occlusion` (遮挡) / `low_light` (光线不足) / `off_camera` (不在画面内) |
| `evidence_sufficient` | bool | 证据是否充分 | `false` 时不触发强等级报警，走 A4 拒判流程 |

---

## 五、 阶段性交付物与开发排期建议

同学A在调用 Claude Code 进行迭代开发时，建议遵循以下一周开发排期：

| 时间节点 | 核心任务 | Claude Code 辅助编写重点 | 最终交付产物标准 |
| :--- | :--- | :--- | :--- |
| **第 1-2 天** | 视频感知基座与数据加载架构 | 编写 `VideoFeatureExtractor`，实现 **“视频流(生产模式)”** 与 **“骨骼数据(验证模式)”** 双路加载。构建滑动窗口特征计算逻辑。 | 能够稳定从 RGB 视频解析，并使用 Skeleton 数据集进行精度对齐；输出符合 `daily_metrics` 定义的结构化特征字典。 |
| **第 3-4 天** | 专项行为算法与逻辑解耦 | 结合 `Untrimmed Annotation` 模拟真实长视频流，编写徘徊、重复行为判定算法，确保逻辑与基础指标解耦。 | 完成 `SpecialBehaviorDetector` 模块，测试用例必须覆盖 `repetitive_path_count` 等业务字段的准确计算。 |
| **第 5 天** | 集成 Qwen2.5-VL 事件复核引擎 | 配置本地 vLLM 推理管线；**强制约束 Prompt 必须返回完全符合第 6.2 节 Schema 定义的 JSON**，禁止 markdown 包装。 | 模型返回的 JSON 必须通过 `json.loads()` 校验，且字段包含 `observable_evidence`, `activity_state` 等标准枚举值。 |
| **第 6-7 天** | 一致性校验与拒判闭环集成 | 编写 A4 级校验逻辑，基于 `evidence_sufficient` 标志位实现拒判。对接问卷与语音模块的接口联调。 | 完整闭环：视频流输入 $\rightarrow$ 异常触发 $\rightarrow$ 大模型复核 $\rightarrow$ **基于 `evidence_sufficient` 的真值过滤** $\rightarrow$ 最终结构化输出。 |

---

