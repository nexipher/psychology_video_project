# Claude Code 自动化开发与工程协作规范

## 一、 工程生命周期与文件驱动规范（核心控制流）

为了确保 Claude Code 的开发过程具有高度的计划性、渐进性和可审计性，项目根目录维护四个核心控制文件：`{project}_tasks.md`、`plan.md`、`claude_operation_log.md` 和 `README.md`。开发必须严格遵循“文件驱动”与“小步快跑”原则。

### 1.1 四轨文件定义与生命周期
*   **`{project}_tasks.md`（项目原始需求，只读）**：
    *   **定义**：存放项目的宏观需求和一级大节（如 1, 2, 3, 4）。
    *   **修改时机**：仅在项目创建时初始化，后期几乎不会更改。Claude Code 禁止擅自修改此文件。
*   **`plan.md`（动态开发计划书，动态追加）**：
    *   **定义**：将 `tasks.md` 的大节拆解为可操作的二级小节（如 1.1, 1.2, 1.3……2.1, 2.2）。
    *   **修改时机 1（项目启动）**：根据 `tasks.md` 的内容，一次性列出全盘项目开展计划并写入 `plan.md`。
    *   **修改时机 2（后期调整）**：后期如需调整或追加工程功能，必须先将详细调整计划追加到 `plan.md` 中（如 5.1, 5.2），严禁无计划盲目写代码。
*   **`claude_operation_log.md`（操作审计日志，实时追加）**：
    *   **定义**：记录 Claude Code 的技术审计轨迹。
    *   **修改时机**：每当完成一个二级小节（如 2.1 或 2.2）的开发与测试后，必须立即更新。
*   **`README.md`（项目主文档，阶段更新）**：
    *   **定义**：面向人类用户的项目说明书。
    *   **修改时机**：每完成一个大节内容（如 整个第 2 大节完成）之后，统一更新 `README.md`。

### 1.2 “小步快跑”原子化 Commit 规范
Claude Code 在开发时必须拒绝“大包大揽”式的一次性重构，必须采取小步更改推送策略：
1.  **原子化开发**：每修改/实现一个小功能点（对应 `plan.md` 中的一个小节，如 1.1），必须单独进行一次本地 `git commit`。
2.  **提交前置条件**：Commit 前必须在无卡或有卡模式下运行通过该功能点的验证测试（导入测试或单元测试）。

---

## 二、 Claude Code 自动化开发审计规范（操作日志要求）

为了确保自动化代码编写过程可追溯、可审计，Claude Code 在每次完成二级小节（如 1.1, 2.1）或处理突发 Bug 之后，必须简要记录操作内容到 `claude_operation_log.md`。

### 2.1 日志记录格式与约束
每次操作完成后，Claude Code 需采用 **增量追加（Append）** 的方式向该文件中写入一条记录。严禁覆盖或删除历史日志。日志必须遵循以下格式：

```markdown
### [%TIMESTAMP%] - [对应 Plan.md 小节编号] 任务阶段名称

* **当前操作动作**：[例如：创建新文件 / 修改逻辑 / 运行单元测试]
* **对应计划锚点**：[例如：实现 plan.md 中的 2.1 小节]
* **核心变更说明**：
  1. [具体变更点 1，例如：增加了 xxx 方法，采用 xxx 以 xxx ]
  2. [具体变更点 2，例如：优化了追踪目标丢失时的置信度衰减系数]
* **涉及/修改的文件清单**：
  - `src/main.py` (Modified)
  - `tests/test_extractor.py` (Created)
* **执行结果与验证状态**：[例如：通过本地单元测试 `pytest tests/test_extractor.py`，无报错]
* **置信度或遗留待办（TODO）**：[例如：显存占用尚待在端侧 16GB 设备上进行长周期压力测试]
---
```

---

## 三、 运行环境与算力平台规范 (AutoDL)

### 3.1 算力硬件环境
部署与测试运行于 **AutoDL 算力云平台**：
*   **GPU**：NVIDIA RTX 4090 (24GB VRAM) * 1
*   **基础镜像**：Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4

### 3.2 无卡模式（CPU 模式）常态工作约束
> ⚠️ **重要工作模式说明**：为了严格控制计费成本，实例在非密集跑批时段通常运行在 “无卡模式” 下。

1.  **无卡模式代码编写规范**：所有核心业务逻辑在编写时，必须确保无需 GPU 参与即可在纯 CPU 环境下独立运行并通过单元测试。
2.  **单元测试隔离 (Mock)**：自动化测试（如 Pytest）中涉及神经网络前向传播（如 Qwen 大模型）的部分，必须 design 常规的 `mock` 机制，在检测到无 GPU 时自动跳过核心权重加载。
3.  **GPU 启用审批机制**：在编写或执行任何需要 GPU 资源的代码（如模型权重加载、训练任务、大规模推理）前，Claude Code 必须先暂停执行，并向用户明确说明预期的算力需求。只有在获得用户明确授权（“确认使用 GPU”）后，方可尝试开启显卡实例进行作业。

---

## 四、 本地数据集与基准数据资产管理

为保证日常居家日常行为（久坐、徘徊、跌倒前置动作等）判定算法的准确性与基准对齐，系统引入了行业标准的居家行为数据集 **Toyota Smarthome dataset**。

### 4.1 已下载并上传的本地数据清单
当前环境已完成以下核心数据资产的下载与部署，具体位于/dataset中，Claude Code 在编写数据加载器（Data Loader）和测试用例时应直接挂载以下资源：
1. **Trimmed RGB Data** (`Toyota_Smarthome/trimmed/rgb/`)：裁剪好的居家日常行为短视频片段，用于算法早期视觉特征提取与 MLLM（Qwen2.5-VL）的感知微调/Prompt 验证。
2. **Trimmed Refined Skeleton Data (V1.2)** (`Toyota_Smarthome/trimmed/skeleton_v1.2/`)：经过精确修正的高质量 3D/2D 人体骨骼关键点时序数据。**此资产为任务 A1/A2 滑动窗口时序特征验证的黄金标准（Ground Truth）**。
3. **Untrimmed Annotation** (`Toyota_Smarthome/untrimmed/annotations/`)：未裁剪的长视频行为时间区间标注文件。用于验证时序徘徊、长周期久坐等复杂事件流检测器的切片和突发事件捕获能力。
4. **Untrimmed RGB Data** (`Toyota_Smarthome/untrimmed/rgb/`)：未裁剪的长视频。因全量数据集太大，暂时只上传了 10 个视频，用于跑通全流程以及验证模型可靠性。

### 4.2 数据集临时认证凭据（备用）
如需通过自动化脚本追加下载其余部分（如 Depth 数据），可使用以下尚在有效期内的临时凭据：
* **USERNAME**: `Smarthome`
* **PASSWORD**: `XoGHTITItYg=`

### 4.3 MLLM 模型资产
*   **模型名称**：Qwen2.5-VL-7B-Instruct
*   **路径**：`/root/autodl-tmp/psychology_video_project/models/models/qwen--Qwen2.5-VL-7B-Instruct/snapshots/master`
*   **显存需求**：FP16 推理 ~16 GB，加载前必须获得用户 GPU 授权（见 §3.2）。

---

## 五、 工程维护与协作规范（Engineering & Maintenance）

### 5.1 版本控制与 Git 同步规范
Claude Code 在每个原子小节（如 1.1, 1.2）完成并验证通过后自动在本机提交。

Git 同步时，遵循以下规范：
*   **Git 同步流程**：
    1.  `git add -A` — 暂存所有变更
    2.  `git commit -m "<描述性提交信息>"` — 提交变更
*   **提交信息格式**：`<type>(<scope>): <description>`
    *   *示例*：`feat(plan-1.1): implement data loader for GenImage`
    *   *示例*：`docs(log): update operation log for section 2.2`
*   **每小节独立提交**：一个 Plan 小节对应一个 commit，严禁合并提交。
*   **提交前检查**：确保所有模块通过导入验证和基础功能测试后，方可提交。

---

## 六、 关键文档结构与编写规范（Documentation & Specifications）

### 6.1 README 编写规范
项目根目录下的 **`README.md`** 为整个工程的唯一入口文档（Entry Point），必须保持内容完整、结构清晰，并随着项目迭代持续维护。README 至少应包含以下内容：

*   **项目概览（Project Overview）**：介绍项目背景、核心功能、设计目标及适用场景。
*   **系统架构（System Architecture）**：使用 Mermaid 绘制整体架构图、算法流程图或模块调用关系图，说明系统各组成部分之间的协作关系。
*   **快速开始（Getting Started）**：包含环境安装方式、依赖配置、AutoDL 镜像要求、数据目录挂载方式及项目启动流程。
*   **工程目录（Directory Structure）**：展示项目目录结构，并对主要模块、核心文件及功能职责进行说明。
*   **接口规范（API & Schemas）**：描述核心 JSON 数据格式、接口参数说明、业务触发条件及相关 Schema 定义。
*   **维护说明（Maintenance Protocol）**：记录操作日志位置、版本更新规范、测试运行方式及项目维护流程。

---