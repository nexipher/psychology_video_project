#!/usr/bin/env python3
"""A1+A2+A3 流式管线 — YOLO+Qwen 共驻显存，A2 实时触发 A3。

Pipeline:
  FileVideoStream → YOLOv8-Pose(GPU) + Qwen2.5-VL(GPU，共驻)
  → MultiObjectTracker → VideoFeatureExtractor (A1)
  → SpecialBehaviorDetector (A2) → A3EventDispatcher → MLLMVerifier (A3)
  → 最终 JSON

与 run_a1_a3_pipeline.py（batch 模式）的区别:
  - YOLO 和 Qwen 同时加载，全程共驻显存
  - A2 检测器触发后实时调用 A3（而非视频结束后批量扫描）
  - 冷却期机制：同 event_type 60-120s 内仅计数不调 MLLM
  - start_sec/end_sec 为视频内实际时间戳
  - 同一 event_type 可在冷却期结束后多次触发

⚠️ 需要 GPU (RTX 4090 24GB) + 用户审批后才能运行。
"""

from __future__ import annotations

import os
import sys

# Fix AutoDL OMP_NUM_THREADS=0 bug (causes libgomp warning)
if os.environ.get("OMP_NUM_THREADS") == "0":
    del os.environ["OMP_NUM_THREADS"]

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_analysis.video_stream import FileVideoStream
from src.video_analysis.pose_estimator import PoseEstimator, check_gpu_available
from src.video_analysis.tracker import MultiObjectTracker
from src.video_analysis.feature_extractor import VideoFeatureExtractor
from src.video_analysis.aggregator import DailyAggregator
from src.video_analysis.special_behavior import SpecialBehaviorDetector
from src.video_analysis.mllm_verifier import MLLMVerifier
from src.video_analysis.event_dispatcher import A3EventDispatcher
from src.utils.schema_validator import get_validator


def _compute_centroid(keypoints: np.ndarray) -> Optional[tuple]:
    N = keypoints.shape[0]
    if N == 0:
        return None
    centroids = []
    for i in range(N):
        if keypoints[i, 11, 2] > 0.1 and keypoints[i, 12, 2] > 0.1:
            cx = (keypoints[i, 11, 0] + keypoints[i, 12, 0]) / 2.0
            cy = (keypoints[i, 11, 1] + keypoints[i, 12, 1]) / 2.0
            centroids.append((cx, cy))
    if centroids:
        return (float(np.mean([c[0] for c in centroids])),
                float(np.mean([c[1] for c in centroids])))
    return None


def _compute_pose_height(kps: np.ndarray) -> float:
    pairs = [
        (5, 6, 15, 16, 0.1),   # shoulders→ankles
        (11, 12, 15, 16, 0.1),  # hips→ankles
        (11, 12, 13, 14, 0.1),  # hips→knees
        (0, 0, 11, 12, 0.1),    # nose→hips
    ]
    for u1, u2, l1, l2, min_c in pairs:
        upper_y = kps[u1, 1] if u1 == u2 else (kps[u1, 1] + kps[u2, 1]) / 2.0
        upper_c = kps[u1, 2] if u1 == u2 else (kps[u1, 2] + kps[u2, 2]) / 2.0
        lower_y = (kps[l1, 1] + kps[l2, 1]) / 2.0
        lower_c = (kps[l1, 2] + kps[l2, 2]) / 2.0
        if upper_c > min_c and lower_c > min_c:
            h = abs(lower_y - upper_y)
            if h > 10:
                return float(h)
    return 0.0


def _compute_is_sedentary(keypoints: np.ndarray, still_streak: int = 0) -> bool:
    N = keypoints.shape[0]
    if N == 0:
        return True
    if still_streak >= 450:
        return True
    heights = []
    for i in range(N):
        h = _compute_pose_height(keypoints[i])
        if h > 0:
            heights.append(h)
    avg_h = float(np.mean(heights)) if heights else 0.0
    return avg_h <= 72


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else (
        str(Path(__file__).resolve().parent.parent / "dataset" / "Videos_mp4" / "P14T14C06.mp4")
    )
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not check_gpu_available():
        print("❌ GPU 不可用！请先开启 GPU 实例。")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"视频: {video_path}")
    print(f"模式: 流式 (YOLO+Qwen 共驻显存, A2 实时触发 A3)")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════
    # 启动 — 加载两个模型
    # ═══════════════════════════════════════

    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)

    # Phase 1: 加载 YOLO
    print("加载 YOLOv8-Pose...")
    estimator = PoseEstimator(mode="real", model_path="yolov8n-pose.pt",
                              conf_threshold=0.25, iou_threshold=0.7, image_size=640)
    t0 = time.time()
    estimator.load_model(approve_gpu=True)
    yolo_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"YOLO 加载完成 ({time.time() - t0:.1f}s, 显存 {yolo_mem:.0f} MB)")

    # Phase 2: 加载 Qwen（共驻）
    print("加载 Qwen2.5-VL-7B...")
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "models"
                     / "qwen--Qwen2.5-VL-7B-Instruct" / "snapshots" / "master")
    verifier = MLLMVerifier(mode="real", model_name=model_path, num_frames=16)
    t0 = time.time()
    verifier.load_model(approve_gpu=True)
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Qwen 加载完成 ({time.time() - t0:.1f}s, 总计显存 {gpu_mem:.1f} GB)")

    # ═══════════════════════════════════════
    # 初始化管线组件
    # ═══════════════════════════════════════

    tracker = MultiObjectTracker(track_high_thresh=0.5, track_low_thresh=0.1,
                                  min_hits=3, max_lost=30)
    extractor = VideoFeatureExtractor(window_size_sec=300.0, window_stride_sec=60.0, fps=15.0)
    aggregator = DailyAggregator(fps=15.0)
    behavior = SpecialBehaviorDetector(fps=15.0)

    # ★ 流式 A3 调度器
    dispatcher = A3EventDispatcher(verifier, video_path)
    behavior.set_trigger_callback(dispatcher.on_trigger)

    print(f"\n管线就绪:")
    print(f"  A3 冷却期: repetitive=60s, social=120s, inactivity=120s")
    print(f"  YOLO + Qwen 共驻显存: {gpu_mem:.1f} GB / {vram_gb:.1f} GB")

    # 多人假阳性过滤
    multi_person_min_frames = 15
    multi_person_min_bbox_size = 40
    multi_person_streak = 0
    still_streak = 0

    print(f"源帧率: {stream.native_fps:.1f} → 目标: {stream.target_fps:.1f} fps")
    print(f"预计帧数: {stream.get_frame_count()}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════
    # 主循环 — 逐帧 A1→A2→(实时A3)
    # ═══════════════════════════════════════

    t_start = time.time()
    total_frames = 0
    window_outputs = 0
    ts = 0.0

    for frame_idx, (rgb_frame, _ts) in enumerate(stream):
        total_frames += 1
        ts = _ts

        # Step 1: YOLO 姿态估计
        pose_result = estimator.estimate(rgb_frame)

        # 过滤小框假阳性
        raw_kps = pose_result["keypoints"]
        raw_bboxes = pose_result["bboxes"]
        raw_confs = pose_result["confidences"]
        valid_mask = np.ones(len(raw_confs), dtype=bool)
        for i in range(len(raw_confs)):
            bw = raw_bboxes[i, 2] - raw_bboxes[i, 0]
            bh = raw_bboxes[i, 3] - raw_bboxes[i, 1]
            if bw < multi_person_min_bbox_size or bh < multi_person_min_bbox_size:
                valid_mask[i] = False
        filtered_kps = raw_kps[valid_mask]
        filtered_bboxes = raw_bboxes[valid_mask]
        filtered_confs = raw_confs[valid_mask]
        n_detected = len(filtered_confs)

        # 多人连续帧过滤
        if n_detected >= 2:
            multi_person_streak += 1
            if multi_person_streak < multi_person_min_frames:
                best_idx = int(np.argmax(filtered_confs))
                filtered_kps = filtered_kps[best_idx:best_idx+1]
                filtered_bboxes = filtered_bboxes[best_idx:best_idx+1]
                filtered_confs = filtered_confs[best_idx:best_idx+1]
                n_detected = 1
        else:
            multi_person_streak = 0

        # Step 2: ByteTrack 跟踪
        if n_detected > 0:
            active = tracker.update(filtered_kps, filtered_bboxes, filtered_confs, frame_idx)
        else:
            active = tracker.update(
                np.empty((0, 17, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32), frame_idx,
            )

        # Step 3: A1 特征提取
        if active:
            kps_list = [t.keypoints for t in active]
            bbox_list = [t.bbox for t in active]
            ids_list = [t.track_id for t in active]
            agg_kps = np.stack(kps_list, axis=0)
            agg_boxes = np.stack(bbox_list, axis=0)
        else:
            agg_kps = np.empty((0, 17, 3), dtype=np.float32)
            agg_boxes = np.empty((0, 4), dtype=np.float32)
            ids_list = []

        window_result = extractor.process_frame(agg_kps, agg_boxes, ids_list, ts, frame_idx)
        if window_result:
            aggregator.add_window(window_result)
            window_outputs += 1

        # Step 4: A2 专项检测（触发时自动回调 A3EventDispatcher）
        centroid = _compute_centroid(agg_kps)
        still_this_frame = False
        if centroid is not None and hasattr(behavior, '_last_centroid') and behavior._last_centroid is not None:
            disp = np.linalg.norm(np.array(centroid) - np.array(behavior._last_centroid))
            still_this_frame = disp < 5.0
        if centroid is not None:
            behavior._last_centroid = centroid
        still_streak = still_streak + 1 if still_this_frame else 0
        is_sed = _compute_is_sedentary(agg_kps, still_streak=still_streak)

        # A2.update() → 内部触发 → dispatcher.on_trigger() → 冷却期判断 → MLLM
        behavior.update(
            centroid_x=centroid[0] if centroid else None,
            centroid_y=centroid[1] if centroid else None,
            is_sedentary=is_sed,
            timestamp=ts,
            keypoints=agg_kps if agg_kps.shape[0] > 0 else None,
            bboxes=agg_boxes if agg_boxes.shape[0] > 0 else None,
            track_ids=ids_list if ids_list else None,
        )

        # 进度
        if total_frames % 500 == 0:
            elapsed = time.time() - t_start
            fps_proc = total_frames / elapsed if elapsed > 0 else 0
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024**2
            cd_status = dispatcher.get_cooldown_status()
            cd_str = " | ".join(
                f"{et}:{'cool' if cd_status[et]['cooldown_until'] else 'idle'}"
                for et in sorted(cd_status)
            )
            print(f"  [{total_frames}] {elapsed:.0f}s | {fps_proc:.1f} fps | "
                  f"检测:{n_detected}人 | GPU:{gpu_mem_mb:.0f}MB | "
                  f"窗口:{window_outputs} | MLLM:{dispatcher.total_mllm_calls}")
            print(f"    冷却期: {cd_str}")

    # ═══════════════════════════════════════
    # 收尾
    # ═══════════════════════════════════════

    stream.close()
    t_total = time.time() - t_start

    # A2 收尾
    behavior.flush(ts)
    a2_summary = behavior.get_daily_summary(datetime.now().strftime("%Y-%m-%d"))

    # A3 收集所有 MLLM 结果
    mllm_results = dispatcher.flush()

    # 卸载模型
    estimator.unload_model()
    verifier.unload_model()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"处理完成: {total_frames} 帧, {t_total:.1f}s, {total_frames/t_total:.1f} fps")
    print(f"窗口输出: {window_outputs}")
    print(f"A3 触发总数: {dispatcher.total_triggers} | MLLM 实际调用: {dispatcher.total_mllm_calls}")

    # ═══════════════════════════════════════
    # 日级聚合 (A1)
    # ═══════════════════════════════════════

    daily = extractor.get_daily_summary(user_id=video_name, date=datetime.now().strftime("%Y-%m-%d"))
    daily["daily_metrics"]["repetitive_path_count"] = a2_summary.get("daily_repetitive_path_count", 0)

    validator = get_validator()
    ok, errors = validator.validate_daily_metrics(daily)
    print(f"§6.1 Schema 校验: {'✅ 通过' if ok else '❌ 失败'}")

    daily["a2_special_behavior"] = {
        "daily_repetitive_path_count": a2_summary.get("daily_repetitive_path_count", 0),
        "daily_hotspot_action_count": a2_summary.get("daily_hotspot_action_count", 0),
        "daily_prolonged_inactive_count": a2_summary.get("daily_prolonged_inactive_count", 0),
        "max_inactive_stretch_sec": a2_summary.get("max_inactive_stretch_sec", 0.0),
        "daily_avg_social_intensity": a2_summary.get("daily_avg_social_intensity", 0.0),
    }
    if "circadian" in a2_summary:
        daily["a2_special_behavior"]["circadian"] = a2_summary["circadian"]

    # ═══════════════════════════════════════
    # 合并输出
    # ═══════════════════════════════════════

    final = {
        "user_id": video_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pipeline_mode": "streaming",
        "daily_metrics": daily.get("daily_metrics", {}),
        "a2_special_behavior": daily.get("a2_special_behavior", {}),
        "a3_mllm_verification": mllm_results,
    }

    results_dir = Path(__file__).resolve().parent.parent / "results" / "A1A3"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{video_name}_streaming_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # 终端预览
    print(f"\n{'='*60}")
    print("A1 日级指标:")
    dm = final["daily_metrics"]
    print(f"  active_minutes={dm.get('active_minutes', 0):.1f}  "
          f"sedentary_ratio={dm.get('sedentary_ratio', 0):.2f}  "
          f"coverage={dm.get('coverage_minutes', 0):.1f}min")
    print(f"  room_transitions={dm.get('room_transition_count', 0)}  "
          f"night_activity={dm.get('night_activity_count', 0)}  "
          f"multi_person={dm.get('multi_person_duration', 0):.1f}min")

    print(f"\nA2 检测:")
    print(f"  久坐事件={a2_summary.get('daily_prolonged_inactive_count', 0)}  "
          f"徘徊={a2_summary.get('daily_repetitive_path_count', 0)}  "
          f"热点={a2_summary.get('daily_hotspot_action_count', 0)}  "
          f"社交={a2_summary.get('daily_avg_social_intensity', 0):.3f}")

    print(f"\nA3 MLLM 复核: {len(mllm_results)} 个事件 "
          f"(触发{dispatcher.total_triggers}次, 实际调用{dispatcher.total_mllm_calls}次)")
    for r in mllm_results:
        status = "✅" if r.get("evidence_sufficient") else "⚠️"
        print(f"  {status} [{r.get('event_type')}] "
              f"t={r.get('start_sec', 0):.0f}s-{r.get('end_sec', 0):.0f}s | "
              f"occurrences={r.get('num_of_occurrences', '?')} | "
              f"{r.get('activity_state', '?')} | "
              f"{r.get('repetition_type', r.get('social_context', '?'))}")


if __name__ == "__main__":
    main()
