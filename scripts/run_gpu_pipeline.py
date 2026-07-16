#!/usr/bin/env python3
"""A1+A2 GPU 全流程脚本 — YOLOv8-Pose 推理 + 专项行为检测。

Pipeline:
  FileVideoStream → YOLOv8-Pose(GPU) → MultiObjectTracker
  → VideoFeatureExtractor → DailyAggregator → §6.1 JSON
  → SpecialBehaviorDetector (A2) → 徘徊/重复/久坐/节律/社交

⚠️ 需要 GPU + 用户审批后才能运行。
"""

from __future__ import annotations

import json
import sys
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
from src.utils.schema_validator import get_validator


def _compute_centroid(keypoints: np.ndarray) -> Optional[tuple]:
    """从 (N,17,3) 关键点计算平均质心（髋部中点）。"""
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
        cx = float(np.mean([c[0] for c in centroids]))
        cy = float(np.mean([c[1] for c in centroids]))
        return (cx, cy)
    return None


def _compute_pose_height(kps: np.ndarray) -> float:
    """从单人多组关键点估算姿态高度，判断站姿/坐姿。"""
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
    """判断当前帧是否静止/久坐。无人→静止。连续静止>30s→坐姿。"""
    N = keypoints.shape[0]
    if N == 0:
        return True
    # 连续静止帧数超过 30 秒 → 大概率坐着
    if still_streak >= 450:
        return True
    # 否则用姿态高度法：下身可见时可用
    heights = []
    for i in range(N):
        h = _compute_pose_height(keypoints[i])
        if h > 0:
            heights.append(h)
    avg_h = float(np.mean(heights)) if heights else 0.0
    return avg_h <= 72  # 姿态高度不足 → 坐/卧


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/root/autodl-tmp/psychology_video_project/dataset/Videos_mp4/P12T05C05.mp4"
    )
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / f"{video_name}_{timestamp}.json")

    # ---- GPU 检查 ----
    if not check_gpu_available():
        print("❌ GPU 不可用！")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU 就绪: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"视频文件: {video_path}")
    print(f"{'='*60}")

    # ---- 初始化 ----
    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)

    estimator = PoseEstimator(mode="real", model_path="yolov8n-pose.pt",
                              conf_threshold=0.25, iou_threshold=0.7, image_size=640)
    tracker = MultiObjectTracker(track_high_thresh=0.5, track_low_thresh=0.1,
                                  min_hits=3, max_lost=30)
    extractor = VideoFeatureExtractor(window_size_sec=300.0, window_stride_sec=60.0, fps=15.0)
    aggregator = DailyAggregator(fps=15.0)

    # ★ A2 专项行为检测器
    behavior = SpecialBehaviorDetector(fps=15.0)

    # ★ 多人假阳性过滤器
    multi_person_min_frames = 15      # 第二人需连续存在 ≥15 帧
    multi_person_min_bbox_size = 40   # 检测框最小边长（像素）
    multi_person_streak = 0
    still_streak = 0                  # 连续静止帧计数器

    print(f"源帧率: {stream.native_fps:.1f} → 目标帧率: {stream.target_fps:.1f}")
    print(f"预计处理帧数: {stream.get_frame_count()}")
    print(f"模型: yolov8n-pose.pt | A1 + A2 全管线")
    print(f"{'='*60}")

    # ---- 加载模型 ----
    print("正在加载 YOLOv8-Pose 模型...")
    t_load = time.time()
    estimator.load_model(approve_gpu=True)
    print(f"模型加载完成 ({time.time() - t_load:.1f}s)")

    # ---- 主循环 ----
    t_start = time.time()
    total_frames = 0
    window_outputs = 0
    total_persons_detected = 0

    try:
        for frame_idx, (rgb_frame, ts) in enumerate(stream):
            total_frames += 1

            # Step 1: GPU 姿态估计
            pose_result = estimator.estimate(rgb_frame)

            # ★ 过滤假阳性：检测框过小（背景杂物误识别）
            raw_kps = pose_result["keypoints"]     # (N, 17, 3)
            raw_bboxes = pose_result["bboxes"]      # (N, 4)
            raw_confs = pose_result["confidences"]  # (N,)
            valid_mask = np.ones(len(raw_confs), dtype=bool)
            for i in range(len(raw_confs)):
                bw = raw_bboxes[i, 2] - raw_bboxes[i, 0]
                bh = raw_bboxes[i, 3] - raw_bboxes[i, 1]
                if bw < multi_person_min_bbox_size or bh < multi_person_min_bbox_size:
                    valid_mask[i] = False
            filtered_kps = raw_kps[valid_mask]
            filtered_bboxes = raw_bboxes[valid_mask]
            filtered_confs = raw_confs[valid_mask]
            n_raw = len(raw_confs)
            n_detected = len(filtered_confs)

            # ★ 多人连续帧过滤：第二人需持续存在
            if n_detected >= 2:
                multi_person_streak += 1
                if multi_person_streak < multi_person_min_frames:
                    # 未达连续帧阈值，只保留置信度最高的一人
                    best_idx = int(np.argmax(filtered_confs))
                    filtered_kps = filtered_kps[best_idx:best_idx+1]
                    filtered_bboxes = filtered_bboxes[best_idx:best_idx+1]
                    filtered_confs = filtered_confs[best_idx:best_idx+1]
                    n_detected = 1
            else:
                multi_person_streak = 0

            total_persons_detected += n_detected

            # Step 2: 多目标跟踪（使用过滤后的检测）
            if n_detected > 0:
                active = tracker.update(
                    filtered_kps, filtered_bboxes, filtered_confs, frame_idx,
                )
            else:
                active = tracker.update(
                    np.empty((0, 17, 3), dtype=np.float32),
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32), frame_idx,
                )

            # Step 3: 聚合活跃 track
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

            # Step 4: A1 特征提取
            window_result = extractor.process_frame(agg_kps, agg_boxes, ids_list, ts, frame_idx)
            if window_result:
                aggregator.add_window(window_result)
                window_outputs += 1

            # Step 5: A2 专项行为检测
            centroid = _compute_centroid(agg_kps)
            # 追踪连续静止帧数（帧间位移 < 5px 视为静止）
            still_this_frame = False
            if centroid is not None and hasattr(behavior, '_last_centroid') and behavior._last_centroid is not None:
                disp = np.linalg.norm(np.array(centroid) - np.array(behavior._last_centroid))
                still_this_frame = disp < 5.0
            if centroid is not None:
                behavior._last_centroid = centroid
            if still_this_frame:
                still_streak += 1
            else:
                still_streak = 0
            is_sed = _compute_is_sedentary(agg_kps, still_streak=still_streak)
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
                gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                print(
                    f"  [{total_frames}] {elapsed:.0f}s | {fps_proc:.1f} fps | "
                    f"检测: {n_detected}人 | track: {tracker.track_count} | "
                    f"GPU: {gpu_mem:.0f}MB | 窗口: {window_outputs}"
                )

    finally:
        stream.close()
        estimator.unload_model()

    t_total = time.time() - t_start

    # ---- A2 收尾 ----
    behavior.flush(ts if total_frames > 0 else 0.0)
    a2_summary = behavior.get_daily_summary(datetime.now().strftime("%Y-%m-%d"))

    print(f"{'='*60}")
    print(f"处理完成: {total_frames} 帧, {t_total:.1f}s, {total_frames/t_total:.1f} fps")
    print(f"窗口输出: {window_outputs}")

    # ---- 日级聚合 (A1 + A2 合并) ----
    daily = extractor.get_daily_summary(user_id=video_name, date=datetime.now().strftime("%Y-%m-%d"))

    # 用 A2 检测器结果回填 daily_metrics
    daily["daily_metrics"]["repetitive_path_count"] = a2_summary.get("daily_repetitive_path_count", 0)

    # ---- §6.1 Schema 校验 ----
    validator = get_validator()
    ok, errors = validator.validate_daily_metrics(daily)
    print(f"\n§6.1 Schema 校验: {'✅ 通过' if ok else '❌ 失败'}")
    if errors:
        for e in errors:
            print(f"  - {e}")

    # 通过校验后，附加 A2 专项结果
    daily["a2_special_behavior"] = {
        "daily_repetitive_path_count": a2_summary.get("daily_repetitive_path_count", 0),
        "daily_hotspot_action_count": a2_summary.get("daily_hotspot_action_count", 0),
        "daily_prolonged_inactive_count": a2_summary.get("daily_prolonged_inactive_count", 0),
        "max_inactive_stretch_sec": a2_summary.get("max_inactive_stretch_sec", 0.0),
        "daily_avg_social_intensity": a2_summary.get("daily_avg_social_intensity", 0.0),
    }
    if "circadian" in a2_summary:
        daily["a2_special_behavior"]["circadian"] = a2_summary["circadian"]

    print(f"\n{'='*60}")
    print("A1 + A2 完整输出:")
    print(json.dumps(daily, indent=2, ensure_ascii=False))

    # ---- 保存 ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(daily, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")

    stats = {
        "total_frames": total_frames,
        "total_time_sec": round(t_total, 1),
        "avg_fps": round(total_frames / t_total, 1),
        "gpu": gpu_name,
        "total_persons_detected": total_persons_detected,
        "avg_persons_per_frame": round(total_persons_detected / max(total_frames, 1), 2),
    }
    print(f"运行统计: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
