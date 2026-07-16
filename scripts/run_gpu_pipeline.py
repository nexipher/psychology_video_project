#!/usr/bin/env python3
"""A1 GPU 全流程脚本 — 使用真实 YOLOv8-Pose 推理。

Pipeline:
  FileVideoStream → YOLOv8-Pose(GPU) → MultiObjectTracker
  → VideoFeatureExtractor → DailyAggregator → §6.1 JSON 输出

⚠️ 需要 GPU + 用户审批后才能运行。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_analysis.video_stream import FileVideoStream
from src.video_analysis.pose_estimator import PoseEstimator, check_gpu_available
from src.video_analysis.tracker import MultiObjectTracker
from src.video_analysis.feature_extractor import VideoFeatureExtractor
from src.video_analysis.aggregator import DailyAggregator
from src.utils.schema_validator import get_validator


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/root/autodl-tmp/psychology_video_project/dataset/Videos_mp4/P12T05C05.mp4"
    )
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_daily_metrics.json"

    # ---- GPU 检查 ----
    if not check_gpu_available():
        print("❌ GPU 不可用！请确认 RTX 4090 已开启。")
        print(f"   torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"   CUDA_VISIBLE_DEVICES = {torch.cuda.device_count()}")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU 就绪: {gpu_name} ({vram_mb:.1f} GB)")

    # ---- 初始化 ----
    print(f"视频文件: {video_path}")
    print(f"{'='*60}")

    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)

    # Real 模式 PoseEstimator
    estimator = PoseEstimator(
        mode="real",
        model_path="yolov8n-pose.pt",
        conf_threshold=0.25,
        iou_threshold=0.7,
        image_size=640,
    )

    tracker = MultiObjectTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        min_hits=3,
        max_lost=30,
    )

    extractor = VideoFeatureExtractor(
        window_size_sec=300.0,     # 5 min window
        window_stride_sec=60.0,    # 1 min output
        fps=15.0,
    )
    aggregator = DailyAggregator(fps=15.0)

    print(f"源帧率: {stream.native_fps:.1f} → 目标帧率: {stream.target_fps:.1f}")
    print(f"预计处理帧数: {stream.get_frame_count()}")
    print(f"模型: yolov8n-pose.pt | 置信度阈值: 0.25")
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

            # Step 1: GPU 姿态估计 (真实推理)
            pose_result = estimator.estimate(rgb_frame)
            n_detected = pose_result["keypoints"].shape[0]
            total_persons_detected += n_detected

            # Step 2: 多目标跟踪
            if n_detected > 0:
                active = tracker.update(
                    pose_result["keypoints"],
                    pose_result["bboxes"],
                    pose_result["confidences"],
                    frame_idx,
                )
            else:
                active = tracker.update(
                    np.empty((0, 17, 3), dtype=np.float32),
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    frame_idx,
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

            # Step 4: 特征提取
            window_result = extractor.process_frame(
                agg_kps, agg_boxes, ids_list, ts, frame_idx,
            )
            if window_result:
                aggregator.add_window(window_result)
                window_outputs += 1

            # 进度
            if total_frames % 500 == 0:
                elapsed = time.time() - t_start
                fps_proc = total_frames / elapsed if elapsed > 0 else 0
                gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                print(
                    f"  [{total_frames}] {elapsed:.0f}s | {fps_proc:.1f} fps | "
                    f"检测: {n_detected}人 | 活跃track: {tracker.track_count} | "
                    f"GPU: {gpu_mem:.0f}MB | 窗口: {window_outputs}"
                )

    finally:
        stream.close()
        estimator.unload_model()

    t_total = time.time() - t_start

    print(f"{'='*60}")
    print(f"处理完成: {total_frames} 帧, {t_total:.1f}s, {total_frames/t_total:.1f} fps")
    print(f"窗口输出: {window_outputs}")
    print(f"平均检测人数: {total_persons_detected / max(total_frames, 1):.2f}")

    # ---- 日级聚合 ----
    daily = extractor.get_daily_summary(user_id="P12T05C05", date="2026-07-15")

    print(f"\n{'='*60}")
    print("日级指标 (§6.1 JSON Schema):")
    print(json.dumps(daily, indent=2, ensure_ascii=False))

    # ---- Schema 校验 ----
    validator = get_validator()
    ok, errors = validator.validate_daily_metrics(daily)
    print(f"\nSchema 校验: {'✅ 通过' if ok else '❌ 失败'}")
    if errors:
        for e in errors:
            print(f"  - {e}")

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
    print(f"\n运行统计: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
