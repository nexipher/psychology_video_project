#!/usr/bin/env python3
"""A1 CPU 全流程测试脚本。

使用 Mock PoseEstimator 验证完整管线：
  FileVideoStream → PoseEstimator(Mock) → MultiObjectTracker
  → VideoFeatureExtractor → DailyAggregator → §6.1 JSON 输出
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_analysis.video_stream import FileVideoStream
from src.video_analysis.pose_estimator import PoseEstimator
from src.video_analysis.tracker import MultiObjectTracker
from src.video_analysis.feature_extractor import VideoFeatureExtractor
from src.video_analysis.aggregator import DailyAggregator
from src.utils.schema_validator import get_validator


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/root/autodl-tmp/psychology_video_project/dataset/Videos_mp4/P12T05C05.mp4"
    )

    print(f"视频文件: {video_path}")
    print(f"{'='*60}")

    # ---- 初始化组件 ----
    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)
    estimator = PoseEstimator(mode="mock")
    tracker = MultiObjectTracker(min_hits=2, max_lost=10)
    extractor = VideoFeatureExtractor(
        window_size_sec=300.0,     # 5 分钟窗口
        window_stride_sec=60.0,    # 1 分钟输出
        fps=15.0,
    )
    aggregator = DailyAggregator(fps=15.0)

    print(f"源帧率: {stream.native_fps:.1f} → 目标帧率: {stream.target_fps:.1f}")
    print(f"预计处理帧数: {stream.get_frame_count()}")
    print(f"{'='*60}")

    # ---- 主循环 ----
    t_start = time.time()
    total_frames = 0
    window_outputs = 0

    for frame_idx, (rgb_frame, ts) in enumerate(stream):
        total_frames += 1

        # Step 1: 姿态估计 (Mock)
        pose_result = estimator.estimate(rgb_frame)

        # Step 2: 多目标跟踪
        active = tracker.update(
            pose_result["keypoints"],
            pose_result["bboxes"],
            pose_result["confidences"],
            frame_idx,
        )

        # Step 3: 聚合活跃 track 数据
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
        if total_frames % 1000 == 0:
            elapsed = time.time() - t_start
            fps_proc = total_frames / elapsed if elapsed > 0 else 0
            print(f"  处理 {total_frames} 帧 | {elapsed:.1f}s | {fps_proc:.0f} fps | 窗口输出: {window_outputs}")

    stream.close()
    t_total = time.time() - t_start

    print(f"{'='*60}")
    print(f"处理完成: {total_frames} 帧, {t_total:.1f}s, {total_frames/t_total:.0f} fps")
    print(f"窗口输出: {window_outputs}")

    # ---- Step 5: 日级聚合 ----
    daily = extractor.get_daily_summary(user_id="P12T05C05", date="2026-07-15")
    # 同时用 aggregator 聚合
    agg_daily = aggregator.aggregate(user_id="P12T05C05", date="2026-07-15")

    print(f"\n{'='*60}")
    print("日级指标 (VideoFeatureExtractor):")
    print(json.dumps(daily, indent=2, ensure_ascii=False))

    print(f"\n日级指标 (DailyAggregator):")
    print(json.dumps(agg_daily, indent=2, ensure_ascii=False))

    # ---- Schema 校验 ----
    validator = get_validator()
    ok, errors = validator.validate_daily_metrics(daily)
    print(f"\nSchema 校验: {'✅ 通过' if ok else '❌ 失败'}")
    if errors:
        for e in errors:
            print(f"  - {e}")

    print(f"\n总耗时: {t_total:.1f}s")
    print("CPU 全流程测试完成！")


if __name__ == "__main__":
    main()
