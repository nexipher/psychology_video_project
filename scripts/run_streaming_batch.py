#!/usr/bin/env python3
"""A1+A2+A3 流式批量跑批 — 所有视频逐一跑流式管线，输出到 results/A1A3/。"""

from __future__ import annotations

import os
import sys

if os.environ.get("OMP_NUM_THREADS") == "0":
    del os.environ["OMP_NUM_THREADS"]

import json
import time
from datetime import datetime
from pathlib import Path

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


def process_video(video_path: str, verifier: MLLMVerifier, video_index: int, total: int) -> dict:
    video_name = Path(video_path).stem
    print(f"\n{'='*70}")
    print(f"[{video_index}/{total}] {video_name}")
    print(f"{'='*70}")

    multi_person_min_frames = 15
    multi_person_min_bbox_size = 40
    multi_person_streak = 0
    still_streak = 0

    # 每个视频独立的组件（避免状态污染）
    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)
    estimator = PoseEstimator(mode="real", model_path="yolov8n-pose.pt",
                              conf_threshold=0.25, iou_threshold=0.7, image_size=640)
    tracker = MultiObjectTracker(track_high_thresh=0.5, track_low_thresh=0.1,
                                  min_hits=3, max_lost=30)
    extractor = VideoFeatureExtractor(window_size_sec=300.0, window_stride_sec=60.0, fps=15.0)
    aggregator = DailyAggregator(fps=15.0)
    behavior = SpecialBehaviorDetector(fps=15.0)

    dispatcher = A3EventDispatcher(verifier, video_path)
    behavior.set_trigger_callback(dispatcher.on_trigger)

    estimator.load_model(approve_gpu=True)

    t_start = time.time()
    total_frames = 0
    window_outputs = 0
    ts = 0.0

    for frame_idx, (rgb_frame, _ts) in enumerate(stream):
        total_frames += 1
        ts = _ts

        pose_result = estimator.estimate(rgb_frame)

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

        if n_detected > 0:
            active = tracker.update(filtered_kps, filtered_bboxes, filtered_confs, frame_idx)
        else:
            active = tracker.update(
                np.empty((0, 17, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32), frame_idx,
            )

        if active:
            agg_kps = np.stack([t.keypoints for t in active], axis=0)
            agg_boxes = np.stack([t.bbox for t in active], axis=0)
            ids_list = [t.track_id for t in active]
        else:
            agg_kps = np.empty((0, 17, 3), dtype=np.float32)
            agg_boxes = np.empty((0, 4), dtype=np.float32)
            ids_list = []

        window_result = extractor.process_frame(agg_kps, agg_boxes, ids_list, ts, frame_idx)
        if window_result:
            aggregator.add_window(window_result)
            window_outputs += 1

        # 质心 + 静止判定
        centroid = None
        if agg_kps.shape[0] > 0:
            valid_hips = (agg_kps[:, 11, 2] > 0.1) & (agg_kps[:, 12, 2] > 0.1)
            if valid_hips.any():
                cxs = (agg_kps[valid_hips, 11, 0] + agg_kps[valid_hips, 12, 0]) / 2.0
                cys = (agg_kps[valid_hips, 11, 1] + agg_kps[valid_hips, 12, 1]) / 2.0
                centroid = (float(np.mean(cxs)), float(np.mean(cys)))

        still_this_frame = False
        if centroid and hasattr(behavior, '_last_centroid') and behavior._last_centroid:
            disp = np.linalg.norm(np.array(centroid) - np.array(behavior._last_centroid))
            still_this_frame = disp < 5.0
        if centroid:
            behavior._last_centroid = centroid
        still_streak = still_streak + 1 if still_this_frame else 0

        # 坐姿判定
        is_sed = _compute_is_sedentary(agg_kps, still_streak)

        behavior.update(
            centroid_x=centroid[0] if centroid else None,
            centroid_y=centroid[1] if centroid else None,
            is_sedentary=is_sed, timestamp=ts,
            keypoints=agg_kps if agg_kps.shape[0] > 0 else None,
            bboxes=agg_boxes if agg_boxes.shape[0] > 0 else None,
            track_ids=ids_list if ids_list else None,
        )

        if total_frames % 1000 == 0:
            elapsed = time.time() - t_start
            fps = total_frames / elapsed if elapsed > 0 else 0
            print(f"  [{total_frames}] {elapsed:.0f}s {fps:.0f}fps | "
                  f"窗口:{window_outputs} MLLM:{dispatcher.total_mllm_calls}")

    stream.close()
    estimator.unload_model()

    total_triggers = dispatcher.total_triggers
    total_mllm = dispatcher.total_mllm_calls
    mllm_results = dispatcher.flush()
    behavior.flush(ts)
    a2_summary = behavior.get_daily_summary(datetime.now().strftime("%Y-%m-%d"))

    daily = extractor.get_daily_summary(user_id=video_name, date=datetime.now().strftime("%Y-%m-%d"))
    daily["daily_metrics"]["repetitive_path_count"] = a2_summary.get("daily_repetitive_path_count", 0)

    a2_sb = {
        "daily_repetitive_path_count": a2_summary.get("daily_repetitive_path_count", 0),
        "daily_hotspot_action_count": a2_summary.get("daily_hotspot_action_count", 0),
        "daily_prolonged_inactive_count": a2_summary.get("daily_prolonged_inactive_count", 0),
        "max_inactive_stretch_sec": a2_summary.get("max_inactive_stretch_sec", 0.0),
        "daily_avg_social_intensity": a2_summary.get("daily_avg_social_intensity", 0.0),
    }
    if "circadian" in a2_summary:
        a2_sb["circadian"] = a2_summary["circadian"]

    t_elapsed = time.time() - t_start
    dm = daily["daily_metrics"]
    print(f"  完成: {total_frames}帧 {t_elapsed:.0f}s {total_frames/t_elapsed:.0f}fps | "
          f"active={dm['active_minutes']:.1f}min sed_ratio={dm['sedentary_ratio']:.2f} "
          f"coverage={dm['coverage_minutes']:.1f}min")
    print(f"  A3: {total_triggers}触发 → {total_mllm}次MLLM → {len(mllm_results)}结果")

    return {
        "video_name": video_name,
        "total_frames": total_frames,
        "elapsed_sec": round(t_elapsed, 1),
        "fps": round(total_frames/t_elapsed, 1),
        "total_triggers": total_triggers,
        "total_mllm_calls": total_mllm,
        "mllm_results": len(mllm_results),
        "data": {
            "user_id": video_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pipeline_mode": "streaming",
            "daily_metrics": dm,
            "a2_special_behavior": a2_sb,
            "a3_mllm_verification": mllm_results,
        },
    }


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
    return (float(np.mean(heights)) if heights else 0.0) <= 72


def _compute_pose_height(kps: np.ndarray) -> float:
    pairs = [
        (5, 6, 15, 16, 0.1), (11, 12, 15, 16, 0.1),
        (11, 12, 13, 14, 0.1), (0, 0, 11, 12, 0.1),
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


def main():
    video_dir = Path(__file__).resolve().parent.parent / "dataset" / "Videos_mp4"
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print("❌ 未找到视频文件")
        sys.exit(1)

    if not check_gpu_available():
        print("❌ GPU 不可用！")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")
    print(f"视频数: {len(videos)}")
    print(f"{'='*70}")

    # 加载 Qwen（全程共驻，每个视频只需加载 YOLO）
    print("加载 Qwen2.5-VL-7B...")
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "models"
                     / "qwen--Qwen2.5-VL-7B-Instruct" / "snapshots" / "master")
    verifier = MLLMVerifier(mode="real", model_name=model_path, num_frames=16)
    verifier.load_model(approve_gpu=True)
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Qwen 就绪 ({gpu_mem:.1f} GB)")

    summary = []
    t_batch_start = time.time()

    for i, vp in enumerate(videos, 1):
        result = process_video(str(vp), verifier, i, len(videos))
        summary.append(result)

    verifier.unload_model()
    torch.cuda.empty_cache()
    t_batch = time.time() - t_batch_start

    # 保存汇总
    batch_summary = {
        "batch_timestamp": timestamp,
        "pipeline_mode": "streaming",
        "total_videos": len(videos),
        "total_elapsed_sec": round(t_batch, 1),
        "gpu": gpu_name,
        "vram_gb": round(vram, 1),
        "videos": summary,
    }

    results_dir = Path(__file__).resolve().parent.parent / "results" / "A1A3"
    results_dir.mkdir(parents=True, exist_ok=True)
    batch_path = results_dir / f"batch_streaming_{timestamp}.json"
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    # 各视频单独存储
    for s in summary:
        vn = s["video_name"]
        out_path = results_dir / f"{vn}_streaming_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(s["data"], f, indent=2, ensure_ascii=False)

    # 终端汇总表
    print(f"\n{'='*70}")
    print(f"全量跑批完成: {len(videos)} 视频, 总耗时 {t_batch:.0f}s")
    print(f"汇总: {batch_path}")
    print(f"{'='*70}")
    print(f"{'视频':<14} {'帧数':>6} {'耗时':>6} {'fps':>6} {'active':>7} {'sed_r':>6} {'cov':>6} {'触发':>5} {'MLLM':>5}")
    print(f"{'─'*70}")
    for s in summary:
        dm = s["data"]["daily_metrics"]
        print(f"{s['video_name']:<14} {s['total_frames']:>6} {s['elapsed_sec']:>6.0f}s "
              f"{s['fps']:>5.0f} {dm['active_minutes']:>6.1f} {dm['sedentary_ratio']:>5.2f} "
              f"{dm['coverage_minutes']:>5.1f} {s['total_triggers']:>5} {s['total_mllm_calls']:>5}")


if __name__ == "__main__":
    main()
