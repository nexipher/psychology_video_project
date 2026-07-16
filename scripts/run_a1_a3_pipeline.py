#!/usr/bin/env python3
"""A1+A2+A3 全流程脚本 — YOLOv8-Pose → 专项行为检测 → Qwen2.5-VL 复核。

Pipeline:
  FileVideoStream → YOLOv8-Pose(GPU) → MultiObjectTracker
  → VideoFeatureExtractor (A1) → DailyAggregator → §6.1 JSON
  → SpecialBehaviorDetector (A2) → 徘徊/重复/久坐/节律/社交
  → [卸载YOLO，加载Qwen2.5-VL]
  → MLLMVerifier (A3) → §6.2 JSON 复核每个A2触发事件
  → 合并输出 §§6.1+6.2

⚠️ 需要 GPU (RTX 4090 24GB) + 用户审批后才能运行。
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
from src.video_analysis.mllm_verifier import MLLMVerifier, generate_mllm_triggers
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


# ═══════════════════════════════════════════════════════════
# Phase 1: A1 + A2 (YOLOv8-Pose)
# ═══════════════════════════════════════════════════════════

def run_a1_a2(video_path: str) -> dict:
    """运行 A1+A2 管线，返回日级汇总。"""
    video_name = Path(video_path).stem

    stream = FileVideoStream(video_path, target_fps=15.0, target_width=640, target_height=480)
    estimator = PoseEstimator(mode="real", model_path="yolov8n-pose.pt",
                              conf_threshold=0.25, iou_threshold=0.7, image_size=640)
    tracker = MultiObjectTracker(track_high_thresh=0.5, track_low_thresh=0.1,
                                  min_hits=3, max_lost=30)
    extractor = VideoFeatureExtractor(window_size_sec=300.0, window_stride_sec=60.0, fps=15.0)
    aggregator = DailyAggregator(fps=15.0)
    behavior = SpecialBehaviorDetector(fps=15.0)

    # 多人假阳性过滤
    multi_person_min_frames = 15
    multi_person_min_bbox_size = 40
    multi_person_streak = 0
    still_streak = 0

    print(f"源帧率: {stream.native_fps:.1f} → 目标: {stream.target_fps:.1f} fps")
    print(f"预计帧数: {stream.get_frame_count()} | 模型: yolov8n-pose.pt")
    print(f"{'='*60}")

    # 加载 YOLO
    print("加载 YOLOv8-Pose...")
    t_load = time.time()
    estimator.load_model(approve_gpu=True)
    print(f"YOLO 加载完成 ({time.time() - t_load:.1f}s)")

    t_start = time.time()
    total_frames = 0
    window_outputs = 0
    ts = 0.0

    try:
        for frame_idx, (rgb_frame, _ts) in enumerate(stream):
            total_frames += 1
            ts = _ts

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

            # 跟踪
            if n_detected > 0:
                active = tracker.update(filtered_kps, filtered_bboxes, filtered_confs, frame_idx)
            else:
                active = tracker.update(
                    np.empty((0, 17, 3), dtype=np.float32),
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32), frame_idx,
                )

            # A1 特征提取
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

            # A2 专项检测
            centroid = _compute_centroid(agg_kps)
            still_this_frame = False
            if centroid is not None and hasattr(behavior, '_last_centroid') and behavior._last_centroid is not None:
                disp = np.linalg.norm(np.array(centroid) - np.array(behavior._last_centroid))
                still_this_frame = disp < 5.0
            if centroid is not None:
                behavior._last_centroid = centroid
            still_streak = still_streak + 1 if still_this_frame else 0
            is_sed = _compute_is_sedentary(agg_kps, still_streak=still_streak)
            behavior.update(
                centroid_x=centroid[0] if centroid else None,
                centroid_y=centroid[1] if centroid else None,
                is_sedentary=is_sed,
                timestamp=ts, keypoints=agg_kps if agg_kps.shape[0] > 0 else None,
                bboxes=agg_boxes if agg_boxes.shape[0] > 0 else None,
                track_ids=ids_list if ids_list else None,
            )

            if total_frames % 500 == 0:
                elapsed = time.time() - t_start
                fps_proc = total_frames / elapsed if elapsed > 0 else 0
                gpu_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  [{total_frames}] {elapsed:.0f}s | {fps_proc:.1f} fps | "
                      f"检测:{n_detected}人 | GPU:{gpu_mem:.0f}MB | 窗口:{window_outputs}")

    finally:
        stream.close()
        estimator.unload_model()
        torch.cuda.empty_cache()

    t_a1a2 = time.time() - t_start
    print(f"\nA1+A2 完成: {total_frames} 帧, {t_a1a2:.1f}s, {total_frames/t_a1a2:.1f} fps")

    # A2 收尾
    behavior.flush(ts)
    a2_summary = behavior.get_daily_summary(datetime.now().strftime("%Y-%m-%d"))

    # 日级聚合
    daily = extractor.get_daily_summary(user_id=video_name, date=datetime.now().strftime("%Y-%m-%d"))
    daily["daily_metrics"]["repetitive_path_count"] = a2_summary.get("daily_repetitive_path_count", 0)

    # §6.1 校验
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

    print(f"\nA2 检测结果:")
    print(f"  久坐事件: {a2_summary.get('daily_prolonged_inactive_count', 0)}")
    print(f"  最长静止: {a2_summary.get('max_inactive_stretch_sec', 0):.0f}s")
    print(f"  徘徊次数: {a2_summary.get('daily_repetitive_path_count', 0)}")
    print(f"  热点动作: {a2_summary.get('daily_hotspot_action_count', 0)}")
    print(f"  社交强度: {a2_summary.get('daily_avg_social_intensity', 0):.3f}")

    return daily


# ═══════════════════════════════════════════════════════════
# Phase 2: A3 (Qwen2.5-VL MLLM 复核)
# ═══════════════════════════════════════════════════════════

def run_a3(video_path: str, a2_daily: dict) -> list[dict]:
    """基于 A2 检测结果，调用 Qwen2.5-VL 进行事件复核。"""
    a2_summary = a2_daily.get("a2_special_behavior", {})

    triggers = generate_mllm_triggers(a2_summary)
    if not triggers:
        print("\n⚠️ A2 未触发任何异常事件，跳过 A3 MLLM 复核")
        return []

    print(f"\nA2 触发了 {len(triggers)} 个事件，开始 A3 MLLM 复核...")
    for t in triggers:
        print(f"  [{t['priority']}] {t['event_type']}: {t['reason']}")

    # 加载 Qwen2.5-VL
    print(f"\n{'='*60}")
    print("加载 Qwen2.5-VL-7B...")
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "models" / "qwen--Qwen2.5-VL-7B-Instruct" / "snapshots" / "master")
    verifier = MLLMVerifier(mode="real", model_name=model_path, num_frames=16)
    t_load = time.time()
    verifier.load_model(approve_gpu=True)
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Qwen2.5-VL 加载完成 ({time.time() - t_load:.1f}s, 显存 {gpu_mem:.1f} GB)")

    # 逐个复核
    results = []
    for i, trigger in enumerate(triggers):
        print(f"\n--- A3 复核 [{i+1}/{len(triggers)}] {trigger['event_type']} ---")
        t0 = time.time()
        try:
            result = verifier.verify(
                video_path=video_path,
                event_type=trigger["event_type"],
                trigger_ts=trigger["trigger_ts"],
            )
            elapsed = time.time() - t0
            print(f"  完成 ({elapsed:.1f}s)")
            print(f"  evidence: {result.get('observable_evidence', 'N/A')[:100]}...")
            print(f"  activity_state: {result.get('activity_state')}")
            print(f"  social_context: {result.get('social_context')}")
            print(f"  repetition_type: {result.get('repetition_type')}")
            print(f"  evidence_sufficient: {result.get('evidence_sufficient')}")
            results.append(result)
        except Exception as exc:
            print(f"  ❌ 复核失败: {exc}")
            results.append({
                "event_type": trigger["event_type"],
                "error": str(exc),
                "evidence_sufficient": False,
            })

    verifier.unload_model()
    torch.cuda.empty_cache()
    print(f"\nA3 复核完成: {len(results)} 个事件")
    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

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
    print(f"{'='*60}\n")

    # Phase 1: A1 + A2
    a2_daily = run_a1_a2(video_path)

    # Phase 2: A3
    mllm_results = run_a3(video_path, a2_daily)

    # ═══════════════════════════════════════
    # 合并输出
    # ═══════════════════════════════════════
    final = {
        "user_id": video_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "daily_metrics": a2_daily.get("daily_metrics", {}),
        "a2_special_behavior": a2_daily.get("a2_special_behavior", {}),
        "a3_mllm_verification": mllm_results,
    }

    # 保存
    results_dir = Path(__file__).resolve().parent.parent / "results" / "A1A3"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{video_name}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"结果已保存: {output_path}")

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
    print(f"\nA3 MLLM 复核: {len(mllm_results)} 个事件")
    for r in mllm_results:
        status = "✅ 证据充分" if r.get("evidence_sufficient") else "⚠️ 证据不足"
        print(f"  [{r.get('event_type')}] {r.get('activity_state', '?')} | "
              f"{r.get('social_context', '?')} | {status}")


if __name__ == "__main__":
    main()
