#!/usr/bin/env python3
"""A1+A2 并行批量视频处理器。

每个 worker 进程独立加载 YOLO 模型，并行处理视频。
GPU 时间片共享，nano 模型仅占 ~45MB/worker，24GB 可支持大规模并行。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_gpu_pipeline.py"
RESULTS_DIR = PROJECT_ROOT / "results"


def process_one(video_path: Path, batch_dir: Path, worker_id: int) -> Dict:
    """单个 worker 进程：复制视频到本地，运行管线，清理，返回结果。"""
    video_name = video_path.stem
    worker_tmp = Path("/tmp") / f"batch_w{worker_id}"
    worker_tmp.mkdir(exist_ok=True)
    local_path = worker_tmp / video_path.name

    t_start = time.time()

    # 1. 复制
    shutil.copy2(video_path, local_path)
    t_copy = time.time() - t_start

    # 2. 运行
    t_run_start = time.time()
    proc = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT), str(local_path)],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    t_run = time.time() - t_run_start

    # 3. 清理
    local_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return {
            "video": video_name, "worker": worker_id, "status": "failed",
            "copy_sec": round(t_copy, 1), "run_sec": round(t_run, 1),
            "error": proc.stderr[-300:] if proc.stderr else "",
        }

    # 4. 移入批量子文件夹
    json_files = sorted(
        RESULTS_DIR.glob(f"{video_name}_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    result_file = ""
    summary = ""
    if json_files:
        dest = batch_dir / json_files[0].name
        json_files[0].rename(dest)
        result_file = dest.name

        # 提取摘要
        try:
            data = json.loads(dest.read_text())
            m = data.get("daily_metrics", {})
            a2 = data.get("a2_special_behavior", {})
            summary = (
                f"active={m.get('active_minutes',0):.1f}min "
                f"sed={m.get('sedentary_ratio',0):.2f} "
                f"rep_path={a2.get('daily_repetitive_path_count',0)} "
                f"inactive={a2.get('daily_prolonged_inactive_count',0)} "
                f"social={a2.get('daily_avg_social_intensity',0):.2f}"
            )
        except Exception:
            summary = ""

    return {
        "video": video_name, "worker": worker_id, "status": "ok",
        "file": result_file,
        "copy_sec": round(t_copy, 1), "run_sec": round(t_run, 1),
        "summary": summary,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="并行 worker 数 (默认 4)")
    parser.add_argument("video_dir", nargs="?", default=None)
    args = parser.parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else (
        PROJECT_ROOT / "dataset" / "Videos_mp4"
    )
    if not video_dir.exists():
        print(f"❌ 目录不存在: {video_dir}")
        sys.exit(1)

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"❌ 未找到 mp4: {video_dir}")
        sys.exit(1)

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = RESULTS_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 视频摘要
    print(f"{'='*70}")
    print(f"并行处理 {len(videos)} 个视频 | {args.workers} workers | → {batch_dir.name}")
    print(f"{'='*70}")
    total_frames = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps, frames = cap.get(5), int(cap.get(7))
        total_frames += frames
        cap.release()
        print(f"  {v.stem}: {frames}帧 {frames/fps:.0f}s ({v.stat().st_size//1024**2}MB)")
    print(f"总计: {total_frames}帧 | 预计 ~{total_frames/100/args.workers:.0f}s "
          f"({total_frames/100/args.workers/60:.1f}min) @ {args.workers} workers")
    print(f"{'='*70}")

    # 并行处理
    total_start = time.time()
    results = []

    # fork 模式兼容 CUDA
    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=args.workers) as pool:
        tasks = [
            pool.apply_async(process_one, (v, batch_dir, i % args.workers))
            for i, v in enumerate(videos)
        ]
        for i, task in enumerate(tasks):
            r = task.get()
            results.append(r)
            status = "✅" if r["status"] == "ok" else "❌"
            detail = r.get("summary", r.get("error", ""))[:80]
            print(f"  [{i+1}/{len(videos)}] {status} {r['video']} "
                  f"({r['copy_sec']+r['run_sec']:.0f}s) {detail}")

    total_time = time.time() - total_start
    ok_count = sum(1 for r in results if r["status"] == "ok")

    print(f"\n{'='*70}")
    print(f"全部完成! {ok_count}/{len(results)} 成功 | 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"结果目录: {batch_dir}")
    print(f"{'='*70}")

    # 元信息
    meta = {
        "batch_name": batch_name,
        "workers": args.workers,
        "total_videos": len(videos),
        "total_time_sec": round(total_time, 1),
        "results": results,
    }
    (batch_dir / "_batch_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
