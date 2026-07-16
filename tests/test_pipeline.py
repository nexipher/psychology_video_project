"""A1 全链路集成测试。

验证 VideoStream → PoseEstimator(Mock) → MultiObjectTracker → VideoFeatureExtractor → DailyAggregator
的端到端 Pipeline，全部在 CPU/Mock 模式下运行。
"""

import numpy as np
import pytest

from src.video_analysis.pose_estimator import PoseEstimator
from src.video_analysis.tracker import MultiObjectTracker
from src.video_analysis.feature_extractor import VideoFeatureExtractor
from src.video_analysis.aggregator import DailyAggregator
from src.utils.schema_validator import get_validator


class TestA1Pipeline:
    """A1 全链路集成测试（CPU Mock 模式）。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pose_estimator = PoseEstimator(mode="mock")
        self.tracker = MultiObjectTracker(min_hits=2, max_lost=10)
        self.extractor = VideoFeatureExtractor(
            window_size_sec=30.0,
            window_stride_sec=10.0,
            fps=15.0,
        )
        self.aggregator = DailyAggregator(fps=15.0)

    def test_full_pipeline_single_person(self):
        """单人 60 秒完整管线。"""
        np.random.seed(42)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        num_frames = 900  # 60s @ 15fps

        for i in range(num_frames):
            ts = i / 15.0

            # Step 1: Pose estimation (mock)
            pose_result = self.pose_estimator.estimate(frame)

            # Step 2: Tracking
            active = self.tracker.update(
                pose_result["keypoints"],
                pose_result["bboxes"],
                pose_result["confidences"],
                i,
            )

            # Step 3: Feature extraction
            # 聚合活跃 track 的关键点和 bboxes
            if active:
                kps_list = []
                bbox_list = []
                ids_list = []
                for t in active:
                    kps_list.append(t.keypoints)
                    bbox_list.append(t.bbox)
                    ids_list.append(t.track_id)
                agg_kps = np.stack(kps_list, axis=0) if kps_list else np.empty((0, 17, 3))
                agg_boxes = np.stack(bbox_list, axis=0) if bbox_list else np.empty((0, 4))
            else:
                agg_kps = np.empty((0, 17, 3), dtype=np.float32)
                agg_boxes = np.empty((0, 4), dtype=np.float32)
                ids_list = []

            window_result = self.extractor.process_frame(
                agg_kps, agg_boxes, ids_list, ts, i,
            )
            if window_result:
                self.aggregator.add_window(window_result)

        # Step 4: Daily aggregation
        daily = self.extractor.get_daily_summary("U001", "2026-07-15")
        assert daily["user_id"] == "U001"

        # Step 5: Schema validation
        val = get_validator()
        ok, errors = val.validate_daily_metrics(daily)
        assert ok, f"Pipeline output failed schema validation: {errors}"

    def test_full_pipeline_multi_person(self):
        """多人场景完整管线。"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for i in range(300):  # 20s
            ts = i / 15.0
            # Mock 2 persons by running estimate twice... actually mock generates varying N
            pose_result = self.pose_estimator.estimate(frame)

            if pose_result["keypoints"].shape[0] > 0:
                active = self.tracker.update(
                    pose_result["keypoints"],
                    pose_result["bboxes"],
                    pose_result["confidences"],
                    i,
                )

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
            else:
                agg_kps = np.empty((0, 17, 3), dtype=np.float32)
                agg_boxes = np.empty((0, 4), dtype=np.float32)
                ids_list = []

            result = self.extractor.process_frame(agg_kps, agg_boxes, ids_list, ts, i)
            if result:
                self.aggregator.add_window(result)

        daily = self.extractor.get_daily_summary("U002", "2026-07-15")
        val = get_validator()
        ok, _ = val.validate_daily_metrics(daily)
        assert ok

    def test_pipeline_skeleton_mode(self, skeleton_json_file):
        """使用 Skeleton 数据的验证模式管线。"""
        from src.video_analysis.data_loader import SkeletonLoader

        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)

        for frame_data in loader.frames():
            # 骨骼数据自带 keypoints，无需 PoseEstimator
            kps = frame_data.keypoints
            if kps.ndim == 2:
                kps = kps[np.newaxis, :, :]  # (K,3) → (1,K,3)
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32) if kps.shape[0] > 0 else np.empty((0, 4))
            confs = np.ones(kps.shape[0], dtype=np.float32) if kps.shape[0] > 0 else np.empty((0,))

            # 跟踪
            active = self.tracker.update(kps, bboxes, confs, frame_data.frame_index)

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

            result = self.extractor.process_frame(
                agg_kps, agg_boxes, ids_list,
                frame_data.timestamp, frame_data.frame_index,
            )
            if result:
                self.aggregator.add_window(result)

        daily = self.extractor.get_daily_summary("U003", "2026-07-15")
        val = get_validator()
        ok, errors = val.validate_daily_metrics(daily)
        assert ok, f"Skeleton pipeline schema validation failed: {errors}"
