"""多目标跟踪器测试。A1.6"""

import numpy as np
import pytest
from src.video_analysis.tracker import (
    MultiObjectTracker,
    TrackState,
    _iou,
    _linear_assignment,
    HAS_SCIPY,
)


class TestIOU:
    """IOU 计算测试。"""

    def test_perfect_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou = _iou(a, b)
        assert abs(iou[0, 0] - 1.0) < 0.01

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = _iou(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        iou = _iou(a, b)
        assert abs(iou[0, 0] - 0.143) < 0.01

    def test_matrix_shape(self):
        a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou = _iou(a, b)
        assert iou.shape == (2, 1)


class TestLinearAssignment:
    """线性分配测试。"""

    def test_simple_assignment(self):
        cost = np.array([[0.1, 0.8], [0.7, 0.3]], dtype=np.float32)
        rows, cols = _linear_assignment(cost)
        assert len(rows) == 2
        assert len(cols) == 2

    def test_empty_matrix(self):
        rows, cols = _linear_assignment(np.empty((0, 0)))
        assert len(rows) == 0


class TestTrackState:
    """TrackState 生命周期测试。"""

    def test_init(self):
        bbox = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox, kps, 0.9, 0)
        assert track.track_id == 1
        assert track.age == 1
        assert track.hits == 1

    def test_predict(self):
        bbox = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox, kps, 0.9, 0)
        track.velocity = np.array([10, 0, 10, 0], dtype=np.float32)
        predicted = track.predict()
        np.testing.assert_array_equal(predicted, [110, 100, 210, 300])

    def test_update_smooths(self):
        bbox1 = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox1, kps, 0.9, 0)
        bbox2 = np.array([120, 100, 220, 300], dtype=np.float32)
        track.update(bbox2, kps, 0.95, 1, smooth_alpha=0.5)
        # 平滑后应在 bbox1 和 bbox2 之间
        assert track.bbox[0] > 100
        assert track.bbox[0] < 120

    def test_mark_missed(self):
        bbox = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox, kps, 0.9, 0)
        track.mark_missed()
        assert track.time_since_update == 1

    def test_is_confirmed(self):
        bbox = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox, kps, 0.9, 0)
        assert not track.is_confirmed(min_hits=3)
        track.hits = 3
        assert track.is_confirmed(min_hits=3)

    def test_is_deleted(self):
        bbox = np.array([100, 100, 200, 300], dtype=np.float32)
        kps = np.zeros((17, 3), dtype=np.float32)
        track = TrackState(1, bbox, kps, 0.9, 0)
        assert not track.is_deleted(max_lost=30)
        track.time_since_update = 31
        assert track.is_deleted(max_lost=30)


class TestMultiObjectTracker:
    """MultiObjectTracker 核心功能测试。"""

    def test_init(self, tracker):
        assert tracker.track_count == 0
        assert len(tracker.all_tracks) == 0

    def test_single_person_tracking(self, tracker):
        """单目标连续跟踪。"""
        for frame_idx in range(20):
            x = 200 + frame_idx * 2
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, :2] = [[x + i * 5, 300 + i * 5] for i in range(17)]
            kps[0, :, 2] = 0.9
            bboxes = np.array([[x - 30, 180, x + 30, 480]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            active = tracker.update(kps, bboxes, confs, frame_idx)

        assert tracker.track_count >= 1
        assert len(active) >= 1

    def test_multi_person_tracking(self, tracker):
        """多目标跟踪，ID 不混淆。"""
        ids_seen = set()
        for frame_idx in range(30):
            p1_x = 200 + frame_idx
            p2_x = 400 - frame_idx * 0.5
            kps = np.zeros((2, 17, 3), dtype=np.float32)
            for k in range(17):
                kps[0, k] = [p1_x + k * 5, 250 + k * 5, 0.9]
                kps[1, k] = [p2_x + k * 5, 300 + k * 5, 0.85]
            bboxes = np.array([
                [p1_x - 30, 130, p1_x + 30, 430],
                [p2_x - 30, 180, p2_x + 30, 480],
            ], dtype=np.float32)
            confs = np.array([0.9, 0.85], dtype=np.float32)
            active = tracker.update(kps, bboxes, confs, frame_idx)
            for t in active:
                ids_seen.add(t.track_id)

        assert len(ids_seen) >= 2  # 两个人至少有两个不同的 track_id

    def test_no_detections_preserves_tracks(self, tracker):
        """无检测帧不应立即删除 track。"""
        # 先建立 track
        for i in range(5):
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            tracker.update(kps, bboxes, confs, i)

        # 空帧
        for i in range(2):
            tracker.update(
                np.empty((0, 17, 3)), np.empty((0, 4)),
                np.empty((0,)), i + 5,
            )

        # Track 不应被删除（max_lost=10）
        assert len(tracker.all_tracks) > 0

    def test_export_active_keypoints(self, tracker):
        for i in range(10):
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, :2] = [[200 + i, 300] for _ in range(17)]
            kps[0, :, 2] = 0.9
            bboxes = np.array([[170, 180, 230, 480]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            tracker.update(kps, bboxes, confs, i)

        exported = tracker.export_active_keypoints()
        assert len(exported) >= 1
        for tid, kps in exported.items():
            assert kps.shape == (17, 3)

    def test_export_active_bboxes(self, tracker):
        for i in range(10):
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            tracker.update(kps, bboxes, confs, i)

        exported = tracker.export_active_bboxes()
        assert len(exported) >= 1

    def test_reset(self, tracker):
        for i in range(5):
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            tracker.update(kps, bboxes, confs, i)

        tracker.reset()
        assert tracker.track_count == 0
        assert len(tracker.all_tracks) == 0

    def test_get_track_by_id(self, tracker):
        for i in range(10):
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            tracker.update(kps, bboxes, confs, i)

        track = tracker.get_track_by_id(1)
        assert track is not None
        assert track.track_id == 1
