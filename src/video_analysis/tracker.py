"""多目标跨帧跟踪模块。

基于 ByteTrack 核心思想（IOU 匹配 + 匀速运动预测）的轻量实现。
纯 CPU，不依赖 GPU，仅需 numpy。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 工具函数
# ============================================================

def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> np.ndarray:
    """计算两组检测框的 IOU 矩阵。

    Args:
        bbox_a: (M, 4) xyxy 格式。
        bbox_b: (N, 4) xyxy 格式。

    Returns:
        (M, N) IOU 矩阵。
    """
    lt = np.maximum(bbox_a[:, None, :2], bbox_b[None, :, :2])
    rb = np.minimum(bbox_a[:, None, 2:], bbox_b[None, :, 2:])
    inter_wh = np.maximum(0.0, rb - lt)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area_a = (bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])
    area_b = (bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1])
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def _linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """解线性分配问题。

    Args:
        cost_matrix: (M, N) 成本矩阵（值越小越好）。

    Returns:
        (row_indices, col_indices) 匹配对。
    """
    M, N = cost_matrix.shape
    if M == 0 or N == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if HAS_SCIPY:
        row, col = linear_sum_assignment(cost_matrix)
        return row.astype(np.int64), col.astype(np.int64)
    else:
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """贪心匹配（scipy 不可用时的回退方案）。"""
    M, N = cost_matrix.shape
    cost = cost_matrix.copy()
    rows, cols = [], []
    used_cols: set = set()

    for _ in range(min(M, N)):
        min_val = np.inf
        min_pos = (-1, -1)
        for i in range(M):
            for j in range(N):
                if j not in used_cols and cost[i, j] < min_val:
                    min_val = cost[i, j]
                    min_pos = (i, j)
        if min_pos == (-1, -1):
            break
        rows.append(min_pos[0])
        cols.append(min_pos[1])
        used_cols.add(min_pos[1])

    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


# ============================================================
# Track 状态
# ============================================================

class TrackState:
    """单个跟踪目标的状态。

    使用匀速运动模型预测位置，指数平滑更新状态。
    """

    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> None:
        self.track_id = track_id
        self.bbox = bbox.copy()
        self.keypoints = keypoints.copy()
        self.confidence = confidence
        self.frame_index = frame_index
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.velocity = np.zeros(4, dtype=np.float32)

    def predict(self) -> np.ndarray:
        """预测当前位置（匀速模型）。"""
        return self.bbox + self.velocity

    def update(
        self,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        confidence: float,
        frame_index: int,
        smooth_alpha: float = 0.6,
    ) -> None:
        """用新检测更新状态（指数平滑）。"""
        new_velocity = bbox - self.bbox
        self.velocity = (
            smooth_alpha * self.velocity + (1 - smooth_alpha) * new_velocity
        )
        self.bbox = smooth_alpha * self.bbox + (1 - smooth_alpha) * bbox
        self.keypoints = keypoints.copy()
        self.confidence = confidence
        self.frame_index = frame_index
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """标记一帧未匹配。"""
        self.time_since_update += 1

    def is_confirmed(self, min_hits: int = 3) -> bool:
        return self.hits >= min_hits

    def is_deleted(self, max_lost: int = 30) -> bool:
        return self.time_since_update > max_lost

    def __repr__(self) -> str:
        return (
            f"TrackState(id={self.track_id}, age={self.age}, "
            f"hits={self.hits}, lost={self.time_since_update})"
        )


# ============================================================
# 多目标跟踪器
# ============================================================

class MultiObjectTracker:
    """ByteTrack 风格的多目标跟踪器。

    用法:
        tracker = MultiObjectTracker()
        tracks = tracker.update(keypoints, bboxes, confidences, frame_index)
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        min_hits: int = 3,
        max_lost: int = 30,
        max_age: int = 60,
        smooth_alpha: float = 0.6,
    ) -> None:
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh = track_low_thresh
        self._match_thresh = match_thresh
        self._min_hits = min_hits
        self._max_lost = max_lost
        self._max_age = max_age
        self._smooth_alpha = smooth_alpha

        self._tracks: List[TrackState] = []
        self._next_id = 1
        self._frame_count = 0

    # ---- 属性 ----

    @property
    def active_tracks(self) -> List[TrackState]:
        return [
            t for t in self._tracks
            if t.is_confirmed(self._min_hits) and not t.is_deleted(self._max_lost)
        ]

    @property
    def all_tracks(self) -> List[TrackState]:
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self.active_tracks)

    # ---- 核心更新 ----

    def update(
        self,
        keypoints: np.ndarray,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        frame_index: int,
    ) -> List[TrackState]:
        """更新跟踪器状态。

        Args:
            keypoints: (N, 17, 3) 关键点。
            bboxes: (N, 4) 检测框 xyxy。
            confidences: (N,) 置信度。
            frame_index: 帧序号。

        Returns:
            活跃 track 列表。
        """
        self._frame_count += 1
        N = len(confidences)

        if N == 0:
            for track in self._tracks:
                track.mark_missed()
            self._prune_tracks()
            return self.active_tracks

        # 分离高/低分检测
        high_mask = confidences >= self._track_high_thresh
        low_mask = (confidences >= self._track_low_thresh) & (~high_mask)
        high_idxs = np.where(high_mask)[0]
        low_idxs = np.where(low_mask)[0]

        # 预测所有 track 位置
        for track in self._tracks:
            track.predict()

        matched_track_idxs: set = set()
        matched_det_idxs: set = set()

        # 第一轮：高分检测 vs 所有 track
        if len(high_idxs) > 0 and len(self._tracks) > 0:
            self._match_round(
                high_idxs, bboxes, keypoints, confidences,
                list(range(len(self._tracks))),
                matched_track_idxs, matched_det_idxs,
                frame_index,
            )

        # 第二轮：低分检测 vs 丢失的 track
        lost_track_idxs = [
            i for i in range(len(self._tracks))
            if i not in matched_track_idxs
            and self._tracks[i].time_since_update > 0
        ]
        if len(low_idxs) > 0 and len(lost_track_idxs) > 0:
            self._match_round(
                low_idxs, bboxes, keypoints, confidences,
                lost_track_idxs,
                matched_track_idxs, matched_det_idxs,
                frame_index,
            )

        # 未匹配的高分检测 → 新 track
        for det_idx in high_idxs:
            if det_idx not in matched_det_idxs:
                self._create_track(
                    bboxes[det_idx], keypoints[det_idx],
                    float(confidences[det_idx]), frame_index,
                )

        # 未匹配的 track → 丢失
        for track_idx in range(len(self._tracks)):
            if track_idx not in matched_track_idxs:
                self._tracks[track_idx].mark_missed()

        self._prune_tracks()
        return self.active_tracks

    def _match_round(
        self,
        det_idxs: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        track_idxs: List[int],
        matched_track_idxs: set,
        matched_det_idxs: set,
        frame_index: int,
    ) -> None:
        """执行一轮 IOU 匹配。"""
        det_boxes = bboxes[det_idxs]
        track_boxes = np.stack([self._tracks[i].bbox for i in track_idxs])

        iou_matrix = _iou(det_boxes, track_boxes)
        cost_matrix = 1.0 - iou_matrix

        det_match, track_match = _linear_assignment(cost_matrix)

        for d_local, t_local in zip(det_match, track_match):
            if iou_matrix[d_local, t_local] >= self._match_thresh:
                d_abs = int(det_idxs[d_local])
                t_abs = track_idxs[t_local]

                self._tracks[t_abs].update(
                    bboxes[d_abs],
                    keypoints[d_abs],
                    float(confidences[d_abs]),
                    frame_index,
                    self._smooth_alpha,
                )
                matched_track_idxs.add(t_abs)
                matched_det_idxs.add(d_abs)

    def _create_track(
        self,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> TrackState:
        track = TrackState(
            track_id=self._next_id,
            bbox=bbox,
            keypoints=keypoints,
            confidence=confidence,
            frame_index=frame_index,
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _prune_tracks(self) -> None:
        self._tracks = [
            t for t in self._tracks
            if not t.is_deleted(self._max_lost)
            and t.age <= self._max_age
        ]

    def get_track_by_id(self, track_id: int) -> Optional[TrackState]:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    def export_active_keypoints(self) -> Dict[int, np.ndarray]:
        return {t.track_id: t.keypoints.copy() for t in self.active_tracks}

    def export_active_bboxes(self) -> Dict[int, np.ndarray]:
        return {t.track_id: t.bbox.copy() for t in self.active_tracks}

    def __repr__(self) -> str:
        return (
            f"MultiObjectTracker(tracks={len(self._tracks)}, "
            f"active={self.track_count}, next_id={self._next_id})"
        )
