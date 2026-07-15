"""通用滑动窗口数据结构。

固定时间窗长度，O(1) 插入/淘汰，线程安全。
纯 CPU 实现，用于时序特征实时计算。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional


class SlidingWindow:
    """固定容量 + 固定时间跨度的滑动窗口。

    每条记录附带宽泛的时间戳 (timestamp)，窗口按两个维度约束:
    1. max_size: 最大记录数
    2. max_duration_sec: 最大时间跨度

    任一约束触发即淘汰最早记录。

    Thread-safe.
    """

    def __init__(
        self,
        max_size: int = 300,
        max_duration_sec: float = 300.0,
        timestamp_key: str = "timestamp",
    ) -> None:
        """
        Args:
            max_size: 最大记录数。
            max_duration_sec: 最大时间跨度（秒）。
            timestamp_key: 每条记录中时间戳字段的键名。
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if max_duration_sec <= 0:
            raise ValueError("max_duration_sec must be positive")

        self._max_size = max_size
        self._max_duration_sec = max_duration_sec
        self._timestamp_key = timestamp_key
        self._deque: deque = deque()
        self._lock = threading.Lock()

    # ---- 属性 ----

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def max_duration_sec(self) -> float:
        return self._max_duration_sec

    @property
    def timestamp_key(self) -> str:
        return self._timestamp_key

    # ---- 核心操作 ----

    def append(self, record: Dict[str, Any]) -> None:
        """添加一条记录，自动淘汰过期条目。

        Args:
            record: 记录字典，必须包含 timestamp_key 对应的键。

        Raises:
            KeyError: 记录缺少时间戳字段。
        """
        ts = record[self._timestamp_key]
        with self._lock:
            self._deque.append(record)
            self._evict(ts)

    def extend(self, records: List[Dict[str, Any]]) -> None:
        """批量添加记录。"""
        if not records:
            return
        with self._lock:
            for rec in records:
                self._deque.append(rec)
            latest_ts = records[-1][self._timestamp_key]
            self._evict(latest_ts)

    def _evict(self, current_ts: float) -> None:
        """淘汰超出约束的记录。"""
        # 按时间淘汰
        cutoff = current_ts - self._max_duration_sec
        while self._deque and self._deque[0][self._timestamp_key] < cutoff:
            self._deque.popleft()

        # 按大小淘汰
        while len(self._deque) > self._max_size:
            self._deque.popleft()

    # ---- 查询 ----

    def get_all(self) -> List[Dict[str, Any]]:
        """返回窗口内所有记录的副本。"""
        with self._lock:
            return list(self._deque)

    def get_count(self) -> int:
        """返回当前记录数。"""
        with self._lock:
            return len(self._deque)

    def get_duration(self) -> float:
        """返回窗口内实际时间跨度（秒）。"""
        with self._lock:
            if len(self._deque) < 2:
                return 0.0
            return self._deque[-1][self._timestamp_key] - self._deque[0][self._timestamp_key]

    def get_field_values(self, key: str) -> List[Any]:
        """提取窗口内所有记录指定字段的值列表。"""
        with self._lock:
            return [rec.get(key) for rec in self._deque]

    def aggregate(
        self,
        func: Callable[[List[Dict[str, Any]]], Any],
    ) -> Any:
        """对窗口内所有记录执行自定义聚合。

        Args:
            func: 聚合函数，接收记录列表，返回聚合结果。
        """
        with self._lock:
            return func(list(self._deque))

    def clear(self) -> None:
        """清空窗口。"""
        with self._lock:
            self._deque.clear()

    def __len__(self) -> int:
        return self.get_count()

    def __repr__(self) -> str:
        n = self.get_count()
        dur = self.get_duration()
        return (
            f"SlidingWindow(size={n}/{self._max_size}, "
            f"duration={dur:.1f}/{self._max_duration_sec:.0f}s)"
        )


class TimedSlidingWindow(SlidingWindow):
    """增强版滑动窗口：支持按固定周期触发回调（如每 60 秒输出一次聚合指标）。

    用于视频特征管线中实时输出窗口级特征。
    """

    def __init__(
        self,
        max_size: int = 300,
        max_duration_sec: float = 300.0,
        timestamp_key: str = "timestamp",
        emit_interval_sec: float = 60.0,
        on_emit: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> None:
        super().__init__(max_size, max_duration_sec, timestamp_key)
        self._emit_interval_sec = emit_interval_sec
        self._on_emit = on_emit
        self._last_emit_ts: Optional[float] = None

    def append(self, record: Dict[str, Any]) -> None:
        """添加记录，若到达 emit 周期则触发回调。"""
        super().append(record)
        ts = record[self._timestamp_key]

        if self._last_emit_ts is None:
            self._last_emit_ts = ts
            return

        if ts - self._last_emit_ts >= self._emit_interval_sec:
            self._last_emit_ts = ts
            if self._on_emit is not None:
                self._on_emit(self.get_all())
