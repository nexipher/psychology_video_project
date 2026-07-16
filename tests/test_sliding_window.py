"""滑动窗口数据结构测试。A1.7"""

import pytest
from src.video_analysis.sliding_window import SlidingWindow, TimedSlidingWindow


class TestSlidingWindow:
    """SlidingWindow 基础功能测试。"""

    def test_append_and_count(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        sw.append({"timestamp": 1.0, "val": "a"})
        sw.append({"timestamp": 2.0, "val": "b"})
        assert sw.get_count() == 2
        assert len(sw) == 2

    def test_duration(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        sw.append({"timestamp": 0.0})
        sw.append({"timestamp": 5.0})
        sw.append({"timestamp": 10.0})
        assert sw.get_duration() == 10.0

    def test_evict_by_duration(self):
        """时间过期的记录应被淘汰。"""
        sw = SlidingWindow(max_size=100, max_duration_sec=5.0)
        sw.append({"timestamp": 0.0})
        sw.append({"timestamp": 3.0})
        sw.append({"timestamp": 6.0})  # 此时 0.0 的记录应被淘汰
        records = sw.get_all()
        timestamps = [r["timestamp"] for r in records]
        assert 0.0 not in timestamps
        assert len(records) == 2

    def test_evict_by_size(self):
        """超过最大容量的记录应被淘汰。"""
        sw = SlidingWindow(max_size=3, max_duration_sec=100.0)
        for i in range(5):
            sw.append({"timestamp": float(i)})
        assert sw.get_count() == 3
        records = sw.get_all()
        assert records[0]["timestamp"] == 2.0

    def test_get_field_values(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        sw.append({"timestamp": 1.0, "score": 0.5})
        sw.append({"timestamp": 2.0, "score": 0.8})
        scores = sw.get_field_values("score")
        assert scores == [0.5, 0.8]

    def test_aggregate(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        for i in range(5):
            sw.append({"timestamp": float(i), "value": i})
        total = sw.aggregate(lambda records: sum(r["value"] for r in records))
        assert total == 10

    def test_clear(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        sw.append({"timestamp": 1.0})
        sw.clear()
        assert sw.get_count() == 0
        assert sw.get_duration() == 0.0

    def test_empty_window(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        assert sw.get_count() == 0
        assert sw.get_duration() == 0.0
        assert sw.get_all() == []

    def test_thread_safety(self):
        """多线程并发追加不抛异常。"""
        import threading
        sw = SlidingWindow(max_size=500, max_duration_sec=100.0)

        def add_records(start: int):
            for i in range(100):
                sw.append({"timestamp": float(start + i)})

        threads = [threading.Thread(target=add_records, args=(i * 100,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sw.get_count() > 0

    def test_missing_timestamp_raises(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=100.0)
        with pytest.raises(KeyError):
            sw.append({"no_timestamp": 1.0})

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            SlidingWindow(max_size=0, max_duration_sec=100)
        with pytest.raises(ValueError):
            SlidingWindow(max_size=10, max_duration_sec=0)

    def test_repr(self):
        sw = SlidingWindow(max_size=10, max_duration_sec=60.0)
        sw.append({"timestamp": 1.0})
        r = repr(sw)
        assert "SlidingWindow" in r


class TestTimedSlidingWindow:
    """TimedSlidingWindow 周期回调测试。"""

    def test_emit_callback(self):
        emitted = []

        def on_emit(records):
            emitted.append(len(records))

        tsw = TimedSlidingWindow(
            max_size=100,
            max_duration_sec=100.0,
            emit_interval_sec=5.0,
            on_emit=on_emit,
        )
        tsw.append({"timestamp": 0.0})
        assert len(emitted) == 0
        tsw.append({"timestamp": 6.0})  # 超过 5s
        assert len(emitted) == 1
        assert emitted[0] == 2

    def test_no_callback_on_first(self):
        tsw = TimedSlidingWindow(
            max_size=100, max_duration_sec=100.0,
            emit_interval_sec=5.0,
        )
        tsw.append({"timestamp": 0.0})
        # 不应抛异常（on_emit 为 None）
